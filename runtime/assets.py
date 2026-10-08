"""Fetch pinned public assets and build the production dense-MTP checkpoint."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

BASE_REPO = 'Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid'
BASE_REV = '8b82f0b7abe3d1150a7827d298c75e86267636ae'
PLE_REPO = 'Saren/Qwen3.8-Flash-Next-ple-table-fp8'
PLE_REV = '50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14'
RECIPE_URL = 'https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast.git'
RECIPE_REV = '0c391a3e74b6a775cfe248691ca7fd855b1876a5'
VOCAB_SHA = 'f667de23949073f1d579bd2f702a18dc733554437c1e8c57f2ec38160f257075'


def tensor_file_complete(path):
    """Detect incomplete safetensors without reading hundreds of GB into RAM."""
    try:
        with path.open('rb') as file:
            length = struct.unpack('<Q', file.read(8))[0]
            if not 2 <= length <= 100_000_000:
                return False
            header = json.loads(file.read(length))
        ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        return bool(ends) and path.stat().st_size == 8 + length + max(ends)
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def model_complete(path):
    try:
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
            json.loads((path / name).read_text())
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        shards = set(index['weight_map'].values())
        if not shards or any(Path(s).name != s for s in shards):
            return False
        report = json.loads((path / 'dense-mtp-build-report.json').read_text())
        return (report['tier'] == 'drafter-dense' and report['group_size'] == 32
                and not report['modules'] and len(report['mtp_modules']) == 9
                and all(tensor_file_complete(path / s) for s in shards))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ple_complete(path):
    return all(tensor_file_complete(path / f'model-{i:05d}-of-00131.safetensors')
               for i in range(5, 38))


def vocab_complete(path):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == VOCAB_SHA


def fetch_snapshot(repo, revision, destination):
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise SystemExit('Missing download dependency: install huggingface_hub in this Python environment')
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    remaining = sum(f.size for f in info.siblings if f.size and
                    (not (destination / f.rfilename).is_file()
                     or (destination / f.rfilename).stat().st_size != f.size))
    if shutil.disk_usage(destination).free < remaining + 6 * 1024**3:
        raise SystemExit(f'Insufficient free space at {destination}: need about {remaining / 1024**3 + 6:.1f} GiB')
    print(f'Downloading {repo}@{revision} into {destination}', flush=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination), max_workers=4)


def fetch_recipe(parent):
    # Fresh private directory: never execute a pre-existing modified checkout.
    recipe = Path(tempfile.mkdtemp(prefix='.ultrafast-recipe-', dir=parent))
    subprocess.run(['git', 'init', '-q', str(recipe)], check=True)
    subprocess.run(['git', '-C', str(recipe), 'fetch', '--depth=1', RECIPE_URL, RECIPE_REV], check=True, timeout=300)
    subprocess.run(['git', '-C', str(recipe), 'checkout', '-q', '--detach', RECIPE_REV], check=True)
    return recipe


def ensure_assets(model, ple, vocab, image, offline=False):
    if model_complete(model) and ple_complete(ple) and vocab_complete(vocab):
        return
    if offline:
        raise SystemExit('Missing/incomplete model, PLE or draft vocabulary; rerun without --no-download')
    model.parent.mkdir(parents=True, exist_ok=True)
    # Serialize preparation for this model; HF provides per-file download locks.
    with (model.parent / f'.{model.name}.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        needs_model = not model_complete(model)
        needs_vocab = not vocab_complete(vocab)
        if needs_model and model.exists() and (not model.is_dir() or any(model.iterdir())):
            raise SystemExit(f'Existing model directory is incomplete or incompatible: {model}. Preserve/move it before rebuilding; it will not be overwritten.')
        if needs_vocab and vocab.exists():
            raise SystemExit(f'Draft vocabulary differs from the pinned production file: {vocab}; refusing overwrite')
        recipe = fetch_recipe(model.parent) if needs_model or needs_vocab else None
        if needs_model:
            base = model.parent / f'.qwen38-base-{BASE_REV}'
            fetch_snapshot(BASE_REPO, BASE_REV, base)
            # The upstream wrapper requires a nonexistent sibling destination.
            staging_parent = Path(tempfile.mkdtemp(prefix='.dense-mtp-', dir=model.parent))
            staging = staging_parent.with_name(staging_parent.name + '-output')
            env = dict(os.environ, MODELS_ROOT=str(model.parent), MODEL_DIR=str(base),
                       OUT_DIR=str(staging), IMAGE=image)
            print('Building and verifying dense-MTP g32 (CPU-only, 8 GiB helper limit)', flush=True)
            subprocess.run(['bash', str(recipe / 'recipe/build/model/build.sh'), '--run'],
                           check=True, env=env, timeout=1800)
            if not model_complete(staging):
                raise SystemExit(f'Built model failed validation; preserved at {staging}')
            if model.exists() and (not model.is_dir() or any(model.iterdir())):
                raise SystemExit('Model destination appeared during preparation; refusing overwrite')
            staging.rename(model)
            staging_parent.rmdir()
        if not ple_complete(ple):
            fetch_snapshot(PLE_REPO, PLE_REV, ple)
            if not ple_complete(ple):
                raise SystemExit('PLE download is incomplete; rerun to resume')
        if needs_vocab:
            data = gzip.decompress((recipe / 'recipe/config/v16b/draft-vocab-ids-K65536.txt.gz').read_bytes())
            if hashlib.sha256(data).hexdigest() != VOCAB_SHA:
                raise SystemExit('Downloaded draft vocabulary failed SHA256 verification')
            vocab.parent.mkdir(parents=True, exist_ok=True)
            # Publish only the fully verified bytes; never replace an existing file.
            fd, temporary = tempfile.mkstemp(prefix='.draft-vocab-', dir=vocab.parent)
            try:
                with os.fdopen(fd, 'wb') as file:
                    file.write(data)
                os.link(temporary, vocab)
            finally:
                Path(temporary).unlink()
