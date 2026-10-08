import hashlib
import gzip
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import assets
import production


def tensor(path):
    header = json.dumps({'weight': {'dtype': 'U8', 'shape': [4], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'abcd')


def checkpoint(path):
    path.mkdir(exist_ok=True)
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        (path / name).write_text('{}')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model.safetensors'}}))
    tensor(path / 'model.safetensors')
    (path / 'dense-mtp-build-report.json').write_text(json.dumps({
        'tier': 'drafter-dense', 'group_size': 32, 'modules': [], 'mtp_modules': [{}] * 9}))


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.ple, self.vocab = [self.root / x for x in ['model', 'ple', 'vocab.txt']]

    def prepare_existing(self):
        checkpoint(self.model)
        self.ple.mkdir()
        for i in range(5, 38):
            tensor(self.ple / f'model-{i:05d}-of-00131.safetensors')
        self.vocab.write_bytes(b'123\n')
        return patch('assets.VOCAB_SHA', hashlib.sha256(self.vocab.read_bytes()).hexdigest())

    def test_complete_assets_require_no_network(self):
        with self.prepare_existing(), patch('assets.fetch_recipe') as recipe, patch('assets.fetch_snapshot') as download:
            assets.ensure_assets(self.model, self.ple, self.vocab, 'image')
        recipe.assert_not_called()
        download.assert_not_called()

    def test_missing_shard_and_truncation_are_detected(self):
        checkpoint(self.model)
        self.assertTrue(assets.model_complete(self.model))
        (self.model / 'model.safetensors').write_bytes(b'partial')
        self.assertFalse(assets.model_complete(self.model))
        (self.model / 'model.safetensors').unlink()
        self.assertFalse(assets.model_complete(self.model))

    def test_offline_missing_does_not_download(self):
        with patch('assets.fetch_recipe') as recipe:
            with self.assertRaisesRegex(SystemExit, 'no-download'):
                assets.ensure_assets(self.model, self.ple, self.vocab, 'image', offline=True)
        recipe.assert_not_called()
        self.assertFalse(self.model.exists())

    def test_existing_partial_model_is_preserved(self):
        self.model.mkdir()
        marker = self.model / 'user-file'
        marker.write_text('preserve')
        with self.assertRaisesRegex(SystemExit, 'will not be overwritten'):
            assets.ensure_assets(self.model, self.ple, self.vocab, 'image')
        self.assertEqual(marker.read_text(), 'preserve')

    def test_missing_model_downloads_pinned_base_and_builds(self):
        with self.prepare_existing():
            # Keep valid PLE/vocabulary; select a new model destination.
            output = self.root / 'new-model'
            output.mkdir()  # An empty user-created destination is supported.
            def build(cmd, **kwargs):
                self.assertEqual(cmd[-1], '--run')
                self.assertEqual(kwargs['env']['IMAGE'], 'pinned-image')
                checkpoint(Path(kwargs['env']['OUT_DIR']))
            with patch('assets.fetch_recipe', return_value=self.root), patch('assets.fetch_snapshot') as download, patch('assets.subprocess.run', side_effect=build):
                assets.ensure_assets(output, self.ple, self.vocab, 'pinned-image')
            download.assert_called_once_with(assets.BASE_REPO, assets.BASE_REV, self.root / f'.qwen38-base-{assets.BASE_REV}')
            self.assertTrue(assets.model_complete(output))

    def test_failed_build_never_publishes_model(self):
        with patch('assets.fetch_recipe', return_value=self.root), patch('assets.fetch_snapshot'), patch('assets.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'build')):
            with self.assertRaises(subprocess.CalledProcessError):
                assets.ensure_assets(self.model, self.ple, self.vocab, 'image')
        self.assertFalse(self.model.exists())

    def test_missing_vocabulary_is_verified_and_installed(self):
        with self.prepare_existing():
            data = self.vocab.read_bytes()
            self.vocab.unlink()
            recipe_path = self.root / 'recipe/config/v16b'
            recipe_path.mkdir(parents=True)
            (recipe_path / 'draft-vocab-ids-K65536.txt.gz').write_bytes(gzip.compress(data))
            with patch('assets.fetch_recipe', return_value=self.root), patch('assets.fetch_snapshot') as download:
                assets.ensure_assets(self.model, self.ple, self.vocab, 'image')
            self.assertEqual(self.vocab.read_bytes(), data)
            download.assert_not_called()

    def test_ple_missing_shard_triggers_pinned_download(self):
        with self.prepare_existing():
            shard = self.ple / 'model-00005-of-00131.safetensors'
            shard.unlink()
            with patch('assets.fetch_snapshot', side_effect=lambda *args: tensor(shard)) as download:
                assets.ensure_assets(self.model, self.ple, self.vocab, 'image')
            download.assert_called_once_with(assets.PLE_REPO, assets.PLE_REV, self.ple)

    def test_prepare_failure_prevents_serving_container_creation(self):
        replies = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0, stdout=''), subprocess.CompletedProcess([], 0)]
        with patch('sys.argv', ['production.py', '--create']), patch('production.subprocess.run', side_effect=replies) as run, patch('production.ensure_assets', side_effect=RuntimeError('download failed')):
            with self.assertRaisesRegex(RuntimeError, 'download failed'):
                production.main()
        self.assertEqual(run.call_count, 3)

    def test_missing_image_prevents_download(self):
        calls = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0, stdout=''), subprocess.CalledProcessError(1, 'inspect')]
        with patch('sys.argv', ['production.py', '--create']), patch('production.subprocess.run', side_effect=calls), patch('production.ensure_assets') as ensure:
            with self.assertRaises(subprocess.CalledProcessError):
                production.main()
        ensure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
