import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

import production


class ProductionTests(unittest.TestCase):
    def test_dry_run_never_contacts_docker(self):
        with patch('sys.argv', ['production.py', '--dry-run']), patch('production.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()) as out:
            production.main()
        run.assert_not_called()
        self.assertIn('docker create', out.getvalue())

    def test_existing_container_is_not_modified(self):
        replies = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0, stdout='qwen38-flash\n')]
        with patch('sys.argv', ['production.py', '--run']), patch('production.subprocess.run', side_effect=replies) as run:
            with self.assertRaisesRegex(SystemExit, 'refusing replacement'):
                production.main()
        self.assertEqual(run.call_count, 2)

    def test_daemon_failure_does_not_create(self):
        with patch('sys.argv', ['production.py', '--create']), patch('production.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'docker')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                production.main()
        self.assertEqual(run.call_count, 1)

    def test_selected_profile(self):
        argv = json.loads(production.MANIFEST.read_text())
        for flag, value in [('--max-num-seqs', '4'), ('--max-model-len', '262144'), ('--kv-cache-memory-bytes', '22g'), ('--tool-call-parser', 'qwen3_xml')]:
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn('VLLM_PREFIX_CACHE_RETENTION_INTERVAL=6400', argv)
        spec = json.loads(argv[argv.index('--speculative-config') + 1])
        self.assertEqual(spec['num_speculative_tokens'], 3)


if __name__ == '__main__':
    unittest.main()
