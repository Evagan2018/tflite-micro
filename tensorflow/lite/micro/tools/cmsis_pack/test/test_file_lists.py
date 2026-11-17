# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Offline tests for inventory validation and the upstream subprocess boundary."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'file_lists.py'
SPEC = importlib.util.spec_from_file_location('file_lists', SCRIPT)
file_lists = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(file_lists)


class FileListTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.base = Path(self.directory.name)
    self.root = self.base / 'checkout'
    self.root.mkdir()
    (self.root / 'src').mkdir()
    (self.root / 'src' / 'a.cc').write_text('// source', encoding='utf-8')
    (self.root / 'src' / 'b.h').write_text('// header', encoding='utf-8')

  def generator(self, body):
    path = self.root / file_lists.GENERATOR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')

  def cli(self, *extra):
    # A real subprocess launched outside the selected checkout.
    return subprocess.run(
      [sys.executable, str(SCRIPT), '--tflm-root', str(self.root), *extra],
      cwd=self.base, capture_output=True, text=True, check=False,
    )

  def test_normalizes_and_deduplicates_relative_and_absolute_paths(self):
    raw = f'src/b.h ./src/a.cc src/../src/a.cc {self.root / "src/b.h"}'
    # Absolute names in an upstream whitespace-delimited list cannot have spaces.
    if any(c.isspace() for c in str(self.root)):
      raw = 'src/b.h ./src/a.cc src/../src/a.cc src/b.h'
    self.assertEqual(
      file_lists.normalize_sources(raw, self.root), ['src/a.cc', 'src/b.h']
    )

  def test_rejects_empty_missing_directory_and_outside_paths(self):
    (self.base / 'outside.cc').write_text('// external', encoding='utf-8')
    for raw in ('', 'src/missing.cc', 'src', '../outside.cc'):
      with self.subTest(raw=raw):
        with self.assertRaises(file_lists.FileListError):
          file_lists.normalize_sources(raw, self.root)

  def test_rejects_external_symlink(self):
    external = self.base / 'outside.cc'
    external.write_text('// external', encoding='utf-8')
    try:
      (self.root / 'link.cc').symlink_to(external)
    except OSError:
      self.skipTest('Symlinks are unavailable for this user/platform')
    with self.assertRaisesRegex(file_lists.FileListError, 'outside TFLM root'):
      file_lists.normalize_sources('link.cc', self.root)

  def test_cli_invokes_generator_from_checkout_with_reference_options(self):
    self.generator(
      'import pathlib, sys\n'
      'assert pathlib.Path("src/a.cc").is_file()\n'
      'assert "--no_copy" in sys.argv\n'
      'assert "--print_src_files" in sys.argv\n'
      'assert "--makefile_options=TARGET=cortex_m_generic '
      'TARGET_ARCH=cortex-m7" in sys.argv\n'
      'print("src/b.h src/a.cc src/b.h")\n'
    )
    result = self.cli()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout, 'src/a.cc\nsrc/b.h\n')
    output = self.base / 'new folder' / 'reference.txt'
    result = self.cli('--out', str(output))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout, '')
    self.assertEqual(output.read_bytes(), b'src/a.cc\nsrc/b.h\n')

  def test_generator_failure_preserves_existing_inventory(self):
    self.generator('import sys\nprint("download failed", file=sys.stderr)\nsys.exit(7)\n')
    output = self.base / 'reference.txt'
    output.write_text('previous inventory\n', encoding='utf-8')
    result = self.cli('--out', str(output))
    self.assertEqual(result.returncode, 1)
    self.assertIn('download failed', result.stderr)
    self.assertIn('exit 7', result.stderr)
    self.assertEqual(result.stdout, '')
    self.assertEqual(output.read_text(), 'previous inventory\n')

  def test_invalid_source_does_not_create_inventory(self):
    self.generator('print("src/a.cc missing.cc")\n')
    output = self.base / 'reference.txt'
    result = self.cli('--out', str(output))
    self.assertEqual(result.returncode, 1)
    self.assertIn('cannot be resolved', result.stderr)
    self.assertFalse(output.exists())

  def test_missing_root_and_generator_have_actionable_errors(self):
    with self.assertRaisesRegex(file_lists.FileListError, 'does not exist'):
      file_lists.generate_reference_sources(self.base / 'missing')
    result = self.cli()
    self.assertEqual(result.returncode, 1)
    self.assertIn('generator is missing', result.stderr)


if __name__ == '__main__':
  unittest.main()
