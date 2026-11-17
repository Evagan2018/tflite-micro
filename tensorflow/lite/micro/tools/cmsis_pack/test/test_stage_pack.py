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
"""Payload integrity, ownership and compilation-selection regression tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage_pack
sys.path.pop(0)


class StageTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.base = Path(self.temp.name)
    self.root = self.base / 'root'
    self.root.mkdir()
    self.inputs = self.base / 'inventories'
    self.inputs.mkdir()
    self.output = self.base / 'stage'
    self.micro = stage_pack.MICRO
    self.downloads = stage_pack.DOWNLOADS
    common = ['LICENSE', self.micro + 'micro_interpreter.cc',
              self.micro + 'system_setup.cc', 'signal/src/fft.cc',
              self.downloads + 'flatbuffers/LICENSE',
              self.downloads + 'flatbuffers/include/flatbuffers/base.h']
    self.lists = {
      'reference': common + [self.micro + 'kernels/conv.cc'],
      'cmsis_nn': common + [self.micro + 'kernels/cmsis_nn/conv.cc',
                            self.downloads + 'cmsis_nn/Source/conv.c'],
    }
    self.config = {'compatibility': {'config_file_versions': {
      'system_setup.cpp': '1.0.0', 'micro_time.cpp': '1.0.0', 'debug_log.cpp': '1.0.0'}}}
    self.git = patch.object(stage_pack, 'git_state', return_value={'commit': 'a' * 40})
    self.git.start()
    self.addCleanup(self.git.stop)

  def write_inputs(self):
    for name in stage_pack.TESTING_SUPPORT:
      source = self.root / name
      source.parent.mkdir(parents=True, exist_ok=True)
      source.write_text('testing fixture ' + name)
    records = {}
    for variant, names in self.lists.items():
      names = sorted(set(names))
      hashes = {}
      for name in names:
        source = self.root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(('fixture ' + name).encode())
        hashes[name] = stage_pack.digest(source)
      inventory = self.inputs / f'{variant}-sources.txt'
      inventory.write_text('\n'.join(names) + '\n', encoding='utf-8', newline='\n')
      records[variant] = {'file': inventory.name, 'sha256': stage_pack.digest(inventory),
                          'count': len(names), 'source_files_sha256': hashes}
    provenance = {'schema_version': 2, 'runtime': {'commit': 'a' * 40},
                  'inventories': records, 'config': {'contents': self.config}}
    (self.inputs / 'provenance.json').write_text(json.dumps(provenance), encoding='utf-8')

  def run_stage(self):
    return stage_pack.stage(self.root, self.inputs, self.output)

  def test_shared_files_copied_once_and_external_files_excluded(self):
    self.write_inputs()
    result = self.run_stage()
    self.assertEqual(len(result['files']), 8 + len(stage_pack.TESTING_SUPPORT))
    self.assertEqual(len(list((self.output / 'payload').rglob('conv.c'))), 0)
    self.assertTrue((self.output / 'payload/third_party/flatbuffers/LICENSE').exists())
    units = result['selections']['cmsis_nn']['compile_units']
    self.assertIn((self.micro + 'kernels/cmsis_nn/conv.cc').casefold(), units)
    self.assertNotIn((self.micro + 'kernels/conv.cc').casefold(), units)
    self.assertEqual(len(units), len(set(units)))
    config = next(e for e in result['files'] if e['source'].endswith('system_setup.cc'))
    self.assertEqual(config['config_version'], '1.0.0')
    self.assertEqual(config['owners'], ['kernel_utils'])
    for entry in result['files']:
      self.assertEqual(stage_pack.digest(self.output / 'payload' / entry['destination']), entry['sha256'])

  def test_source_tampering_leaves_no_output(self):
    self.write_inputs()
    (self.root / 'LICENSE').write_text('changed')
    with self.assertRaisesRegex(stage_pack.BuildError, 'Source hash mismatch'):
      self.run_stage()
    self.assertFalse(self.output.exists())

  def test_inventory_tampering_leaves_no_output(self):
    self.write_inputs()
    (self.inputs / 'reference-sources.txt').write_text('missing.cc\n')
    with self.assertRaisesRegex(stage_pack.BuildError, 'Inventory hash mismatch'):
      self.run_stage()
    self.assertFalse(self.output.exists())

  def test_unknown_path_is_not_implicitly_bundled(self):
    self.lists['reference'].append('unknown.cc')
    self.write_inputs()
    with self.assertRaisesRegex(stage_pack.BuildError, 'Unclassified'):
      self.run_stage()

  def test_destination_collision_rejected(self):
    self.lists['reference'].append('third_party/flatbuffers/LICENSE')
    self.write_inputs()
    with self.assertRaisesRegex(stage_pack.BuildError, 'Destination collision'):
      self.run_stage()

  def test_conflicting_kernel_implementations_rejected(self):
    self.lists['cmsis_nn'].append(self.micro + 'kernels/conv.cc')
    self.write_inputs()
    with self.assertRaisesRegex(stage_pack.BuildError, 'Conflicting kernel'):
      self.run_stage()

  def test_shared_component_variation_requires_review(self):
    self.lists['cmsis_nn'].append('signal/src/extra.cc')
    self.write_inputs()
    with self.assertRaisesRegex(stage_pack.BuildError, 'Shared component differs'):
      self.run_stage()

  def test_missing_bundled_license_rejected(self):
    for names in self.lists.values():
      names.remove(self.downloads + 'flatbuffers/LICENSE')
    self.write_inputs()
    with self.assertRaisesRegex(stage_pack.BuildError, 'Missing license'):
      self.run_stage()

  def test_omitted_downloaded_license_is_added_with_provenance(self):
    for names in self.lists.values():
      names.append(self.downloads + 'ruy/ruy/profiler/instrumentation.h')
    self.write_inputs()
    license_path = self.root / (self.downloads + 'ruy/LICENSE')
    license_path.write_text('ruy license', encoding='utf-8')
    result = self.run_stage()
    self.assertTrue((self.output / 'payload/third_party/ruy/LICENSE').is_file())
    entry = next(e for e in result['supplemental_inputs'] if e['source'].endswith('ruy/LICENSE'))
    self.assertEqual(entry['sha256'], stage_pack.digest(license_path))
    self.assertEqual(sorted(entry['variants']), ['cmsis_nn', 'reference'])
    self.assertNotIn(self.downloads + 'ruy/LICENSE',
                     (self.inputs / 'reference-sources.txt').read_text())

  def test_existing_output_preserved(self):
    self.write_inputs()
    self.output.mkdir()
    sentinel = self.output / 'keep.txt'
    sentinel.write_text('keep')
    with self.assertRaisesRegex(stage_pack.BuildError, 'Output already exists'):
      self.run_stage()
    self.assertEqual(sentinel.read_text(), 'keep')

  def test_testing_support_has_provenance_and_shared_ownership(self):
    self.write_inputs()
    result = self.run_stage()
    for name in stage_pack.TESTING_SUPPORT:
      entry = next(e for e in result['files'] if e['source'] == name)
      self.assertEqual(entry['owners'], ['testing'])
      self.assertEqual(sorted(entry['variants']), ['cmsis_nn', 'reference'])
      self.assertEqual(entry['sha256'], stage_pack.digest(self.root / name))
      self.assertIn(name, result['components']['testing']['files'])
    self.assertFalse(any(e['source'].endswith('_test.cc') for e in result['files']))

  def test_missing_testing_support_fails_without_partial_output(self):
    self.write_inputs()
    (self.root / stage_pack.TESTING_SUPPORT[0]).unlink()
    with self.assertRaises(FileNotFoundError):
      self.run_stage()
    self.assertFalse(self.output.exists())

  def test_already_inventoried_testing_support_is_not_duplicated(self):
    for sources in self.lists.values():
      sources.extend(stage_pack.TESTING_SUPPORT)
    self.write_inputs()
    result = self.run_stage()
    self.assertFalse(result['supplemental_inputs'])
    self.assertEqual(len(result['components']['testing']['files']), len(stage_pack.TESTING_SUPPORT))

  def test_unsafe_paths_rejected(self):
    for name in ('../outside', '/outside', 'C:/outside', 'a\\b', './LICENSE'):
      with self.subTest(name=name), self.assertRaises(stage_pack.BuildError):
        stage_pack.safe_file(self.root, name)

  def test_runtime_revision_mismatch_rejected(self):
    self.write_inputs()
    with patch.object(stage_pack, 'git_state', return_value={'commit': 'b' * 40}):
      with self.assertRaisesRegex(stage_pack.BuildError, 'Runtime commit differs'):
        self.run_stage()

  def test_source_change_during_copy_leaves_no_partial_output(self):
    self.write_inputs()
    def changed_copy(source, target):
      Path(target).write_bytes(b'changed during copy')
    with patch.object(stage_pack.shutil, 'copyfile', side_effect=changed_copy):
      with self.assertRaisesRegex(stage_pack.BuildError, 'Source changed during staging'):
        self.run_stage()
    self.assertFalse(self.output.exists())

  def test_symlink_cannot_copy_external_file(self):
    outside = self.base / 'outside'
    outside.write_text('outside')
    try:
      (self.root / 'link').symlink_to(outside)
    except OSError:
      self.skipTest('Symlinks unavailable on this platform')
    with self.assertRaisesRegex(stage_pack.BuildError, 'within its root'):
      stage_pack.safe_file(self.root, 'link')


if __name__ == '__main__':
  unittest.main()
