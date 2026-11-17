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
"""Offline builder integration tests using real Git and a fixture generator."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


SCAFFOLD = Path(__file__).resolve().parents[1]
GENERATOR = Path('tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py')


class BuildPackTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.base = Path(self.temp.name)
    self.runtime = self.base / 'runtime'
    self.runtime.mkdir()
    self.builder = self.base / 'builder'
    shutil.copytree(SCAFFOLD, self.builder, ignore=shutil.ignore_patterns('__pycache__'))
    self.config = self.builder / 'pack.yml'
    self.generator = self.runtime / GENERATOR
    self.generator.parent.mkdir(parents=True)
    self.generator.write_text(
      'import sys\n'
      'assert "--no_copy" in sys.argv\n'
      'assert "--print_src_files" in sys.argv\n'
      'assert "--makefile_options=TARGET=cortex_m_generic TARGET_ARCH=cortex-m7" in sys.argv\n'
      'print("b.h a.cc a.cc")\n', encoding='utf-8',
    )
    (self.runtime / 'a.cc').write_text('// source\n', encoding='utf-8')
    (self.runtime / 'b.h').write_text('// header\n', encoding='utf-8')
    for repo in (self.runtime, self.builder):
      self.git(repo, 'init', '-q')
      self.git(repo, 'add', '.')
      self.git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
               '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
    self.output = self.base / 'new output'

  def git(self, repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

  def cli(self, *extra):
    return subprocess.run(
      [sys.executable, '-B', str(self.builder / 'scripts/build_pack.py'),
       '--tflm-root', str(self.runtime), '--out', str(self.output), *extra],
      cwd=self.base, text=True, capture_output=True, check=False,
    )

  def test_records_separate_revisions_and_actual_file_hashes(self):
    (self.runtime / 'a.cc').write_text('// modified\n', encoding='utf-8')
    result = self.cli()
    self.assertEqual(result.returncode, 0, result.stderr)
    inventory = (self.output / 'reference-sources.txt').read_bytes()
    self.assertEqual(inventory, b'a.cc\nb.h\n')
    data = json.loads((self.output / 'provenance.json').read_text())
    self.assertEqual(data['runtime']['commit'], self.git(self.runtime, 'rev-parse', 'HEAD'))
    self.assertEqual(data['builder']['commit'], self.git(self.builder, 'rev-parse', 'HEAD'))
    self.assertIn('a.cc', data['runtime']['status_porcelain'])
    self.assertEqual(data['inventory']['sha256'], hashlib.sha256(inventory).hexdigest())
    self.assertEqual(data['inventory']['source_files_sha256']['a.cc'],
                     hashlib.sha256((self.runtime / 'a.cc').read_bytes()).hexdigest())
    self.assertFalse(data['release_qualified'])
    self.assertIsNone(data['config']['contents']['release_baseline']['tflite_micro']['commit'])
    self.assertEqual(sorted(p.name for p in self.output.iterdir()),
                     ['classification.json', 'provenance.json', 'reference-sources.txt'])

  def backend_generator(self, fail_ethos=False, mutate=False):
    self.generator.write_text(
      'from pathlib import Path\nimport sys\n'
      'options = next(x for x in sys.argv if x.startswith("--makefile_options="))\n'
      'paths = ["a.cc", "tensorflow/lite/micro/tools/make/downloads/flatbuffers/LICENSE"]\n'
      'if "OPTIMIZED_KERNEL_DIR=cmsis_nn" in options:\n'
      '  assert "TARGET_ARCH=cortex-m55" in options\n'
      '  paths += ["tensorflow/lite/micro/kernels/cmsis_nn/conv.cc",\n'
      '            "tensorflow/lite/micro/tools/make/downloads/cmsis_nn/Source/conv.c"]\n'
      'if "CO_PROCESSOR=ethos_u" in options:\n'
      '  assert "ETHOSU_ARCH=u55" in options\n'
      '  assert "OPTIMIZED_KERNEL_DIR=cmsis_nn" in options\n'
      f'  if {fail_ethos!r}: raise SystemExit("Ethos fixture failure")\n'
      '  paths += ["tensorflow/lite/micro/kernels/ethos_u/ethosu.cc",\n'
      '            "tensorflow/lite/micro/tools/make/downloads/ethos_u_core_driver/src/ethosu_driver.c"]\n'
      f'  if {mutate!r}: Path("a.cc").write_text("changed")\n'
      'for name in paths:\n'
      '  p = Path(name)\n'
      '  p.parent.mkdir(parents=True, exist_ok=True)\n'
      '  if not p.exists(): p.write_text("fixture")\n'
      'print(" ".join(paths))\n', encoding='utf-8',
    )

  def test_all_variants_and_external_classification(self):
    self.backend_generator()
    result = self.cli('--variant', 'all')
    self.assertEqual(result.returncode, 0, result.stderr)
    data = json.loads((self.output / 'provenance.json').read_text())
    report = json.loads((self.output / 'classification.json').read_text())
    self.assertEqual(set(data['inventories']), {'reference', 'cmsis_nn', 'ethos_u'})
    self.assertEqual(data['schema_version'], 2)
    self.assertNotIn('inventory', data)
    groups = report['variants']['ethos_u']['groups']
    self.assertEqual(len(groups['external/ethos_u_core_driver']), 1)
    self.assertEqual(len(groups['external/cmsis_nn']), 1)
    self.assertEqual(len(groups['tflm/cmsis_nn_integration']), 1)
    self.assertEqual(len(groups['tflm/ethos_u_integration']), 1)
    self.assertEqual(groups['unclassified'], ['a.cc'])
    self.assertIn('unclassified paths', result.stderr)
    for name, entry in data['inventories'].items():
      payload = (self.output / entry['file']).read_bytes()
      self.assertEqual(entry['sha256'], hashlib.sha256(payload).hexdigest())
    self.assertEqual(data['classification']['sha256'],
                     hashlib.sha256((self.output / 'classification.json').read_bytes()).hexdigest())

  def test_single_optimized_variants(self):
    self.backend_generator()
    for name in ('cmsis_nn', 'ethos_u'):
      with self.subTest(name=name):
        self.output = self.base / name
        result = self.cli('--variant', name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / f'{name}-sources.txt').is_file())
        self.assertFalse((self.output / 'reference-sources.txt').exists())

  def test_later_variant_failure_does_not_publish_partial_results(self):
    self.backend_generator(fail_ethos=True)
    result = self.cli('--variant', 'all')
    self.assertEqual(result.returncode, 1)
    self.assertIn('Ethos fixture failure', result.stderr)
    self.assertFalse(self.output.exists())

  def test_shared_input_mutation_between_variants_is_rejected(self):
    self.backend_generator(mutate=True)
    result = self.cli('--variant', 'all')
    self.assertEqual(result.returncode, 1)
    self.assertIn('Shared input changed', result.stderr)
    self.assertFalse(self.output.exists())

  def test_ethos_requires_cmsis_nn_and_unsupported_policy_rejected(self):
    config = yaml.safe_load(self.config.read_text())
    config['variants']['ethos_u']['requires'] = ['cmsis', 'ethos_u_driver']
    self.config.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = self.cli('--variant', 'ethos_u')
    self.assertEqual(result.returncode, 1)
    self.assertIn('requires must contain exactly', result.stderr)
    config['third_party']['bundled'].append('cmsis_nn')
    self.config.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = self.cli()
    self.assertEqual(result.returncode, 1)
    self.assertIn('Unsupported third_party.bundled', result.stderr)

  def test_existing_output_is_preserved(self):
    self.output.mkdir()
    (self.output / 'sentinel').write_text('keep')
    result = self.cli()
    self.assertEqual(result.returncode, 1)
    self.assertIn('Output already exists', result.stderr)
    self.assertEqual((self.output / 'sentinel').read_text(), 'keep')

  def test_generator_failure_leaves_no_partial_output(self):
    self.generator.write_text('raise SystemExit("fixture failure")\n')
    result = self.cli()
    self.assertEqual(result.returncode, 1)
    self.assertIn('fixture failure', result.stderr)
    self.assertFalse(self.output.exists())

  def test_duplicate_and_unsafe_yaml_rejected(self):
    for contents in ('status: draft\nstatus: release\n', '!!python/object:os.system {}'):
      with self.subTest(contents=contents):
        self.config.write_text(contents, encoding='utf-8')
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn('Traceback', result.stderr)

  def test_invalid_config_rejected_before_generation(self):
    original = yaml.safe_load(self.config.read_text())
    for field, value in (('schema_version', True), ('status', 'release'),
                         ('pack', None), ('variants', []), ('source_lists', {})):
      with self.subTest(field=field):
        config = dict(original)
        config[field] = value
        self.config.write_text(yaml.safe_dump(config), encoding='utf-8')
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn('Traceback', result.stderr)

  def test_custom_config_is_used_and_unsupported_backend_rejected(self):
    config = yaml.safe_load(self.config.read_text())
    config['variants']['reference']['make_options']['OPTIMIZED_KERNEL_DIR'] = 'cmsis_nn'
    custom = self.base / 'custom.yml'
    custom.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = self.cli('--config', str(custom))
    self.assertEqual(result.returncode, 1)
    self.assertIn('supports only Reference', result.stderr)
    self.assertFalse(self.output.exists())

  def test_parent_checkout_not_accepted_as_runtime_root(self):
    result = self.cli('--tflm-root', str(self.runtime / 'tensorflow'))
    self.assertEqual(result.returncode, 1)
    self.assertIn('must be the Git checkout root', result.stderr)

  def test_missing_source_leaves_no_output(self):
    self.generator.write_text('print("missing.cc")\n')
    result = self.cli()
    self.assertEqual(result.returncode, 1)
    self.assertIn('cannot be resolved', result.stderr)
    self.assertFalse(self.output.exists())


if __name__ == '__main__':
  unittest.main()
