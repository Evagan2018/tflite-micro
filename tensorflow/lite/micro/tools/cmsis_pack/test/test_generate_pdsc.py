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
"""Development metadata, component dependencies and copied-source adaptations."""

import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import test_stage_pack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import generate_pdsc
sys.path.pop(0)


class PdscTests(unittest.TestCase):
  def setUp(self):
    self.fixture = test_stage_pack.StageTests()
    self.fixture.setUp()
    self.addCleanup(self.fixture.doCleanups)
    self.fixture.config.update(
      pack={'vendor': 'TensorFlow', 'name': 'LiteRT', 'version': '1.0.0', 'description': 'A & B <preview>'},
      requirements={
        'cmsis': {'vendor': 'ARM', 'name': 'CMSIS', 'version_range': None,
                  'component': {'Cclass': 'CMSIS', 'Cgroup': 'CORE'}, 'status': 'unresolved'},
        'cmsis_nn': {'vendor': 'ARM', 'name': 'CMSIS-NN', 'version_range': '8.0.0:8.99.99',
                     'component': {'Cclass': 'CMSIS', 'Cgroup': 'NN Lib'}, 'status': 'proposed_unverified'},
        'ethos_u_driver': {'vendor': 'ARM', 'name': 'ethos-u-core-driver', 'version_range': '2.0.0:2.99.99',
                           'component': None, 'status': 'proposed_unverified'},
      },
    )
    self.fixture.config['compatibility']['defines'] = ['RTE_ML_TF_LITE', 'RTE_ML_LITERT']
    self.fixture.lists['ethos_u'] = list(self.fixture.lists['cmsis_nn']) + [self.fixture.micro + 'kernels/ethos_u/ethosu.cc']
    self.fixture.write_inputs()
    self.fixture.run_stage()
    self.output = self.fixture.base / 'pdsc'

  def generate(self):
    return generate_pdsc.generate(self.fixture.output, self.output, '2026-11-17')

  def test_development_version_dependencies_and_config_versions(self):
    report = self.generate()
    root = ET.parse(self.output / 'TensorFlow.LiteRT.pdsc').getroot()
    self.assertEqual(root.find('description').text, 'A & B <preview>')
    self.assertEqual(root.find('releases/release').get('version'), '1.0.0-dev.0')
    condition = root.find("conditions/condition[@id='LiteRT kernel ethos_u']")
    self.assertIsNotNone(condition.find("require[@Cgroup='NN Lib']"))
    self.assertIsNotNone(condition.find("require[@Csub='Ethos-U Driver']"))
    reference = root.find("conditions/condition[@id='LiteRT kernel reference']")
    self.assertIsNone(reference.find("require[@Cgroup='NN Lib']"))
    configs = root.findall(".//file[@attr='config']")
    self.assertTrue(configs)
    self.assertTrue(all(item.get('version') == '1.0.0' for item in configs))
    self.assertTrue(all(item.get('name').startswith('config/') for item in configs))
    self.assertTrue((self.output / 'config/system_setup.cpp').exists())
    self.assertFalse((self.output / 'tensorflow/lite/micro/system_setup.cc').exists())
    self.assertFalse(report['release_qualified'])
    self.assertTrue(any('selector' in item for item in report['blockers']))
    for item in root.findall('.//files/file'):
      self.assertTrue((self.output / item.get('name')).exists())

  def test_hash_mismatch_rejected_without_output(self):
    (self.fixture.output / 'payload/LICENSE').write_text('changed')
    with self.assertRaisesRegex(generate_pdsc.BuildError, 'Payload hash mismatch'):
      self.generate()
    self.assertFalse(self.output.exists())

  def test_unknown_component_reference_rejected(self):
    path = self.fixture.output / 'component-map.json'
    data = json.loads(path.read_text())
    data['components']['kernel/reference']['files'].append('../outside.cc')
    path.write_text(json.dumps(data))
    with self.assertRaisesRegex(generate_pdsc.BuildError, 'Invalid component ownership'):
      self.generate()

  def test_existing_destination_preserved(self):
    self.output.mkdir()
    (self.output / 'keep').write_text('keep')
    with self.assertRaisesRegex(generate_pdsc.BuildError, 'Output already exists'):
      self.generate()
    self.assertEqual((self.output / 'keep').read_text(), 'keep')

  def test_old_incomplete_testing_map_requires_restaging(self):
    path = self.fixture.output / 'component-map.json'
    mapping = json.loads(path.read_text())
    mapping['components']['testing']['files'].remove(generate_pdsc.TESTING_SUPPORT[0])
    with self.assertRaisesRegex(generate_pdsc.BuildError, 'Incomplete Testing'):
      generate_pdsc.render(mapping, self.fixture.config, '2026-11-17')

  def test_cmsis_nn_rewrite_preserves_header_identity(self):
    source = b'#include "Include/arm_nn_types.h"\n#include "Include/arm_nnfunctions.h"\n'
    result, changes = generate_pdsc.adapt_source('tensorflow/lite/micro/kernels/cmsis_nn/pad.cc', source)
    self.assertEqual(result, b'#include "arm_nn_types.h"\n#include "arm_nnfunctions.h"\n')
    self.assertEqual(len(changes), 1)

  def test_timing_adaptation_requires_known_block(self):
    name = 'tensorflow/lite/micro/cortex_m_generic/micro_time.cc'
    result, changes = generate_pdsc.adapt_source(name, generate_pdsc.TIME_INCLUDE.encode())
    self.assertIn(b'#include CMSIS_device_header', result)
    self.assertTrue(changes)
    with self.assertRaisesRegex(generate_pdsc.BuildError, 'Unrecognized'):
      generate_pdsc.adapt_source(name, b'unknown upstream implementation')


if __name__ == '__main__':
  unittest.main()
