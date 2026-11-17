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
"""Classification checks: external libraries must not become TFLM payload."""

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/classify_sources.py'
SPEC = importlib.util.spec_from_file_location('classify_sources', SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ClassificationTests(unittest.TestCase):
  def test_legacy_runtime_paths_are_explicitly_recognized(self):
    for name in module.LEGACY_RUNTIME:
      self.assertEqual(module.category(name), 'tflm/runtime_and_support')
    self.assertEqual(module.category('tensorflow/compiler/mlir/lite/unknown.cc'), 'unclassified')

  def test_external_boundaries_and_integration_sources(self):
    prefix = module.DOWNLOADS
    for path, expected in {
      prefix + 'cmsis/CMSIS/Core/Include/core_cm55.h': 'external/cmsis',
      prefix + 'cmsis/Cortex_DFP/Device/ARMCM55/Include/ARMCM55.h': 'external/cortex_dfp',
      prefix + 'cmsis_nn/Source/conv.c': 'external/cmsis_nn',
      prefix + 'ethos_u_core_driver/src/ethosu_driver.c': 'external/ethos_u_core_driver',
      'tensorflow/lite/micro/kernels/cmsis_nn/conv.cc': 'tflm/cmsis_nn_integration',
      'tensorflow/lite/micro/kernels/ethos_u/ethosu.cc': 'tflm/ethos_u_integration',
      prefix + 'cmsis_nn_extra/foo.c': 'unclassified',
      prefix + 'unknown/foo.h': 'unclassified',
      'tensorflow/lite/micro/tools/tool.cc': 'unclassified',
    }.items():
      with self.subTest(path=path):
        self.assertEqual(module.category(path), expected)

  def test_bundled_libraries_and_licenses(self):
    for name in module.BUNDLED:
      for prefix in (module.DOWNLOADS, 'third_party/'):
        self.assertEqual(module.category(prefix + name + '/LICENSE'), 'bundled/' + name)
    self.assertEqual(module.category('LICENSE'), 'tflm/license')

  def test_partition_preserves_each_path_once(self):
    paths = ['LICENSE', 'LICENSE', 'signal/src/fft.cc', 'unknown.cc']
    result = module.classify(paths)
    flattened = [path for values in result['groups'].values() for path in values]
    self.assertEqual(sorted(flattened), sorted(set(paths)))
    self.assertFalse(result['all_paths_classified'])
    self.assertFalse(result['payload_approved'])

  def test_unsafe_relative_paths_rejected(self):
    for path in ('../outside.cc', '/outside.cc', 'tensorflow/../outside.cc', 'foo\\bar.cc'):
      with self.subTest(path=path), self.assertRaises(ValueError):
        module.category(path)


if __name__ == '__main__':
  unittest.main()
