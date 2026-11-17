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
"""Check the smoke harness selects a single kernel variant and shared components."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'smoke'))
import compile_smoke
sys.path.pop(0)


class SelectionTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.pdsc = Path(self.temp.name) / 'test.pdsc'

  def write(self, extra=''):
    self.pdsc.write_text('''<package><components>
      <component Csub="Kernel" Cvariant="Reference"><files>
        <file category="sourceCpp" name="reference.cc"/>
      </files></component>
      <component Csub="Kernel" Cvariant="CMSIS-NN"><files>
        <file category="sourceCpp" name="optimized.cc"/>
      </files></component>
      <component Csub="Kernel Utils"><files>
        <file category="sourceC" name="shared.c"/>
        <file category="include" name="./"/>
      </files></component>''' + extra + '</components></package>')

  def test_selects_one_variant_and_shared_sources(self):
    self.write()
    self.assertEqual(compile_smoke.selected_files(self.pdsc, 'Reference'),
                     (['reference.cc', 'shared.c'], ['./']))
    self.assertEqual(compile_smoke.selected_files(self.pdsc, 'CMSIS-NN')[0],
                     ['optimized.cc', 'shared.c'])

  def test_missing_variant_rejected(self):
    self.write()
    with self.assertRaisesRegex(ValueError, 'absent'):
      compile_smoke.selected_files(self.pdsc, 'Ethos-U')

  def test_duplicate_compile_unit_rejected(self):
    self.write('<component><files><file category="sourceC" name="shared.c"/></files></component>')
    with self.assertRaisesRegex(ValueError, 'Duplicate'):
      compile_smoke.selected_files(self.pdsc, 'Reference')

  def test_escaping_source_rejected(self):
    self.write('<component><files><file category="sourceC" name="../outside.c"/></files></component>')
    with self.assertRaisesRegex(ValueError, 'Unsafe'):
      compile_smoke.selected_files(self.pdsc, 'Reference')


if __name__ == '__main__':
  unittest.main()
