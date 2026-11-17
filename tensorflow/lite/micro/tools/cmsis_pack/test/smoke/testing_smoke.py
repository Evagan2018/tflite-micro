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
"""Compile Testing consumers for Cortex-M and execute them on the Linux host.

Uses only the generated pack payload, not headers from a runtime checkout.
Host runs check framework success/failure reporting, helper functions and fixtures.
Arm ELF files are link checks only; no board or FVP execution is attempted.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from compile_smoke import sha, run

MICRO = 'tensorflow/lite/micro/'


def consumer(version):
  header = 'micro_test.h' if version == 1 else 'micro_test_v2.h'
  extra = '' if version == 1 else '''
#include "tensorflow/lite/micro/kernels/conv_test.h"
#include "tensorflow/lite/micro/kernels/lstm_eval_test.h"
#include "tensorflow/lite/micro/kernels/decode_test_helpers.h"
'''
  begin = 'TF_LITE_MICRO_TESTS_BEGIN' if version == 1 else ''
  test = 'TF_LITE_MICRO_TEST(TestingSupport)' if version == 1 else 'TEST(Pack, TestingSupport)'
  expect = 'TF_LITE_MICRO_EXPECT_EQ' if version == 1 else 'EXPECT_EQ'
  end = 'TF_LITE_MICRO_TESTS_END' if version == 1 else 'TF_LITE_MICRO_TESTS_MAIN'
  return f'''#include "{MICRO}testing/{header}"
#include "{MICRO}test_helpers.h"
#include "{MICRO}micro_interpreter.h"
#include "{MICRO}testing/test_conv_model.h"
#include "{MICRO}kernels/testdata/conv_test_data.h"
#include "{MICRO}cortex_m_generic/debug_log_callback.h"
#include <cstdio>
{extra}
namespace {{
void print_log(const char* text) {{ std::fputs(text, stdout); }}
struct Logging {{ Logging() {{ RegisterDebugLogCallback(print_log); }} }} logging;
}}
{begin}
{test} {{
  int dimensions[] = {{2, 2, 3}};
  auto* dims = tflite::testing::IntArrayFromInts(dimensions);
  {expect}(dims->size, 2);
  {expect}(dims->data[1], 3);
  float values[] = {{1, 2, 3, 4, 5, 6}};
  auto tensor = tflite::testing::CreateTensor(values, dims);
  {expect}(tensor.data.f[5], 6.0f);
  {expect}(kTestConvModelDataSize > 0, true);
  {expect}(tflite::GetModel(kTestConvModelData)->version(), TFLITE_SCHEMA_VERSION);
#ifdef INJECT_FAILURE
  {expect}(1, 2);
#endif
}}
{end}
'''


def build(args):
  pack = args.pack_dir.resolve(strict=True)
  if args.out.exists():
    raise ValueError('Choose a new output directory')
  report = json.loads((pack / 'pdsc-report.json').read_text())
  pdsc = pack / 'TensorFlow.LiteRT.pdsc'
  if sha(pdsc) != report['pdsc_sha256']:
    raise ValueError('PDSC hash mismatch')
  for name, digest in report['output_sha256'].items():
    path = (pack / name).resolve(strict=True)
    if not path.is_relative_to(pack) or sha(path) != digest:
      raise ValueError('Payload hash mismatch or unsafe path: ' + name)
  components = ET.parse(pdsc).getroot().findall('components/component')
  testing = [c for c in components if c.get('Csub') == 'Testing']
  if len(testing) != 1:
    raise ValueError('Expected exactly one Testing component')
  sources = [f.get('name') for f in testing[0].findall('files/file') if f.get('category') in ('sourceC', 'sourceCpp')]
  sources += ['config/debug_log.cpp', 'config/system_setup.cpp', MICRO + 'micro_log.cc', MICRO + 'micro_utils.cc']
  sources += ['tensorflow/lite/kernels/kernel_util.cc', 'tensorflow/lite/core/c/common.cc']
  includes = sorted({f.get('name') for c in components for f in c.findall('files/file') if f.get('category') == 'include'})
  for name in sources:
    if name not in report['output_sha256']:
      raise ValueError('Unverified Testing source: ' + str(name))
  args.out.mkdir(parents=True)
  receipt = {'pdsc_sha256': sha(pdsc), 'hardware_executed': False,
             'release_qualified': False, 'source_sha256': report['output_sha256'], 'targets': {}}
  targets = [('host', Path(shutil.which('g++') or 'g++'), []),
             ('cortex-m7', args.toolchain_bin / 'arm-none-eabi-g++', ['-mcpu=cortex-m7', '-mthumb', '-mfloat-abi=soft']),
             ('cortex-m55', args.toolchain_bin / 'arm-none-eabi-g++', ['-mcpu=cortex-m55', '-mthumb', '-mfloat-abi=hard'])]
  for target, compiler, arch in targets:
    work = args.out / target
    work.mkdir()
    flags = [*arch, '-std=c++17', '-O0', '-DTF_LITE_STATIC_MEMORY', '-ffunction-sections', '-fdata-sections',
             '-fno-exceptions', '-fno-rtti', *['-I' + str(pack / n) for n in includes]]
    objects = []
    result = {'compiler': subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0],
              'compiler_sha256': sha(compiler.resolve()), 'tests': {}}
    receipt['targets'][target] = result
    for i, name in enumerate(sources):
      obj = work / f'{i}.o'
      if run([compiler, *flags, '-c', pack / name, '-o', obj], work / f'{i}.log'):
        raise ValueError(f'Compile failed: {target}/{i}.log ({name})')
      objects.append(obj)
    for version in (1, 2):
      for fail in ((False, True) if target == 'host' else (False,)):
        name = f'v{version}' + ('-expected-failure' if fail else '')
        source = work / (name + '.cc')
        source.write_text(consumer(version))
        binary = work / (name + '.elf')
        options = ['-DINJECT_FAILURE'] if fail else []
        if target != 'host':
          options += ['--specs=nosys.specs']
        if run([compiler, *flags, *options, source, *objects, '-Wl,--gc-sections', '-o', binary], work / (name + '-link.log')):
          raise ValueError(f'Link failed: {target}/{name}-link.log')
        check = {'linked': True, 'elf_sha256': sha(binary), 'executed': target == 'host'}
        if target == 'host':
          code = run([binary], work / (name + '-run.log'))
          check.update(exit_code=code, expected_failure=fail)
          if code != (1 if fail else 0):
            raise ValueError(f'Unexpected test result: {name}: {code}')
        result['tests'][name] = check
    (args.out / 'testing-report.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(target + ': Testing consumers passed', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('pack-dir', 'toolchain-bin', 'out'):
    parser.add_argument('--' + name, type=Path, required=True)
  args = parser.parse_args()
  try:
    build(args)
    return 0
  except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
    print(str(error), file=sys.stderr)
    return 1


if __name__ == '__main__':
  sys.exit(main())
