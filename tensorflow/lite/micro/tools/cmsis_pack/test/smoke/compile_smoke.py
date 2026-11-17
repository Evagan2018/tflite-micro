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
"""Offline GCC compile/link checks of a generated development PDSC payload.

This is a direct compiler harness, not CMSIS dependency resolution or a board run.
It intentionally preserves failed build logs and records the external inputs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


PROFILES = {'reference': ('Reference', 'cortex-m7', 'ARMCM7'),
            'cmsis_nn': ('CMSIS-NN', 'cortex-m55', 'ARMCM55'),
            'ethos_u': ('Ethos-U', 'cortex-m55', 'ARMCM55')}


def sha(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, log):
  result = subprocess.run([str(x) for x in command], capture_output=True, text=True, check=False)
  log.write_text('COMMAND: ' + json.dumps([str(x) for x in command]) + '\n' + result.stdout + result.stderr, encoding='utf-8')
  return result.returncode


def selected_files(pdsc, variant):
  sources, includes = [], set()
  found = False
  for component in ET.parse(pdsc).getroot().findall('components/component'):
    if component.get('Csub') == 'Kernel' and component.get('Cvariant') != variant:
      continue
    if component.get('Csub') == 'Kernel':
      found = True
    for item in component.findall('files/file'):
      kind, name = item.get('category'), item.get('name')
      if not name or not (pdsc.parent / name).resolve().is_relative_to(pdsc.parent.resolve()):
        raise ValueError('Unsafe PDSC file path')
      if kind in ('sourceC', 'sourceCpp'):
        sources.append(name)
      elif kind == 'include':
        includes.add(name)
  if not found:
    raise ValueError('Requested Kernel variant is absent')
  if len(sources) != len(set(sources)):
    raise ValueError('Duplicate translation units in selected PDSC components')
  return sorted(sources), sorted(includes)


def build(args):
  if args.out.exists():
    raise ValueError('Output already exists; choose a new directory')
  report = json.loads((args.pack_dir / 'pdsc-report.json').read_text())
  for name, expected in report['output_sha256'].items():
    path = (args.pack_dir / name).resolve()
    if not path.is_relative_to(args.pack_dir.resolve()) or sha(path) != expected:
      raise ValueError(f'Payload hash mismatch or unsafe path: {name}')
  pdsc = args.pack_dir / 'TensorFlow.LiteRT.pdsc'
  if sha(pdsc) != report['pdsc_sha256']:
    raise ValueError('PDSC hash mismatch')
  compiler = args.toolchain_bin / 'arm-none-eabi-g++'
  version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
  args.out.mkdir(parents=True)
  payload = args.out / 'payload'
  payload.mkdir()
  for name in report['output_sha256']:
    target = payload / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.pack_dir / name, target)
  shutil.copyfile(pdsc, payload / pdsc.name)
  external = args.out / 'external'
  for label, source in [('cmsis', args.cmsis), ('cortex_dfp', args.cortex_dfp),
                        ('cmsis_nn', args.cmsis_nn), ('driver', args.driver)]:
    shutil.copytree(source, external / label)
  receipt = {'compiler': version, 'compiler_sha256': sha(compiler),
             'pdsc_sha256': sha(pdsc), 'model_sha256': sha(args.model),
             'release_qualified': False, 'hardware_executed': False,
             'dependency_ranges_validated': False,
             'link_coverage': 'FullyConnected, Reshape; additionally Ethos-U registration in ethos_u profile',
             'external_paths': {key: str(getattr(args, key)) for key in ('cmsis', 'cortex_dfp', 'cmsis_nn', 'driver')},
             'external_input_hashes': {p.relative_to(external).as_posix(): sha(p) for p in external.rglob('*') if p.is_file()},
             'profiles': {}}
  model = args.model.read_bytes()
  for key, (variant, cpu, device) in PROFILES.items():
    work = args.out / key
    work.mkdir()
    (work / 'objects').mkdir()
    (work / 'logs').mkdir()
    print(f'Building {key} for {cpu}', flush=True)
    sources, include_names = selected_files(payload / pdsc.name, variant)
    paths = [payload / name for name in sources]
    includes = [payload / name for name in include_names]
    includes += [work, external / 'cmsis/CMSIS/Core/Include',
                 external / f'cortex_dfp/Device/{device}/Include']
    flags = ['-mcpu=' + cpu, '-mthumb', '-mfloat-abi=soft' if key == 'reference' else '-mfloat-abi=hard',
             '-Os', '-ffunction-sections', '-fdata-sections', '-DTF_LITE_STATIC_MEMORY=1',
             '-D' + device, '-DCPU_' + ('M7' if key == 'reference' else 'M55') + '=1']
    if key == 'reference':
      flags += ['-mfpu=fpv5-d16']
    else:
      flags += ['-DCMSIS_NN=1', '-DARM_MODEL_USE_PMU_COUNTERS', '-DARM_NN_ENABLE_F16=0', '-DARM_NN_ENABLE_F32=0']
      includes += [external / 'cmsis_nn/Include']
      paths += sorted(p for p in (external / 'cmsis_nn/Source').rglob('*.c')
                      if not p.name.endswith(('_f16.c', '_f32.c')) and p.name != 'arm_nntables_flt.c')
    if key == 'ethos_u':
      includes += [external / 'driver/include', external / 'driver/src']
      flags += ['-DETHOSU_ARCH=u55', '-DETHOSU55', '-DETHOSU_LOG_SEVERITY=ETHOSU_LOG_WARN', '-DSMOKE_ETHOS_U=1']
      paths += [external / ('driver/src/' + name) for name in ('ethosu_driver.c', 'ethosu_pmu.c', 'ethosu_device_u55_u65.c')]
    (work / 'RTE_Components.h').write_text(f'#define CMSIS_device_header "{device}.h"\n#define RTE_ML_LITERT\n#define RTE_ML_TF_LITE\n')
    (work / 'model.cc').write_text('alignas(16) extern const unsigned char smoke_model[] = {' + ','.join(str(v) for v in model) + '};\n')
    (work / 'main.cc').write_text('''#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern const unsigned char smoke_model[];
alignas(16) static unsigned char arena[16384];
int main() {
  tflite::MicroMutableOpResolver<3> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) return 1;
  if (resolver.AddReshape() != kTfLiteOk) return 2;
#ifdef SMOKE_ETHOS_U
  if (resolver.AddEthosU() != kTfLiteOk) return 3;
#endif
  tflite::MicroInterpreter interpreter(tflite::GetModel(smoke_model), resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) return 4;
  interpreter.input(0)->data.int8[0] = 0;
  return interpreter.Invoke() == kTfLiteOk ? 0 : 5;
}
''')
    paths += [work / 'model.cc', work / 'main.cc']
    objects = [work / 'objects' / (str(index) + '.o') for index in range(len(paths))]
    def compile_one(index):
      source = paths[index]
      cxx = source.suffix != '.c'
      tool = compiler if cxx else args.toolchain_bin / 'arm-none-eabi-gcc'
      extra = ['-std=c++17', '-fno-exceptions', '-fno-rtti', '-fno-threadsafe-statics'] if cxx else ['-std=c11']
      return run([tool, *flags, *extra, *['-I' + str(p) for p in includes], '-c', source, '-o', objects[index]], work / 'logs' / f'{index}.log')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
      codes = list(pool.map(compile_one, range(len(paths))))
    failures = [{'source': str(paths[i]), 'log': str(work / 'logs' / f'{i}.log')} for i, code in enumerate(codes) if code]
    entry = {'translation_units': len(paths), 'compile_failures': failures, 'linked': False}
    if not failures:
      # Ordinary bare-metal default linker script and nosys stubs: not a board image.
      entry['link_exit_code'] = run([compiler, *flags, *objects, '--specs=nosys.specs',
                                    '-Wl,--gc-sections', '-o', work / 'smoke.elf'], work / 'link.log')
      entry['linked'] = entry['link_exit_code'] == 0
      if entry['linked']:
        entry['elf_sha256'] = sha(work / 'smoke.elf')
    receipt['profiles'][key] = entry
    (args.out / 'smoke-report.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"{key}: {len(failures)} compile failures; linked={entry['linked']}", flush=True)
  return all(value['linked'] for value in receipt['profiles'].values())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('pack-dir', 'toolchain-bin', 'cmsis', 'cortex-dfp', 'cmsis-nn', 'driver', 'model', 'out'):
    parser.add_argument('--' + name, required=True, type=Path)
  parser.add_argument('--jobs', type=int, default=4)
  args = parser.parse_args()
  if args.jobs < 1:
    parser.error('--jobs must be positive')
  try:
    return 0 if build(args) else 1
  except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  sys.exit(main())
