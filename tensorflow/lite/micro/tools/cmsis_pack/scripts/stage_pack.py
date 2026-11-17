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
"""Stage a draft component payload from hash-verified backend inventories.

No downloads, source patches, PDSC generation or release qualification occur here.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

from build_pack import BuildError, VARIANTS, digest, git_state
from classify_sources import BUNDLED, DOWNLOADS, category


MICRO = 'tensorflow/lite/micro/'
CONFIG_FILES = {
  MICRO + 'system_setup.cc': 'system_setup.cpp',
  MICRO + 'cortex_m_generic/micro_time.cc': 'micro_time.cpp',
  MICRO + 'cortex_m_generic/debug_log.cc': 'debug_log.cpp',
}
UTIL_FILES = {*CONFIG_FILES, 'tensorflow/lite/kernels/kernel_util.cc'}
SOURCE_SUFFIXES = {'.c', '.cc', '.cpp', '.s', '.S'}
TESTING_SUPPORT = tuple(MICRO + name for name in (
    'testing/micro_test.h', 'testing/micro_test_v2.h',
    'testing/test_conv_model.h', 'testing/test_conv_model.cc',
    'kernels/testdata/conv_test_data.h', 'kernels/testdata/conv_test_data.cc',
    'kernels/testdata/lstm_test_data.h', 'kernels/testdata/lstm_test_data.cc',
))


def add_testing_support(root, lists, hashes):
  """Supplement the inference inventory with reviewed reusable Testing inputs.

  Deliberately do not glob test programs, generators or platform scripts.
  Missing files in a different upstream revision require explicit review.
  """
  supplements = []
  for name in TESTING_SUPPORT:
    sha = digest(safe_file(root, name))
    if name in hashes and hashes[name] != sha:
      raise BuildError(f'Testing source changed after inventory validation: {name}')
    added = []
    for variant, sources in lists.items():
      if name not in sources:
        sources.append(name)
        sources.sort()
        added.append(variant)
    hashes[name] = sha
    if added:
      supplements.append({'source': name, 'sha256': sha,
                          'reason': 'Reviewed reusable Testing support omitted by inference inventory',
                          'variants': sorted(added)})
  return supplements


def safe_file(root, name):
  """Reject traversal, alternate separators and symlinks escaping the root."""
  if not isinstance(name, str) or not name or ':' in name or '\\' in name:
    raise BuildError(f'Unsafe relative path: {name!r}')
  path = PurePosixPath(name)
  if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
    raise BuildError(f'Unsafe relative path: {name!r}')
  resolved = (root / name).resolve(strict=True)
  if not resolved.is_relative_to(root) or not resolved.is_file():
    raise BuildError(f'Input is not a file within its root: {name}')
  return resolved


def load_inputs(root, inventory_dir):
  """Do not trust classification or source paths without checking provenance."""
  provenance_path = safe_file(inventory_dir, 'provenance.json')
  provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
  if provenance.get('schema_version') != 2:
    raise BuildError('Staging requires backend inventory provenance schema 2')
  runtime = git_state(root, exact_root=True)
  if runtime['commit'] != provenance['runtime']['commit']:
    raise BuildError('Runtime commit differs from the inventory provenance')
  records = provenance['inventories']
  if not records or set(records) - set(VARIANTS):
    raise BuildError('Missing or unsupported inventory variants')
  lists, hashes = {}, {}
  for variant, record in records.items():
    path = safe_file(inventory_dir, record['file'])
    if digest(path) != record['sha256']:
      raise BuildError(f'Inventory hash mismatch: {variant}')
    sources = path.read_text(encoding='utf-8').splitlines()
    if sources != sorted(set(sources)) or len(sources) != record['count']:
      raise BuildError(f'Inventory is not sorted/unique or count differs: {variant}')
    if set(sources) != set(record['source_files_sha256']):
      raise BuildError(f'Incomplete source hashes: {variant}')
    for name in sources:
      sha = record['source_files_sha256'][name]
      if digest(safe_file(root, name)) != sha:
        raise BuildError(f'Source hash mismatch: {name}')
      if name in hashes and hashes[name] != sha:
        raise BuildError(f'Conflicting input hashes: {name}')
      hashes[name] = sha
    lists[variant] = sources
  return provenance, lists, hashes, digest(provenance_path)


def destination(name):
  # Preserve the runtime include tree and C++ extensions. Flatten downloads only.
  return 'third_party/' + name[len(DOWNLOADS):] if name.startswith(DOWNLOADS) else name


def add_missing_licenses(root, lists, hashes):
  """Record license supplements omitted by older upstream inventories."""
  supplements = {}
  for variant, sources in lists.items():
    for library in BUNDLED:
      members = [name for name in sources if category(name) == 'bundled/' + library]
      if not members or any(PurePosixPath(name).name.startswith(('LICENSE', 'COPYING')) for name in members):
        continue
      bases = {DOWNLOADS + library if name.startswith(DOWNLOADS) else 'third_party/' + library for name in members}
      found = False
      for base in sorted(bases):
        for filename in ('LICENSE', 'LICENSE.txt', 'COPYING'):
          name = base + '/' + filename
          if not (root / name).exists():
            continue
          source = safe_file(root, name)
          hashes[name] = digest(source)
          sources.append(name)
          entry = supplements.setdefault(name, {'source': name, 'sha256': hashes[name],
                                               'reason': 'License omitted by upstream inventory', 'variants': []})
          entry['variants'].append(variant)
          found = True
      if not found:
        raise BuildError(f'Missing license for bundled {library}')
    sources.sort()
  return list(supplements.values())


def owner(name, variant):
  group = category(name)
  if group == 'unclassified':
    raise BuildError(f'Unclassified path cannot be staged: {name}')
  if group.startswith('external/'):
    return None
  if group.startswith('bundled/'):
    return group
  if name == 'LICENSE':
    return 'package_license'
  if name in UTIL_FILES:
    return 'kernel_utils'
  filename = PurePosixPath(name).name
  if name in TESTING_SUPPORT or name.startswith(MICRO + 'testing/') or filename.startswith('test_helper') or filename.endswith(('_test.h', '_test_helpers.h')):
    return 'testing'
  if group == 'tflm/signal':
    return 'signal'
  return 'kernel/' + variant


def identity(component):
  if component.startswith('bundled/'):
    name = component.split('/')[1]
    return {'Cclass': 'Data Exchange' if name == 'flatbuffers' else 'Data Processing',
            'Cgroup': 'Serialization' if name == 'flatbuffers' else 'Math',
            'Csub': 'gemmlowp fixed-point' if name == 'gemmlowp' else name}
  result = {'Cclass': 'Machine Learning', 'Cgroup': 'LiteRT'}
  if component.startswith('kernel/'):
    result.update(Csub='Kernel', Cvariant=VARIANTS[component.split('/')[1]][0])
  else:
    result['Csub'] = {'kernel_utils': 'Kernel Utils', 'testing': 'Testing', 'signal': 'Signal Library'}[component]
    if component == 'signal':
      result['Cvariant'] = 'Reference'
  return result


def make_mapping(lists, hashes, config):
  entries, components, excluded = {}, {}, {}
  variants = sorted(lists)
  versions = config['compatibility']['config_file_versions']
  for variant, sources in lists.items():
    excluded[variant] = []
    for name in sources:
      component = owner(name, variant)
      if component is None:
        excluded[variant].append(name)
        continue
      if PurePosixPath(name).name.endswith(('_test.cc', '_test.cpp')):
        raise BuildError(f'Test program requires explicit review before staging: {name}')
      target = destination(name)
      key = target.casefold()
      if key in entries and entries[key]['source'] != name:
        raise BuildError(f'Destination collision: {name} and {entries[key]["source"]}')
      entry = entries.setdefault(key, {'source': name, 'destination': target,
                                      'sha256': hashes[name], 'owners': [], 'variants': []})
      if variant not in entry['variants']:
        entry['variants'].append(variant)
      if component not in entry['owners']:
        entry['owners'].append(component)
      if component == 'package_license':
        continue
      current = components.setdefault(component, {'identity': identity(component), 'files': []})
      if target not in current['files']:
        current['files'].append(target)
      if name in CONFIG_FILES:
        version = versions[CONFIG_FILES[name]]
        if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+', version):
          raise BuildError(f'Invalid independent config-file version: {name}')
        entry['config_version'] = version

  # Optional shared components must not silently union architecture-specific code.
  for entry in entries.values():
    shared = any(not name.startswith('kernel/') for name in entry['owners'])
    if shared and sorted(entry['variants']) != variants:
      raise BuildError(f'Shared component differs between variants: {entry["source"]}')
  if not any(e['source'] == 'LICENSE' for e in entries.values()):
    raise BuildError('Missing root LICENSE in inventories')
  for name in BUNDLED:
    component = 'bundled/' + name
    if component in components:
      files = components[component]['files']
      if not any(PurePosixPath(f).name.startswith(('LICENSE', 'COPYING')) for f in files):
        raise BuildError(f'Missing license for bundled {name}')

  # Every supported selection has exactly one Kernel variant plus shared groups.
  selections = {}
  by_target = {e['destination']: e for e in entries.values()}
  for variant in variants:
    selected = [name for name in components if not name.startswith('kernel/') or name == 'kernel/' + variant]
    units = {}
    operators = {}
    for name in selected:
      for path in components[name]['files']:
        if PurePosixPath(path).suffix not in SOURCE_SUFFIXES:
          continue
        if path.casefold() in units:
          raise BuildError(f'Duplicate compilation in {variant}: {path}')
        units[path.casefold()] = path
        source = by_target[path]['source']
        if source.startswith(MICRO + 'kernels/'):
          relative = source[len(MICRO + 'kernels/'):]
          if relative.startswith(('cmsis_nn/', 'ethos_u/')):
            relative = relative.split('/', 1)[1]
          if relative in operators:
            raise BuildError(f'Conflicting kernel implementations in {variant}: {relative}')
          operators[relative] = source
    selections[variant] = {'components': sorted(selected), 'compile_units': sorted(units.values()),
                           'external_requirements': VARIANTS[variant][2]}
  for component in components.values():
    component['files'].sort()
  return {'schema_version': 1, 'release_qualified': False,
          'status': 'draft-staged-payload', 'components': components,
          'selections': selections, 'files': sorted(entries.values(), key=lambda e: e['destination']),
          'excluded_external': excluded,
          'absent_bundled_components': [name for name in BUNDLED if 'bundled/' + name not in components],
          'limitations': ['No PDSC or consumer compile qualification.',
                          'No source rewriting or platform configuration adaptation yet.']}


def stage(root, inventory_dir, output):
  root, inventory_dir = root.resolve(strict=True), inventory_dir.resolve(strict=True)
  output = output.absolute()
  if output.exists() or output.is_symlink():
    raise BuildError('Output already exists; choose a new directory')
  provenance, lists, hashes, provenance_hash = load_inputs(root, inventory_dir)
  supplements = add_missing_licenses(root, lists, hashes)
  supplements += add_testing_support(root, lists, hashes)
  mapping = make_mapping(lists, hashes, provenance['config']['contents'])
  mapping['supplemental_inputs'] = supplements
  mapping['input_provenance_sha256'] = provenance_hash
  mapping['pack'] = provenance['config']['contents'].get('pack')
  mapping['runtime_commit'] = provenance['runtime']['commit']
  scripts = Path(__file__).resolve().parent
  mapping['staging_scripts_sha256'] = {
    name: digest(scripts / name) for name in ('stage_pack.py', 'classify_sources.py', 'build_pack.py')
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='.litert-stage-', dir=output.parent) as temporary:
    result = Path(temporary) / 'result'
    payload = result / 'payload'
    payload.mkdir(parents=True)
    for entry in mapping['files']:
      source = safe_file(root, entry['source'])
      target = payload / entry['destination']
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(source, target)
      if digest(target) != entry['sha256']:
        raise BuildError(f'Source changed during staging: {entry["source"]}')
    actual = {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}
    if actual != {e['destination'] for e in mapping['files']}:
      raise BuildError('Staged payload does not match component mapping')
    if git_state(root, exact_root=True)['commit'] != mapping['runtime_commit']:
      raise BuildError('Runtime HEAD changed during staging')
    if digest(inventory_dir / 'provenance.json') != provenance_hash:
      raise BuildError('Provenance changed during staging')
    shutil.copyfile(inventory_dir / 'provenance.json', result / 'input-provenance.json')
    if digest(result / 'input-provenance.json') != provenance_hash:
      raise BuildError('Provenance changed while copying')
    (result / 'component-map.json').write_text(json.dumps(mapping, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    if output.exists() or output.is_symlink():
      raise BuildError('Output appeared during staging')
    result.rename(output)
  return mapping


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tflm-root', required=True, type=Path)
  parser.add_argument('--inventory-dir', required=True, type=Path)
  parser.add_argument('--out', required=True, type=Path)
  args = parser.parse_args(argv)
  try:
    result = stage(args.tflm_root, args.inventory_dir, args.out)
  except (BuildError, OSError, ValueError, KeyError, TypeError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1
  print(f"Staged {len(result['files'])} files in {len(result['components'])} components -> {args.out}")
  print('Draft only: no PDSC, archive or consumer compile qualification.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
