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
"""Create a development PDSC and copied payload from a verified staging map.

Draft mode is explicit: unresolved release requirements are reported, not resolved.
No release archive is created and no input files are modified.
"""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

from build_pack import BuildError, digest
from stage_pack import MICRO, TESTING_SUPPORT, safe_file


DRIVER_CANDIDATE = {'Cclass': 'Machine Learning', 'Cgroup': 'NPU Support',
                    'Csub': 'Ethos-U Driver'}
TIME_INCLUDE = ('#ifdef CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE\n'
                '#include CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE\n#endif')


def output_name(entry):
  """Keep project configuration templates separate from regular include paths."""
  if 'config_version' in entry:
    return 'config/' + PurePosixPath(entry['destination']).stem + '.cpp'
  return entry['destination']


def adapt_source(name, data):
  """Narrow, audited adaptations of copied files only."""
  changes = []
  if name == MICRO + 'cortex_m_generic/micro_time.cc':
    text = data.decode('utf-8')
    if text.count(TIME_INCLUDE) != 1:
      raise BuildError('Unrecognized micro_time device-include block; review required')
    text = text.replace(TIME_INCLUDE, '#include "RTE_Components.h"\n#include CMSIS_device_header')
    changes.append('Use CMSIS device-header selection from RTE_Components.h')
    data = text.encode('utf-8')
  if name.startswith(MICRO + 'kernels/cmsis_nn/') and name.endswith(('.cc', '.h')):
    text = data.decode('utf-8')
    text, count = re.subn(r'(#include\s+["<])(?:CMSIS/NN/Include/|Include/)(arm_nn[^">]+)([">])', r'\1\2\3', text)
    if count:
      changes.append(f'Use CMSIS-NN pack include paths ({count} includes)')
    data = text.encode('utf-8')
  return data, changes


def require_component(parent, component):
  return ET.SubElement(parent, 'require', component['identity'])


def render(mapping, config, release_date):
  testing_files = mapping['components'].get('testing', {}).get('files', [])
  if not set(TESTING_SUPPORT).issubset(testing_files):
    raise BuildError('Incomplete Testing support; regenerate staging with the current builder')
  """Generate escaped XML, with all substitutions handled structurally."""
  date.fromisoformat(release_date)
  pack = config['pack']
  version = pack['version'] + '-dev.0'
  if pack['vendor'] != 'TensorFlow' or pack['name'] != 'LiteRT':
    raise BuildError('Unexpected pack identity')
  if not re.fullmatch(r'\d+\.\d+\.\d+-dev\.0', version):
    raise BuildError('Invalid development version')
  root = ET.Element('package', {'schemaVersion': '1.7.9',
    'xmlns:xs': 'http://www.w3.org/2001/XMLSchema-instance',
    'xs:noNamespaceSchemaLocation': 'PACK.xsd'})
  root.append(ET.Comment(' DEVELOPMENT ONLY - see pdsc-report.json for unresolved release requirements '))
  for name, value in [('vendor', pack['vendor']), ('name', pack['name']),
                      ('description', pack['description']),
                      ('url', 'https://github.com/MDK-Packs/tensorflow-pack/releases/download/'),
                      ('license', 'LICENSE')]:
    ET.SubElement(root, name).text = value
  ET.SubElement(root, 'repository', {'type': 'git'}).text = 'https://github.com/MDK-Packs/tensorflow-pack.git'
  release = ET.SubElement(ET.SubElement(root, 'releases'), 'release', {'version': version, 'date': release_date})
  release.text = 'Development preview only. Runtime ' + mapping['runtime_commit'] + '. Not release-qualified.'
  blockers = []
  packages = ET.SubElement(ET.SubElement(root, 'requirements'), 'packages')
  selected_ids = sorted({dep for selection in mapping['selections'].values() for dep in selection['external_requirements']})
  for name in selected_ids:
    requirement = config['requirements'][name]
    attributes = {'vendor': requirement['vendor'], 'name': requirement['name']}
    if requirement['version_range']:
      attributes['version'] = requirement['version_range']
    else:
      blockers.append(f'{name}: package version range unresolved; draft declares dependency without a version constraint')
    if requirement.get('status') != 'verified':
      blockers.append(f'{name}: compatibility is not verified')
    ET.SubElement(packages, 'package', attributes)
  ET.SubElement(ET.SubElement(root, 'taxonomy'), 'description', {'Cclass': 'Machine Learning'}).text = 'Software components for machine learning'
  conditions = ET.SubElement(root, 'conditions')
  components = mapping['components']
  for key in sorted(components):
    if key.startswith('bundled/'):
      continue
    condition = ET.SubElement(conditions, 'condition', {'id': 'LiteRT ' + key.replace('/', ' ')})
    if key.startswith('kernel/'):
      variant = key.split('/')[1]
      require_component(condition, components['kernel_utils'])
      for dependency in sorted(components):
        if dependency.startswith('bundled/'):
          require_component(condition, components[dependency])
      for name in mapping['selections'][variant]['external_requirements']:
        selector = config['requirements'][name].get('component')
        if selector is None:
          if name != 'ethos_u_driver':
            raise BuildError(f'No draft selector available for {name}')
          selector = DRIVER_CANDIDATE
          blockers.append('ethos_u_driver: legacy component selector used as draft candidate; 2.x selector remains unverified')
        ET.SubElement(condition, 'require', dict(selector))
    elif key == 'kernel_utils':
      ET.SubElement(condition, 'require', dict(config['requirements']['cmsis']['component']))
    else:
      ET.SubElement(condition, 'require', {'Cclass': 'Machine Learning', 'Cgroup': 'LiteRT', 'Csub': 'Kernel'})
      if key == 'signal' and 'bundled/kissfft' in components:
        require_component(condition, components['bundled/kissfft'])

  indexed = {entry['destination']: entry for entry in mapping['files']}
  container = ET.SubElement(root, 'components')
  for key, component in sorted(components.items()):
    attributes = dict(component['identity'], Cversion=version)
    if not key.startswith('bundled/'):
      attributes['condition'] = 'LiteRT ' + key.replace('/', ' ')
    node = ET.SubElement(container, 'component', attributes)
    ET.SubElement(node, 'description').text = 'LiteRT ' + key.replace('/', ' ') + ' (development preview)'
    if key.startswith('kernel/'):
      ET.SubElement(node, 'RTE_Components_h').text = '\n' + '\n'.join('#define ' + value for value in config['compatibility']['defines']) + '\n'
      flags = '#define TF_LITE_STATIC_MEMORY 1\n'
      if key != 'kernel/reference':
        flags += '#define CMSIS_NN 1\n'
      ET.SubElement(node, 'Pre_Include_Global_h').text = '\n' + flags
    files = ET.SubElement(node, 'files')
    includes = set()
    if key.startswith('bundled/'):
      library = key.split('/')[1]
      includes.add('third_party/' + library + ('/include/' if library == 'flatbuffers' else '/'))
      if library == 'kissfft':
        includes.update(str(PurePosixPath(p).parent) + '/' for p in component['files'] if p.endswith('.h'))
    else:
      includes.add('./')
    for path in sorted(component['files']):
      entry = indexed[path]
      suffix = PurePosixPath(path).suffix
      category = {'.c': 'sourceC', '.cc': 'sourceCpp', '.cpp': 'sourceCpp',
                  '.h': 'header', '.hpp': 'header', '.s': 'sourceAsm', '.S': 'sourceAsm'}.get(suffix, 'doc')
      attrs = {'category': category, 'name': output_name(entry)}
      if 'config_version' in entry:
        attrs.update(attr='config', version=entry['config_version'])
      ET.SubElement(files, 'file', attrs)
    for path in sorted(includes):
      ET.SubElement(files, 'file', {'category': 'include', 'name': path})
  blockers += ['Full release baseline and source pins are not qualified.',
               'Consumer compile/link and target testing have not passed.',
               'Example staging and final archive assembly remain publisher work.',
               'Platform timing CPU macros and counter behavior require target validation.']
  ET.indent(root, space='  ')
  return ET.tostring(root, encoding='utf-8', xml_declaration=True), sorted(set(blockers))


def generate(stage_dir, output, release_date):
  stage_dir = stage_dir.resolve(strict=True)
  payload = (stage_dir / 'payload').resolve(strict=True)
  output = output.absolute()
  if output.exists() or output.is_symlink():
    raise BuildError('Output already exists; choose a new directory')
  map_path = safe_file(stage_dir, 'component-map.json')
  mapping = json.loads(map_path.read_text(encoding='utf-8'))
  provenance_path = safe_file(stage_dir, 'input-provenance.json')
  if digest(provenance_path) != mapping['input_provenance_sha256']:
    raise BuildError('Input provenance hash mismatch')
  provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
  if mapping['runtime_commit'] != provenance['runtime']['commit']:
    raise BuildError('Runtime identity mismatch')
  seen = set()
  destinations = set()
  for entry in mapping['files']:
    path = entry['destination']
    if path.casefold() in seen:
      raise BuildError(f'Duplicate payload path: {path}')
    seen.add(path.casefold())
    target_name = output_name(entry).casefold()
    if target_name in destinations:
      raise BuildError(f'Duplicate adapted payload path: {target_name}')
    destinations.add(target_name)
    if digest(safe_file(payload, path)) != entry['sha256']:
      raise BuildError(f'Payload hash mismatch: {path}')
  indexed = {entry['destination']: entry for entry in mapping['files']}
  for key, component in mapping['components'].items():
    if len(component['files']) != len(set(component['files'])):
      raise BuildError(f'Duplicate component files: {key}')
    for path in component['files']:
      if path not in indexed or key not in indexed[path]['owners']:
        raise BuildError(f'Invalid component ownership: {key}: {path}')
  xml, blockers = render(mapping, provenance['config']['contents'], release_date)
  report = {'schema_version': 1, 'release_qualified': False, 'blockers': blockers,
            'component_map_sha256': digest(map_path), 'runtime_commit': mapping['runtime_commit'],
            'generator_sha256': digest(Path(__file__)), 'adaptations': [], 'output_sha256': {},
            'configuration_templates': {entry['destination']: output_name(entry)
                                        for entry in mapping['files'] if 'config_version' in entry}}
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='.litert-pdsc-', dir=output.parent) as temporary:
    result = Path(temporary) / 'result'
    result.mkdir()
    for entry in mapping['files']:
      path = entry['destination']
      source = safe_file(payload, path)
      original = source.read_bytes()
      if hashlib.sha256(original).hexdigest() != entry['sha256']:
        raise BuildError(f'Input changed during copying: {path}')
      data, changes = adapt_source(path, original)
      target = result / output_name(entry)
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(data)
      report['output_sha256'][output_name(entry)] = digest(target)
      if changes:
        report['adaptations'].append({'file': path, 'changes': changes, 'input_sha256': entry['sha256'], 'output_sha256': digest(target)})
      elif digest(target) != entry['sha256']:
        raise BuildError(f'Input changed during copying: {path}')
    pdsc = result / 'TensorFlow.LiteRT.pdsc'
    pdsc.write_bytes(xml + b'\n')
    for item in ET.fromstring(xml).findall('.//files/file'):
      name = item.attrib['name']
      if item.attrib['category'] == 'include':
        if not (result / name).is_dir():
          raise BuildError(f'Include directory missing: {name}')
      else:
        safe_file(result.resolve(), name)
    report['pdsc_sha256'] = digest(pdsc)
    (result / 'pdsc-report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    if output.exists() or output.is_symlink():
      raise BuildError('Output appeared during generation')
    result.rename(output)
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--stage-dir', required=True, type=Path)
  parser.add_argument('--out', required=True, type=Path)
  parser.add_argument('--date', required=True, help='Explicit development release date (YYYY-MM-DD)')
  parser.add_argument('--draft', required=True, action='store_true', help='Acknowledge unresolved release requirements')
  args = parser.parse_args(argv)
  try:
    report = generate(args.stage_dir, args.out, args.date)
  except (BuildError, OSError, ValueError, KeyError, TypeError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1
  print(f'Development PDSC -> {args.out / "TensorFlow.LiteRT.pdsc"}')
  print(f"Not release-qualified: {len(report['blockers'])} items remain; see pdsc-report.json")
  return 0


if __name__ == '__main__':
  sys.exit(main())
