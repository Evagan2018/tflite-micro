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
"""Build draft backend inventories, classification and provenance, not a releasable pack."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from file_lists import FileListError, GENERATOR, generate_sources
from classify_sources import BUNDLED, EXTERNAL, classify


class BuildError(RuntimeError):
  """Configuration, revision recording or output preparation failed."""


VARIANTS = {
  'reference': ('Reference', {'TARGET': 'cortex_m_generic', 'TARGET_ARCH': 'cortex-m7'}, ['cmsis']),
  'cmsis_nn': ('CMSIS-NN', {'TARGET': 'cortex_m_generic', 'TARGET_ARCH': 'cortex-m55',
                         'OPTIMIZED_KERNEL_DIR': 'cmsis_nn'}, ['cmsis', 'cmsis_nn']),
  'ethos_u': ('Ethos-U', {'TARGET': 'cortex_m_generic', 'TARGET_ARCH': 'cortex-m55',
                       'OPTIMIZED_KERNEL_DIR': 'cmsis_nn', 'CO_PROCESSOR': 'ethos_u',
                       'ETHOSU_ARCH': 'u55'}, ['cmsis', 'cmsis_nn', 'ethos_u_driver']),
}


def read_config(path, selected=('reference',)):
  """Validate only the configuration consumed by this draft milestone."""
  try:
    import yaml
  except ImportError as error:
    raise BuildError('Install the builder requirements.txt with pip first') from error

  class UniqueLoader(yaml.SafeLoader):
    pass

  def mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
      key = loader.construct_object(key_node)
      if not isinstance(key, str) or key in result:
        raise BuildError(f'Duplicate or non-string YAML key: {key!r}')
      result[key] = loader.construct_object(value_node)
    return result

  UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping,
  )
  try:
    config = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
  except yaml.YAMLError as error:
    raise BuildError(f'Invalid YAML: {error}') from error

  def require(condition, message):
    if not condition:
      raise BuildError(message)

  require(isinstance(config, dict), 'Configuration must be a mapping')
  require(type(config.get('schema_version')) is int and
          config['schema_version'] == 1, 'Only schema_version: 1 is supported')
  require(config.get('status') == 'draft', 'This builder supports draft mode only')
  pack = config.get('pack')
  require(isinstance(pack, dict), 'pack must be a mapping')
  require(pack.get('vendor') == 'TensorFlow' and pack.get('name') == 'LiteRT',
          'Expected pack identity TensorFlow::LiteRT')
  require(isinstance(pack.get('version'), str) and
          re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)',
                       pack['version']), 'pack.version must be an x.y.z string')
  variants = config.get('variants')
  require(isinstance(variants, dict), 'variants must be a mapping')
  options_by_variant = {}
  requirements = config.get('requirements')
  require(isinstance(requirements, dict), 'requirements must be a mapping')
  for name in selected:
    require(name in VARIANTS, f'Unknown variant: {name}')
    label, expected_options, expected_requires = VARIANTS[name]
    variant = variants.get(name)
    require(isinstance(variant, dict), f'variants.{name} must be a mapping')
    require(variant.get('Cvariant') == label, f'Expected {label} Cvariant')
    require(variant.get('make_options') == expected_options,
            f'This milestone supports only {label} options: {expected_options}')
    requires = variant.get('requires')
    require(isinstance(requires, list) and
            all(isinstance(key, str) and isinstance(requirements.get(key), dict)
                for key in requires) and sorted(requires) == sorted(expected_requires),
            f'{name}.requires must contain exactly {expected_requires}')
    options_by_variant[name] = tuple(f'{key}={value}' for key, value in expected_options.items())
  third_party = config.get('third_party')
  require(isinstance(third_party, dict), 'third_party must be a mapping')
  for key, expected in (('bundled', BUNDLED), ('external', EXTERNAL)):
    value = third_party.get(key)
    require(isinstance(value, list) and all(isinstance(item, str) for item in value)
            and sorted(value) == sorted(expected), f'Unsupported third_party.{key} policy')
  source_lists = config.get('source_lists')
  require(isinstance(source_lists, dict) and
          source_lists.get('generator') == GENERATOR.as_posix() and
          source_lists.get('print_src_files') is True and
          source_lists.get('no_copy') is True,
          'source_lists must select the upstream generator with both flags true')
  return config, options_by_variant


def git_state(directory, exact_root=False):
  """Record Git identity without altering configuration or trusting parent roots."""
  def git(*arguments):
    result = subprocess.run(
      ['git', '-C', str(directory), *arguments], capture_output=True,
      text=True, encoding='utf-8', errors='replace', check=False,
    )
    if result.returncode:
      raise BuildError(f'Cannot record Git state for {directory}: {result.stderr.strip()}')
    return result.stdout.strip()

  root = Path(git('rev-parse', '--show-toplevel')).resolve()
  if exact_root and root != directory:
    raise BuildError('--tflm-root must be the Git checkout root')
  return {
    'root': str(root),
    'commit': git('rev-parse', 'HEAD'),
    'status_porcelain': git('status', '--porcelain=v1', '--untracked-files=all'),
  }


def digest(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(tflm_root, output, config_path, selected=('reference',)):
  config_path = config_path.resolve(strict=True)
  config_hash = digest(config_path)
  config, options_by_variant = read_config(config_path, selected)
  root = tflm_root.resolve(strict=True)
  output = output.absolute()
  if output.exists() or output.is_symlink():
    raise BuildError('Output already exists; choose a new output directory')
  runtime_before = git_state(root, exact_root=True)
  scripts = Path(__file__).resolve().parent
  # Standalone copies are useful for development, but are not a builder revision.
  try:
    builder = git_state(scripts)
  except BuildError as error:
    builder = {'commit': None, 'unavailable_reason': str(error)}
  builder['files_sha256'] = {
    name: digest(scripts / name) for name in ('build_pack.py', 'file_lists.py', 'classify_sources.py')
  }
  generator_hash = digest(root / GENERATOR)
  inventories = {}
  records = {}
  classifications = {}
  observed_hashes = {}
  for name, options in options_by_variant.items():
    sources = generate_sources(root, make_options=options)
    hashes = {path: digest(root / path) for path in sources}
    for path, sha in hashes.items():
      if path in observed_hashes and observed_hashes[path] != sha:
        raise BuildError(f'Shared input changed between variants: {path}')
    observed_hashes.update(hashes)
    inventory = '\n'.join(sources) + '\n'
    filename = f'{name}-sources.txt'
    inventories[filename] = inventory
    classifications[name] = classify(sources)
    records[name] = {
      'file': filename, 'count': len(sources), 'make_options': list(options),
      'sha256': hashlib.sha256(inventory.encode('utf-8')).hexdigest(),
      'source_files_sha256': hashes,
      'classification_counts': classifications[name]['counts'],
    }
  # Later generator runs must not silently change earlier variant inputs.
  if any(digest(root / path) != sha for path, sha in observed_hashes.items()):
    raise BuildError('Source inputs changed during generation')
  classification_text = json.dumps({
    'schema_version': 1, 'variants': classifications,
    'note': 'Ownership classification only, not final PDSC components or payload approval. '
            'External files remain in raw inventories for audit; do not bundle them. '
            'Unclassified paths require review before staging.',
  }, indent=2, sort_keys=True) + '\n'
  runtime = git_state(root, exact_root=True)
  if runtime['commit'] != runtime_before['commit']:
    raise BuildError('Runtime HEAD changed during generation; rerun with a stable checkout')
  if digest(config_path) != config_hash or digest(root / GENERATOR) != generator_hash:
    raise BuildError('Configuration or generator changed during generation')
  provenance = {
    'schema_version': 2,
    'artifact_kind': 'draft-backend-inventories',
    'release_qualified': False,
    'limitations': [
      'No PDSC, archive, example staging or release-policy qualification yet.',
      'Configured release candidates are recorded, not fetched or verified.',
      'File hashes describe observed inputs, not a dependency lockfile.',
    ],
    'pack': config['pack'],
    'runtime': runtime,
    'builder': builder,
    'config': {'path': str(config_path), 'sha256': config_hash, 'contents': config},
    'generation': {
      'generator': GENERATOR.as_posix(), 'generator_sha256': generator_hash,
      'selected_variants': list(selected), 'flags': ['--no_copy', '--print_src_files'],
      'python_version': sys.version,
    },
    'inventories': records,
    'classification': {
      'file': 'classification.json',
      'sha256': hashlib.sha256(classification_text.encode('utf-8')).hexdigest(),
    },
  }
  # Retain the original convenience field for single-variant callers.
  if len(selected) == 1:
    provenance['inventory'] = records[selected[0]]
  output.parent.mkdir(parents=True, exist_ok=True)
  # Publish all outputs together, leaving no partial destination on failure.
  with tempfile.TemporaryDirectory(prefix='.litert-', dir=output.parent) as scratch:
    staged = Path(scratch) / 'result'
    staged.mkdir()
    for filename, inventory in inventories.items():
      (staged / filename).write_text(inventory, encoding='utf-8', newline='\n')
    (staged / 'classification.json').write_text(classification_text, encoding='utf-8', newline='\n')
    (staged / 'provenance.json').write_text(
      json.dumps(provenance, indent=2, sort_keys=True) + '\n',
      encoding='utf-8', newline='\n',
    )
    if output.exists() or output.is_symlink():
      raise BuildError('Output appeared during generation; refusing to replace it')
    staged.rename(output)
  return provenance


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tflm-root', type=Path, required=True)
  parser.add_argument('--out', type=Path, required=True, help='New output directory')
  parser.add_argument('--config', type=Path,
                      default=Path(__file__).resolve().parents[1] / 'pack.yml')
  parser.add_argument('--variant', choices=(*VARIANTS, 'all'), default='reference',
                      help='Backend inventory to generate (default: reference)')
  args = parser.parse_args(argv)
  try:
    selected = tuple(VARIANTS) if args.variant == 'all' else (args.variant,)
    result = build(args.tflm_root, args.out, args.config, selected)
  except (BuildError, FileListError, OSError, ValueError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1
  for name, record in result['inventories'].items():
    print(f"Draft {name} inventory: {record['count']} files -> {args.out / record['file']}")
    unknown = record['classification_counts']['unclassified']
    if unknown:
      print(f'warning: {name}: {unknown} unclassified paths require review', file=sys.stderr)
  print('Not release-qualified; no .pack archive generated.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
