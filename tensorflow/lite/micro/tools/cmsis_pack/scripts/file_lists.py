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
"""Generate a checked Reference source inventory using TFLM's tree generator.

Paths include sources, headers and any other files returned by the upstream
generator. This is an inventory, not a filtered pack payload. Dependency
classification and additional backends will be implemented separately.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


GENERATOR = Path(
  'tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py'
)
REFERENCE_MAKE_OPTIONS = (
  'TARGET=cortex_m_generic',
  'TARGET_ARCH=cortex-m7',
)


class FileListError(RuntimeError):
  """Source generation or inventory validation failed."""


def normalize_sources(source_text: str, tflm_root: Path) -> list[str]:
  """Return sorted, unique, checkout-relative POSIX paths to existing files.

  The upstream format is whitespace-delimited and cannot represent filenames
  containing whitespace. Resolve symlinks before checking containment so an
  external source cannot be silently included in a supposedly self-contained
  checkout inventory.
  """
  root = Path(tflm_root).resolve(strict=True)
  if not root.is_dir():
    raise FileListError(f'TFLM root is not a directory: {root}')
  entries = source_text.split()
  if not entries:
    raise FileListError('The upstream generator returned an empty source list')

  sources = set()
  for entry in entries:
    candidate = Path(entry)
    if not candidate.is_absolute():
      candidate = root / candidate
    try:
      resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
      raise FileListError(f'Source path cannot be resolved: {entry}') from error
    try:
      relative = resolved.relative_to(root)
    except ValueError as error:
      raise FileListError(f'Source path is outside TFLM root: {entry}') from error
    if not resolved.is_file():
      raise FileListError(f'Source path is not a regular file: {entry}')
    sources.add(relative.as_posix())
  return sorted(sources)


def generate_reference_sources(
  tflm_root: Path, make_options: tuple[str, ...] = REFERENCE_MAKE_OPTIONS,
) -> list[str]:
  """Backward-compatible Reference entry point."""
  return generate_sources(tflm_root, make_options)


def generate_sources(tflm_root: Path, make_options: tuple[str, ...]) -> list[str]:
  """Invoke the existing generator without copying its output tree.

  The generator still runs third_party_downloads and may populate checkout
  caches. Use the current Python interpreter and run Make from the selected
  checkout, independently of the caller's current directory.
  """
  try:
    root = Path(tflm_root).resolve(strict=True)
  except OSError as error:
    raise FileListError(f'TFLM root does not exist: {tflm_root}') from error
  if not root.is_dir():
    raise FileListError(f'TFLM root is not a directory: {root}')
  generator = root / GENERATOR
  if not generator.is_file():
    raise FileListError(f'Upstream source generator is missing: {generator}')

  # output_dir is mandatory upstream, even with --no_copy. A temporary location
  # avoids treating a source directory as a potential generation destination.
  with tempfile.TemporaryDirectory(prefix='litert-file-lists-') as scratch:
    command = [
      sys.executable,
      str(generator),
      scratch,
      '--no_copy',
      '--print_src_files',
      '--makefile_options=' + ' '.join(make_options),
    ]
    try:
      result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='strict',
        check=False,
      )
    except (OSError, UnicodeError) as error:
      raise FileListError(f'Could not run upstream generator: {error}') from error
  if result.returncode:
    diagnostic = result.stderr.strip() or result.stdout.strip()
    raise FileListError(
      f'Upstream generator failed (exit {result.returncode}).\n{diagnostic}'
    )
  # Keep diagnostic output separate from the machine-readable source list.
  if result.stderr:
    print(result.stderr, end='', file=sys.stderr)
  return normalize_sources(result.stdout, root)


def write_sources(sources: list[str], output: Path) -> None:
  """Atomically replace an inventory only after generation and checks succeed."""
  output = Path(output).absolute()
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(
      mode='w', encoding='utf-8', newline='\n', dir=output.parent,
      prefix=f'.{output.name}.', suffix='.tmp', delete=False,
    ) as stream:
      temporary = Path(stream.name)
      stream.write('\n'.join(sources) + '\n')
    os.replace(temporary, output)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--tflm-root', required=True, type=Path, help='Selected TFLM checkout root'
  )
  parser.add_argument(
    '--out', type=Path,
    help='Inventory file; omit to print one relative path per line to stdout',
  )
  args = parser.parse_args(argv)
  try:
    sources = generate_reference_sources(args.tflm_root)
    if args.out is None:
      print('\n'.join(sources))
    else:
      write_sources(sources, args.out)
  except (FileListError, OSError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())
