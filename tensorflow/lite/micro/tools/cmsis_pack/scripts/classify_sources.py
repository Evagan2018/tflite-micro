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
"""Classify normalized inventory paths; this is not a final payload allowlist."""

from pathlib import PurePosixPath


DOWNLOADS = 'tensorflow/lite/micro/tools/make/downloads/'
BUNDLED = ('flatbuffers', 'gemmlowp', 'kissfft', 'ruy')
EXTERNAL = ('cmsis', 'cmsis_nn', 'ethos_u_core_driver')
LEGACY_RUNTIME = {
  'tensorflow/compiler/mlir/lite/' + name for name in (
    'core/api/error_reporter.cc', 'core/api/error_reporter.h',
    'core/c/builtin_op_data.h', 'core/c/tflite_types.h',
    'kernels/internal/compatibility_macros.h', 'schema/schema_generated.h',
    'schema/schema_utils.cc', 'schema/schema_utils.h',
  )
}


def category(path):
  """Use directory boundaries so similarly named dependencies stay unknown."""
  parts = PurePosixPath(path).parts
  if not parts or path.startswith('/') or '\\' in path or '..' in parts:
    raise ValueError(f'Expected normalized checkout-relative path: {path}')
  if path.startswith(DOWNLOADS):
    if path.startswith(DOWNLOADS + 'cmsis/Cortex_DFP/'):
      return 'external/cortex_dfp'
    dependency = path[len(DOWNLOADS):].split('/')[0]
    if dependency in BUNDLED:
      return 'bundled/' + dependency
    if dependency in EXTERNAL:
      return 'external/' + dependency
    return 'unclassified'
  # Some runtime revisions keep third-party headers directly in the checkout.
  if path.startswith('third_party/'):
    dependency = parts[1] if len(parts) > 1 else ''
    if dependency in BUNDLED:
      return 'bundled/' + dependency
    return 'unclassified'
  if path == 'LICENSE':
    return 'tflm/license'
  if path in LEGACY_RUNTIME:
    return 'tflm/runtime_and_support'
  if path.startswith('signal/'):
    return 'tflm/signal'
  if path.startswith('tensorflow/lite/micro/tools/'):
    return 'unclassified'
  if path.startswith('tensorflow/lite/micro/kernels/cmsis_nn/'):
    return 'tflm/cmsis_nn_integration'
  if path.startswith('tensorflow/lite/micro/kernels/ethos_u/'):
    return 'tflm/ethos_u_integration'
  if path.startswith('tensorflow/lite/micro/kernels/'):
    return 'tflm/kernels'
  if path.startswith('tensorflow/lite/'):
    return 'tflm/runtime_and_support'
  return 'unclassified'


def classify(sources):
  groups = {'unclassified': []}
  for path in sorted(set(sources)):
    groups.setdefault(category(path), []).append(path)
  return {
    'groups': groups,
    'counts': {key: len(paths) for key, paths in sorted(groups.items())},
    'all_paths_classified': not groups['unclassified'],
    'payload_approved': False,
  }
