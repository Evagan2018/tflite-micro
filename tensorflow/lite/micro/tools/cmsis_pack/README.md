# TensorFlow::LiteRT CMSIS-Pack builder

## Status

The draft backend-inventory milestone is implemented: `scripts/build_pack.py` validates
its configuration subset, invokes `scripts/file_lists.py`, and writes selected backend inventories, ownership classification
and input provenance. It does not yet generate a PDSC or `.pack` archive.
Reference, CMSIS-NN and Ethos-U inventories are supported. `stage_pack.py` maps
components and stages a checked draft payload from those inventories. An explicit
`--draft` PDSC generator now creates development metadata and applies narrow
platform adaptations to copied files. Consumer and release qualification remain
planned.

## Purpose

Build the proposed `TensorFlow::LiteRT` CMSIS-Pack directly from a selected
TensorFlow Lite for Microcontrollers (TFLM) checkout. The initial planned pack
version is `1.0.0`; pack versions are independent of TensorFlow and Ethos-U
release numbers.

The pack is intended to contain:

- TFLM runtime components and Reference, CMSIS-NN and Ethos-U kernel variants.
- Kernel utilities, testing support and the signal library.
- flatbuffers, gemmlowp, kissfft and ruy as components within the same pack.
- Required licenses, configuration files and provenance documentation.
- The sample application, staged through publisher integration into the final pack.

CMSIS-NN and the Ethos-U core driver remain external pack requirements. Their
sources must not be bundled into this pack. The Ethos-U kernel variant must
also declare its CMSIS-NN component dependency.

## Responsibilities and repository ownership

| Area | Responsibility | Development and intended home |
| --- | --- | --- |
| Builder | Obtain source lists, stage runtime files, apply reviewed patches, generate pack metadata and validate the core pack. | This directory in the `Evagan2018/tflite-micro` fork; a reduced contribution may later be proposed to `tensorflow/tflite-micro`. |
| Publisher | Resolve and pin release inputs, check out the builder/runtime revisions, build the separate driver pack, orchestrate tests and publish release assets and index updates. | `Evagan2018/tensorflow-pack` for development; `MDK-Packs/tensorflow-pack` is the intended publisher. |
| Example source | Maintain the reference application's solution, model, board layers and migration changes. | `MatthiasHertelArm/CMSIS-LiteRT`. |
| Example staging | Fetch the pinned example, filter generated files, stamp compatible pack requirements, add example metadata and test the assembled pack. | Publisher integration; the exact staging implementation remains to be agreed. |

The builder must be usable without GitHub publishing credentials or the example
repository. It does not create GitHub releases, update the Keil pack index,
deprecate old packs or build `ARM::ethos-u-core-driver`.

The original renewal plan listed `stage_example.py` alongside the builder while
also excluding example integration from the proposed upstream contribution.
This README treats example staging as publisher integration. Its implementation
location and the final archive assembly interface still need to be settled.
There should be one runtime builder code path, rather than separate fork and
publisher implementations.

## Directory layout

All paths below are relative to this directory:

```text
tensorflow/lite/micro/tools/cmsis_pack/
|-- README.md                         This document
|-- pack.yml                          Draft pack configuration
|-- requirements.txt                  Python builder dependency
|-- scripts/
|   |-- build_pack.py                 Backend inventory entry point
|   |-- file_lists.py                 Upstream generator adapter
|   |-- classify_sources.py           Ownership classification
|   |-- stage_pack.py                 Checked draft component mapping and staging
|   `-- generate_pdsc.py              Development PDSC and adapted payload
|-- contributions/
|   |-- add/Documentation/            Planned documentation payload
|   `-- patches/                      Planned reviewed patches
`-- test/
    |-- test_build_pack.py            CLI and provenance tests
    |-- test_file_lists.py            Adapter tests
    |-- test_classify_sources.py      Ownership boundary tests
    |-- test_stage_pack.py            Payload integrity and selection tests
    |-- test_generate_pdsc.py         Development PDSC and adaptation tests
    |-- validate_pack.py              Planned pack validation
    `-- smoke/                        Planned consumer build projects
```

The README, configuration, requirements, five scripts and their tests exist.
XML is generated structurally with ElementTree; no string-substitution template
is needed. Archive and full release-validation work remain planned. Release
history generation and its publisher interface also remain to be agreed.

## Inputs

The intended inputs are:

1. **TFLM checkout:** an explicit repository root supplied through
   `--tflm-root`. Record the full runtime commit SHA and whether the checkout
   contains local modifications.
2. **Builder revision:** record the builder commit separately when it differs
   from the selected runtime revision. Run the builder from its own checkout and supply the runtime
   checkout separately through `--tflm-root`.
3. **Pack configuration:** `pack.yml` proposes vendor, name, version,
   component requirements, backend make options and third-party classification.
   Its field layout is provisional. The entry point validates the draft
   identity, version, selected backend settings, requirement references,
   third-party classification policy and generator flags.
   Other configuration sections are recorded but not semantically validated.
   `requirements` contains compatible consumer ranges, while
   `release_baseline` records unverified candidate release inputs. A `null`
   value means unresolved, not permission to select the latest version.
   The CMSIS range and driver component selectors remain unresolved. The
   proposed `release_policy` must be enforced by future code before release
   artifacts can be qualified; it does not prevent draft source inventories.
4. **Third-party inputs:** source dependencies selected by the chosen runtime
   revision, together with their licenses and recorded provenance.
5. **Contributions:** reviewed patches and additional documentation from this
   directory.
6. **Output location:** an explicit writable directory supplied through `--out`.

For release builds, the publisher resolves the Ethos-U manifest and supplies
immutable inputs. The proposed Ethos-U `26.08` / TFLM `90b983c2` baseline is not
yet verified; resolve the manifest commit and full source SHAs before treating
it as a qualified release baseline. Do not silently substitute current `main`.

## Source-list generation

The Reference adapter uses the existing
[`create_tflm_tree.py`](../project_generation/create_tflm_tree.py) with
`--print_src_files` and `--no_copy`. Keep the adapter separate from pack
generation so that a future source-list mechanism can replace it.

The proposed backend settings from the renewal plan are:

| Backend | Make options to verify against the selected runtime revision |
| --- | --- |
| Reference | `TARGET=cortex_m_generic TARGET_ARCH=cortex-m7` |
| CMSIS-NN | `TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn` |
| Ethos-U | CMSIS-NN options plus `CO_PROCESSOR=ethos_u ETHOSU_ARCH=u55` |

The adapter resolves paths, rejects missing/non-file entries and paths outside
the checkout (including symlinks escaping it), removes duplicates and sorts the
result. Output uses checkout-relative paths with forward slashes. The upstream
list includes sources, headers and potentially other dependency files; it is
not a list of compilation units only. The ownership report separates bundled libraries from external dependencies.
It does not yet approve files for staging or assign final PDSC components. These source-list settings are
not the complete consumer compile/test matrix.

In the current source-generation script, even `--no_copy` invokes the Makefile's
`third_party_downloads` target before listing files. It prevents copying the
output tree; it does not prevent dependency downloads or writes to dependency
caches in the checkout. Plan for network access when dependencies are absent.

## How to run

### Reference source inventory (implemented)

Run in a Linux build environment such as WSL Ubuntu with the upstream
Makefile's dependencies installed. For this Windows checkout, the WSL invocation
is shown below. The adapter itself uses only the Python standard library.

```sh
TFLM_ROOT=/mnt/c/working/Git/github_com/MDK-Packs/WSP__tflite-micro/wk4626/tflite-micro
python3 "$TFLM_ROOT/tensorflow/lite/micro/tools/cmsis_pack/scripts/file_lists.py" \
  --tflm-root "$TFLM_ROOT" \
  --out /tmp/litert-reference-sources.txt
```

`--out` here is an inventory **file**, not the future pack builder's output
directory. Omit it to print the inventory to standard output. Each line contains
one file path. Generation or validation failure returns a nonzero exit status
and preserves any existing output inventory; a new inventory is written only
after all paths pass validation. Upstream diagnostic messages go to standard
error. No pack is created and source files are not staged or patched.

The command can be launched from any directory. It runs the upstream generator
from `--tflm-root` with the Reference options shown above. The standalone adapter uses
its default Reference options. The builder reads and validates these options
from `pack.yml` and passes them to the adapter. Neither qualifies a release. The upstream whitespace-delimited output cannot
represent filenames containing whitespace. An upstream dependency download
failure is reported as a generation failure, not replaced by a partial list.

Run the offline tests from the repository root (no downloads or Make required):

```sh
python3 -B -m unittest discover \
  -s tensorflow/lite/micro/tools/cmsis_pack/test -p test_file_lists.py -v
```

The tests use a fixture generator to check the subprocess contract and error
handling. They do not qualify the real runtime or its dependency versions.

### Draft backend inventory builder (implemented)

Use Linux/WSL with Python 3.12+, Git, GNU Make and the shell/download tools
required by the selected TFLM revision. Install `requirements.txt` into your
build environment (preferably a virtual environment):

```sh
python3 -m pip install -r /work/builder/tensorflow/lite/micro/tools/cmsis_pack/requirements.txt
python3 /work/builder/tensorflow/lite/micro/tools/cmsis_pack/scripts/build_pack.py \
  --tflm-root /work/selected-tflm \
  --out /work/litert-reference-output
```

Replace the example absolute paths with your paths. `--out` is a **new directory**;
an existing destination is rejected and preserved. `--config /path/to/pack.yml`
can override the default configuration beside the builder. Commands work from
any current directory. Runtime input must be the root of a Git checkout with a
commit. The builder never fetches or checks out a candidate release revision.

The default remains Reference (`cortex_m_generic` / `cortex-m7`). Select
`--variant cmsis_nn` for Cortex-M55 with CMSIS-NN, `--variant ethos_u` for
Cortex-M55/U55 with CMSIS-NN CPU support, or `--variant all` for all three.
Unknown options and alternate targets (including M85/U85) are rejected in this
milestone; they need their own configuration and validation work.
Unresolved release fields are allowed because the output is always marked draft.
This is subset validation, not the final release schema or release-policy gate.

Example using the WSL paths from this project, after activating your existing
Python environment:

```sh
TFLM_ROOT=/mnt/c/working/Git/github_com/MDK-Packs/WSP__tflite-micro/wk4626/tflite-micro
BUILDER="$TFLM_ROOT/tensorflow/lite/micro/tools/cmsis_pack"
python -B "$BUILDER/scripts/build_pack.py" \
  --tflm-root "$TFLM_ROOT" --variant all \
  --out "$HOME/litert-all-backends-output"
```

All selected generations must succeed before any output directory is published.
If a later generation changes a shared input, the build fails rather than
recording inconsistent source hashes. Outputs are:

- `<variant>-sources.txt`: sorted, unique checkout-relative paths for each
  selected variant (`reference`, `cmsis_nn`, `ethos_u`).
- `classification.json`: disjoint groups per variant for TFLM runtime/support,
  kernels, optimized integration, signal, licenses, bundled libraries and
  external dependencies; unknown paths go to `unclassified` with a CLI warning.
- `provenance.json`: actual runtime and builder commits, Git status including
  untracked files, builder/generator/configuration SHA-256 hashes, the complete
  configuration, make options, inventory hashes and each inventoried file's hash.
  Provenance schema version 2 uses `inventories` keyed by variant. For one
  selected variant, the `inventory` convenience field is retained.

A copied builder outside Git records a null commit with an explanation. Local
builder changes are captured by hashes; the commit alone does not identify them.
Runtime dependency caches may be Git-ignored, so file hashes also record their
observed contents. The classifier recognizes exact dependency directories under
`tensorflow/lite/micro/tools/make/downloads/`, and bundled dependencies under
`third_party/` for runtime revisions that keep headers there. `cmsis/Cortex_DFP/`
is classified separately as external device support. Its presence is not an
automatic universal Cortex_DFP pack requirement. Absent libraries are not added
to an inventory merely because the plan mentions them. All license paths returned
by the generator are retained; completeness still requires staging-time review.

Classification is ownership only: every report has `payload_approved: false`.
External and unclassified paths must not be blindly copied into the pack.
Unknown paths require review before payload staging; they do not prevent a draft
inventory from being emitted. TFLM support groups may contain tests or headers,
so these groups are not an approved runtime compilation list.

This record is not a complete dependency lockfile or proof of
reproducibility. It does not verify the candidate `26.08` baseline. Inputs should
remain unchanged while generation runs; concurrent editing is unsupported.

Generation may download dependencies into runtime caches even with `--no_copy`.
It does not stage payload files, fetch examples, generate an archive or publish.
Native Windows fixture tests do not qualify native Windows upstream builds.

Run all offline tests (fixture generator and temporary Git repositories):

```sh
python3 -B -m unittest discover \
  -s tensorflow/lite/micro/tools/cmsis_pack/test -p 'test_*.py' -v
```

## Component mapping and draft payload staging (implemented)

Staging is a separate command so it can reuse verified inventories without
running Make or downloading dependencies again. Activate the existing Python
build environment, set `TFLM_ROOT` and `BUILDER` as above, then run:

```sh
python -B "$BUILDER/scripts/stage_pack.py" \
  --tflm-root "$TFLM_ROOT" \
  --inventory-dir "$HOME/litert-all-backends-verified" \
  --out "$HOME/litert-staged-payload"
```

The inventory directory must contain schema-2 `provenance.json` and its named
source lists. The explicit runtime checkout must have the same commit and file
hashes. The output must be a new directory. This command does not modify inputs.

Outputs:

- `payload/`: selected files, copied once even when multiple variants use them.
- `component-map.json`: component identities, file ownership, allowed selections,
  compilation lists, independent configuration-file versions, exclusions and
  SHA-256 hashes. This is a draft mapping, not a PDSC or compile qualification.
- `input-provenance.json`: the original input provenance, bound by its hash.

The mapping preserves the plan's `Machine Learning:LiteRT` component structure:

| Component | Mapping rule |
| --- | --- |
| Kernel Reference / CMSIS-NN / Ethos-U | Each variant's inventoried runtime and kernel sources, excluding the shared components below. Exactly one Kernel variant is selectable. |
| Kernel Utils | `system_setup.cc`, Cortex-M `micro_time.cc` and `debug_log.cc`, plus legacy `tensorflow/lite/kernels/kernel_util.cc` when present. Configuration versions use the independent values in `pack.yml`. |
| Testing | Inventoried test helpers plus an explicit, hashed set of framework headers, convolution model and convolution/LSTM test data. Standalone test programs and host scripts are excluded. |
| Signal Library / Reference | Inventoried `signal/` files. |
| flatbuffers / gemmlowp fixed-point / kissfft / ruy | Bundled libraries actually present in the selected inventories; no empty components are created. |

Shared components must have the same file membership across selected variants.
The mapping validates each Kernel variant together with all available shared
components to detect duplicate compilation and conflicting ordinary/optimized
kernel implementations. Shared files can appear in multiple mutually exclusive
Kernel file lists, but are copied only once. This does not prove absence of all
possible linker symbol or object-basename collisions; consumer builds follow.

Runtime paths and `.cc` extensions are preserved. Downloaded bundled dependencies
move to `third_party/<dependency>/`; files already there retain their paths.
Licenses returned by the generator are retained, and each present bundled library
must have a license entry. If an older generator omits it, the stager checks the
existing downloaded library root for LICENSE, LICENSE.txt or COPYING. Added
licenses are separately recorded in supplemental_inputs with SHA-256 hashes;
original inventories are preserved. Complete notice and license review is still required.
External CMSIS, CMSIS-NN and Ethos-U driver paths are recorded as excluded, never
copied. Unknown paths, missing files, changed hashes, destination collisions
(including case-only collisions), differing shared components and test programs
cause staging to fail. Outputs are published only after successful checks.

For the verified current checkout `b8d75cbf6b8009762bc1e3ee4817bc33e5dcee47`,
flatbuffers and kissfft occur in the inventories; gemmlowp and ruy do not. TFLM's
`tensorflow/lite/micro/kernels/internal/common.h` explicitly describes dropping
the direct gemmlowp dependency, and `internal/fixedpoint.h` contains adapted code.
This is a development checkout observation, not a change to the agreed four-library
release design. The planned `90b983c2b3f1480642e040bbd2ddb37b998f3424` baseline
explicitly downloads and uses gemmlowp and ruy. Use that runtime checkout for
baseline payload validation; the builder can remain on its feature branch.

The older baseline also uses eight inventoried runtime files under
`tensorflow/compiler/mlir/lite/`; these exact paths are classified as runtime
support. This does not include compiler tools generally. Its generator omits
ruy's LICENSE, so the stager records and includes that license as a supplement.

Example for the separate baseline clone (after activating the builder environment):

```sh
python -B "$BUILDER/scripts/build_pack.py" \
  --tflm-root "$HOME/litert-runtime-26.08" --variant all \
  --out "$HOME/litert-26.08-inventories-checked"
python -B "$BUILDER/scripts/stage_pack.py" \
  --tflm-root "$HOME/litert-runtime-26.08" \
  --inventory-dir "$HOME/litert-26.08-inventories-checked" \
  --out "$HOME/litert-26.08-payload"
```

These directories must not already exist. The runtime checkout must be at the
full planned SHA above. Validating this runtime does not verify the entire
Ethos-U release manifest or qualify the final CMSIS-Pack.

The next command generates development component conditions and include paths,
and adapts copied configuration files. Platform runtime
behavior, consumer builds and example integration still require qualification. The staged tree is
not yet an installable or release-qualified CMSIS-Pack.

## Development PDSC (implemented)

Run after payload staging. `--draft` is mandatory and the generated component and
pack version is `<configured-version>-dev.0`, never the production version.
Supply an explicit date for reproducibility:

```sh
python -B "$BUILDER/scripts/generate_pdsc.py" \
  --stage-dir "$HOME/litert-26.08-payload" \
  --out "$HOME/litert-26.08-pdsc" --date 2026-11-17 --draft
```

The new output directory contains the copied payload, `TensorFlow.LiteRT.pdsc`
and `pdsc-report.json`. Source hashes, the input-provenance hash, component
ownership, duplicate destinations and all PDSC file/include references are
checked before publishing output. XML values are escaped structurally. Input
staging files and runtime source files are unchanged.

The PDSC declares mutually exclusive Kernel variants, Kernel Utils, Testing,
Signal Library and the bundled libraries present in the staging map. Kernel
conditions require Kernel Utils and the bundled libraries; CMSIS-NN and Ethos-U
also require `CMSIS:NN Lib`. Ethos-U additionally requires the driver component.
Testing and Signal Library depend on a Kernel selection. External package
requirements are declared separately from component-selection conditions.

Two reviewed adaptations are applied only to the copied payload:

- CMSIS-NN integration includes use `arm_nnfunctions.h` / `arm_nn_types.h`
  without the upstream `Include/` prefix. Header identity is preserved.
- The known `micro_time.cc` device-header block uses `RTE_Components.h` and
  `CMSIS_device_header`. An unexpected upstream block causes a failure.

The three configuration templates are placed in `config/` as `.cpp` files,
separate from normal header directories. Their independent configured versions
remain `1.0.0`. The report records original/output hashes and template relocation.
Other runtime `.cc` files keep their extensions. Timing counter behavior and CPU
macros still require target testing; successful metadata checks do not prove
firmware correctness.

Draft handling of unresolved inputs is explicit in `pdsc-report.json`:

- CMSIS remains a package requirement, but its unresolved version range is
  omitted in the development PDSC. This does not resolve consumer compatibility.
- Proposed CMSIS-NN and driver package ranges are retained and marked unverified
  in the report.
- If the driver selector is null, draft mode uses the legacy candidate
  `Machine Learning:NPU Support:Ethos-U Driver`. It is not a verified driver-2.x
  selector and does not update `pack.yml`.
- Release baseline, target execution and example assembly
  remain blockers. No release archive is produced.

Validate the PDSC with a specific packchk and its matching schema, supplying
installed dependency PDSCs through repeated `-i` arguments. For example:

```sh
packchk /path/to/TensorFlow.LiteRT.pdsc -s /path/to/PACK.xsd \
  -i /path/to/ARM.CMSIS.pdsc \
  -i /path/to/ARM.CMSIS-NN.pdsc \
  -i /path/to/ARM.ethos-u-core-driver.pdsc \
  -b /path/to/packchk.log
```

Do not suppress unresolved-dependency warnings or change a range merely to match
an installed pack. A clean schema/static check is not release qualification.

## Standalone consumer compile/link checks

`test/smoke/compile_smoke.py` tests a generated development PDSC payload with
Arm GNU Toolchain on Linux/WSL (Python 3.11 or newer). It compiles all selected
sources, including Testing and Signal Library, for Reference/Cortex-M7,
CMSIS-NN/Cortex-M55 and Ethos-U/Cortex-M55+U55. It links a small consumer with a
supplied int8 hello-world model, Fully Connected and Reshape registrations;
the Ethos-U profile additionally registers the accelerator operator.

```sh
python3 tensorflow/lite/micro/tools/cmsis_pack/test/smoke/compile_smoke.py \
  --pack-dir /path/to/generated-pdsc-output \
  --toolchain-bin /path/to/gcc/bin \
  --cmsis /mnt/c/PACK/ARM/CMSIS/6.2.0 \
  --cortex-dfp /mnt/c/PACK/ARM/Cortex_DFP/1.1.0 \
  --cmsis-nn /mnt/c/PACK/ARM/CMSIS-NN/8.0.0 \
  --driver /mnt/c/PACK/ARM/ethos-u-core-driver/1.26.5/ethos_u_core_driver \
  --model tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  --out "$HOME/litert-consumer-smoke" --jobs 4
```

Choose a new output directory for each run. The harness checks the generated
payload hashes and preserves compiler commands, logs, copied inputs, ELF files
and `smoke-report.json`, including external input hashes.

This is a direct compiler test, not CMSIS dependency resolution: explicit paths
can exercise a driver outside the proposed PDSC version range. Successful linking
does not approve that range or establish compatibility with an untested driver.
The linker removes unreachable code, so this is not exhaustive operator link
coverage. The default linker script and nosys stubs produce a link-test ELF,
not a board firmware image. Neither model inference nor accelerator execution
is performed; the supplied model is not Vela compiled by this test.
Linker warnings about unimplemented system calls are expected with `nosys`;
the real example must supply its platform startup, memory layout and I/O support.

### Initial validation (2026-11-17)

For runtime revision `90b983c2b3f1480642e040bbd2ddb37b998f3424`, all three
profiles compiled and linked with Arm GNU Toolchain 14.3.Rel1 (GCC 14.3.1).
Inputs were CMSIS 6.2.0, Cortex_DFP 1.1.0, CMSIS-NN 8.0.0 and, for Ethos-U,
driver pack 1.26.5. The unit suite passed 49 tests.

| Profile | Translation units compiled | Compile/link result |
| --- | ---: | --- |
| Reference / Cortex-M7 | 206 | Passed |
| CMSIS-NN / Cortex-M55 | 319 | Passed |
| Ethos-U / Cortex-M55 + U55 | 322 | Passed |

The driver pack used here reports driver API 1.0.0 in its header; the runtime's
downloaded driver reports API 0.16.0. Neither establishes the proposed 2.x
baseline. This experimental check deliberately used an explicit driver path
outside the draft PDSC range. Keep that range unverified until the intended
release dependency is selected and qualified. No hardware/FVP execution was
performed. Testing-component validation is described below.

## Testing component (implemented)

The component contains 15 reusable support files at the pinned runtime revision:
the existing test helpers and kernel-test headers, both `micro_test.h` and
`micro_test_v2.h`, `test_conv_model.h/.cc`, and convolution/LSTM test-data
header/source pairs. Runtime kernel-runner utilities remain supplied by the
selected Kernel component. Testing requires a Kernel selection, which in turn
requires Kernel Utils and the bundled libraries.

The eight files absent from upstream's inference inventory are explicitly listed
in `stage_pack.py`, copied without rewriting and recorded in `supplemental_inputs`
with SHA-256 hashes and variant membership. Missing support files fail staging;
they are not silently omitted. Older incomplete staged trees must be regenerated
before PDSC generation. This component supplies reusable test support, not the
entire upstream test suite, model-generation scripts or test runners. The checks
below use the normal uncompressed runtime configuration.

Run the dedicated consumer check under Linux/WSL with native `g++` and the Arm
GNU Toolchain available:

```sh
python3 tensorflow/lite/micro/tools/cmsis_pack/test/smoke/testing_smoke.py \
  --pack-dir /path/to/generated-pdsc-output \
  --toolchain-bin /path/to/arm-gnu-toolchain/bin \
  --out "$HOME/litert-testing-consumers"
```

Choose a new output directory. The harness verifies payload hashes and uses only
packaged headers/sources. It compiles every Testing source and exercises tensor
helpers and the convolution model with both framework APIs. The v2 consumer also
includes the convolution, LSTM and decode test-helper headers, checking their
transitive includes. Linux execution checks successful assertions and deliberately
failing assertions (expected exit code 1). The same successful consumers are
cross-compiled and linked for Cortex-M7 and Cortex-M55, without target execution.
Commands, logs and hashes are retained in `testing-report.json` and adjacent files.

Validation on 2026-11-17: 53 builder tests passed; both framework consumers passed
the Linux success/failure checks and Cortex-M7/M55 compile/link checks. The updated
504-file payload passed packchk with zero errors and the existing unresolved
driver-2.x warning. Hardware behavior, other compile-time feature configurations
and full kernel test-suite execution remain separate release qualification work.

## Outputs and validation

| Stage | Intended output | Completion checks |
| --- | --- | --- |
| First milestone | Selected backend inventories, `classification.json` and `provenance.json` (implemented). | Every file exists; duplicates are removed; input revision and make options are recorded; invocation works from another directory. |
| Complete core builder | Staged runtime payload, `TensorFlow.LiteRT.pdsc`, `TensorFlow.LiteRT.<version>.pack` and provenance documentation. | Script tests pass; referenced files exist; licenses are retained; component conditions and dependency ranges are checked; pinned packchk reports zero errors. |
| Publisher integration | Final pack including the staged example, plus the separately built ARM driver pack and release assets. | Revalidate the final assembled artifacts; compile the consumer matrix and run the packaged example/FVP tests before publication. |

Source inventories, ownership classification, draft provenance and a checked
component-mapped payload and development PDSC can now be generated. A successful core pack build
alone is not evidence that example integration, driver compatibility or release
qualification has passed.

## Next implementation steps

1. Broaden operator link coverage and verify platform
   configuration behavior with target/FVP tests. Initial standalone consumer
   compile/link checks are implemented and passed for the profiles above.
2. Resolve and verify CMSIS and driver compatibility, the actual driver-2.x
   component selectors, and all release pins. Require resolved dependencies for
   release-mode metadata and archive generation.
3. Implement final pack archive creation and release-policy validation.
4. Implement publisher integration for the packaged sample and release history;
   validate the final pack including the example.
