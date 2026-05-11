# PRD: Sequence Parallelism Test Scripts for Wan 2.2

## 1. Purpose

Create two standalone benchmark/test scripts that validate sequence parallelism for Wan 2.2 text-to-video (T2V) and image-to-video (I2V) generation using the Diffusers implementation. The scripts must make it easy to run comparable latency measurements after warmup while exercising compile, FP8 quantization, Lightning LoRA, and multiple sequence-parallel attention backends.

## 2. Existing requirements preserved

- target model: Wan-AI/Wan2.2-T2V-A14B, Wan-AI/Wan2.2-I2V-A14B
- target model에는 Kijai/WanVideo_comfy에 있는 4-step lightning lora 적용. 또한 diffusers 구현체를 사용해야함.
- 작업 대상 파일: wan_t2v_sp_test.py, wan_i2v_sp_test.py
- 각 스크립트 파일은 argparse func, main func으로만 구성
- warmup 이후에 latency 로깅 기능 제공 필요
- torch.compile, torchao fp8 quantization 적용
- sequence parallelism은 ring attention, ulysses attention, usp를 지원하도록 구현
- output은 480p or 720p
- torch 2.11, cu130, torchao 0.17.0, transformers 5.7.0, diffusers git 최신 버전 사용

## 3. Goals

1. Provide one executable T2V script and one executable I2V script for Wan 2.2 A14B models.
2. Use Diffusers pipelines only; do not implement against ComfyUI runtime APIs.
3. Apply the 4-step Lightning LoRA weights from `Kijai/WanVideo_comfy` so the default inference path is a 4-step generation benchmark.
4. Support sequence-parallel attention modes for:
   - Ring Attention
   - Ulysses Attention
   - USP
5. Apply `torch.compile` and TorchAO FP8 quantization in the benchmark path, with CLI flags that make the selected behavior explicit.
6. Log post-warmup latency consistently enough to compare backends, resolutions, and model modes.
7. Generate outputs at either 480p or 720p.

## 4. Non-goals

- Training, fine-tuning, or LoRA creation.
- Implementing a new video model architecture.
- Supporting non-Wan models beyond the two target model IDs.
- Supporting ComfyUI workflows as the runtime implementation.
- Building a production serving API; this is a local benchmark/test utility.

## 5. Target files and structure constraints

### 5.1 Files

- `wan_t2v_sp_test.py`: text-to-video sequence-parallel test script.
- `wan_i2v_sp_test.py`: image-to-video sequence-parallel test script.

### 5.2 Script shape

Each script must contain only these top-level functions:

1. `parse_args()`
   - Builds and returns the CLI parser or parsed arguments.
   - Owns defaults, choices, validation-friendly argument definitions, and help text.
2. `main()`
   - Performs setup, model loading, LoRA loading, quantization, compile, generation, timing, output writing, and summary logging.

If shared behavior is needed, prefer inline logic inside `main()` or imports from established libraries. Do not add extra project-local helper functions unless this PRD is intentionally revised.

## 6. Functional requirements

### 6.1 Common CLI requirements

Both scripts should expose a consistent CLI surface:

- `--model-id`: defaults to the matching target model:
  - T2V: `Wan-AI/Wan2.2-T2V-A14B`
  - I2V: `Wan-AI/Wan2.2-I2V-A14B`
- `--lora-repo` or equivalent: defaults to `Kijai/WanVideo_comfy`.
- `--lora-weight-name` / `--lora-subfolder` or equivalent: identifies the 4-step Lightning LoRA weight inside the LoRA repository.
- `--resolution`: choice of `480p` or `720p`.
- `--height` and `--width`: optional explicit dimensions, validated to match the selected resolution class when provided.
- `--num-frames`: number of output frames.
- `--num-inference-steps`: defaults to `4` to match the Lightning LoRA requirement.
- `--guidance-scale`: configurable, with a documented default suitable for the selected Wan pipeline.
- `--seed`: optional deterministic seed.
- `--warmup-runs`: number of untimed warmup generations.
- `--benchmark-runs`: number of timed generations after warmup.
- `--sp-backend`: choices include `ring`, `ulysses`, `usp`; an optional `none` baseline may be included if useful for comparison.
- `--compile` / `--no-compile`: explicit control over `torch.compile`.
- `--fp8` / `--no-fp8`: explicit control over TorchAO FP8 quantization.
- `--output-dir`: directory for generated videos and logs.
- `--log-file`: optional JSONL or CSV latency log path.
- `--dtype`: precision selection, defaulting to a CUDA-friendly inference dtype such as `bfloat16` when supported.
- `--device`: defaults to CUDA when available.

### 6.2 T2V-specific CLI requirements

`wan_t2v_sp_test.py` must additionally support:

- `--prompt`: required or defaulted text prompt.
- `--negative-prompt`: optional negative prompt.

### 6.3 I2V-specific CLI requirements

`wan_i2v_sp_test.py` must additionally support:

- `--image`: input image path, defaulting to `samples/i2v/sample_i2v.avif` when available.
- `--prompt`: required or defaulted image-conditioning prompt.
- `--negative-prompt`: optional negative prompt.

### 6.4 Diffusers pipeline behavior

- Load the appropriate Wan Diffusers pipeline for the selected model ID.
- Load and activate the 4-step Lightning LoRA from `Kijai/WanVideo_comfy`.
- Keep the LoRA scale configurable if the Diffusers API supports it.
- Use Diffusers-native generation calls for both T2V and I2V.
- Save generated video artifacts under `--output-dir` with filenames that include script type, resolution, backend, seed, and timestamp or run index.

### 6.5 Sequence parallelism behavior

- The implementation must run correctly under distributed launch, e.g. `torchrun`, for sequence-parallel modes.
- The selected backend must be visible in logs and output filenames.
- Required backend behavior:
  - `ring`: use ring-attention style sequence communication.
  - `ulysses`: use Ulysses-style sequence partition/all-to-all behavior.
  - `usp`: use Unified Sequence Parallel style behavior.
- Validate that required distributed environment variables are present for multi-rank execution.
- Fail fast with a clear error when a requested backend cannot be initialized.
- If a single-rank run is allowed, document whether it is a functional smoke test or a true sequence-parallel benchmark.

### 6.6 Compile and quantization behavior

- Apply TorchAO FP8 quantization before timed benchmark runs.
- Apply `torch.compile` before timed benchmark runs.
- Ensure warmup runs include any compile graph capture / first-run overhead so measured runs reflect steady-state latency.
- Log whether compile and FP8 quantization were enabled, skipped, or failed.
- Requested features should fail fast by default rather than silently falling back; any fallback must be explicit in logs.

### 6.7 Latency logging

After warmup, log each benchmark run with at least:

- script name (`t2v` or `i2v`)
- model ID
- LoRA repo / weight identifier
- sequence-parallel backend
- world size and rank information when distributed
- resolution and frame count
- dtype
- compile enabled/disabled
- FP8 enabled/disabled
- run index
- latency seconds
- output path
- seed

At the end, print summary statistics:

- mean latency
- median latency
- min latency
- max latency
- number of warmup runs
- number of measured runs

Optional but preferred:

- peak CUDA memory allocated/reserved
- tokens/frames/latent-shape metadata if easily available
- per-rank timing details for distributed execution

## 7. Resolution requirements

The scripts must support these output classes:

- `480p`: a 480p video shape appropriate for Wan 2.2 and Diffusers constraints.
- `720p`: a 720p video shape appropriate for Wan 2.2 and Diffusers constraints.

Dimension validation should account for model requirements such as divisibility by latent downsampling factors. If exact width/height presets differ between T2V and I2V pipeline recommendations, document the selected defaults in CLI help and logs.

## 8. Environment and dependency targets

Target environment:

- PyTorch: `torch 2.11`
- CUDA: `cu130`
- TorchAO: `torchao 0.17.0`
- Transformers: `transformers 5.7.0`
- Diffusers: latest git version

The scripts should include startup logging that prints the detected versions of:

- `torch`
- `torchao`
- `transformers`
- `diffusers`
- CUDA runtime / device name when available

## 9. Example usage

### 9.1 T2V smoke run

```bash
python wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 480p \
  --sp-backend ring \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --output-dir outputs/t2v-ring-480p
```

### 9.2 I2V smoke run

```bash
python wan_i2v_sp_test.py \
  --image samples/i2v/sample_i2v.avif \
  --prompt "Animate the scene with gentle camera motion" \
  --resolution 480p \
  --sp-backend ulysses \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --output-dir outputs/i2v-ulysses-480p
```

### 9.3 Distributed sequence-parallel run

```bash
torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A futuristic city flythrough" \
  --resolution 720p \
  --sp-backend usp \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --output-dir outputs/t2v-usp-720p
```

## 10. Validation and acceptance criteria

Implementation is complete when:

1. `wan_t2v_sp_test.py` and `wan_i2v_sp_test.py` both provide `argparse()` and `main()` as their only top-level functions.
2. Both scripts can load the target Wan 2.2 model through Diffusers.
3. Both scripts can apply the 4-step Lightning LoRA from `Kijai/WanVideo_comfy`.
4. Both scripts expose and log Ring, Ulysses, and USP backend selection.
5. Warmup runs execute before measured runs and are excluded from latency summaries.
6. Timed runs log per-run latency and final summary statistics.
7. `torch.compile` and TorchAO FP8 quantization are applied or fail with clear actionable errors when requested.
8. 480p and 720p output modes are supported and validated.
9. T2V writes a generated video artifact for a text prompt.
10. I2V writes a generated video artifact from an input image and prompt.
11. Startup logs include relevant library and CUDA/device versions.
12. Distributed runs either complete for each supported backend or fail fast with a backend-specific initialization message.

## 11. Testing plan

Minimum validation before declaring the scripts ready:

1. CLI help check:
   - `python wan_t2v_sp_test.py --help`
   - `python wan_i2v_sp_test.py --help`
2. Argument validation check for unsupported resolution and backend values.
3. Single-rank smoke test with 480p, one warmup run, and one measured run.
4. Single-rank smoke test for both T2V and I2V.
5. Distributed launch smoke test for each backend using the smallest feasible frame count and 480p output.
6. One 720p run for at least one backend after 480p smoke tests pass.
7. Verify generated logs contain all required latency fields and summary statistics.
8. Verify output videos are written to the requested output directory.

## 12. Risks and mitigations

- **Large model memory pressure**: keep resolution, frame count, dtype, compile, and FP8 settings explicit; log CUDA memory usage when available.
- **Diffusers API drift**: use latest git Diffusers but keep startup version logging and clear errors around pipeline/LoRA loading.
- **LoRA weight naming ambiguity**: expose LoRA repo, subfolder, and weight-name arguments instead of hardcoding only one file path.
- **Distributed backend incompatibility**: validate rank/world-size setup before model execution and fail fast with backend-specific messages.
- **Compile warmup distortion**: keep compile and graph-capture overhead inside warmup, not measured iterations.
