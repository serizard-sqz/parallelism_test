# Wan 2.2 Sequence Parallelism 벤치마크

Wan 2.2 T2V/I2V 모델에서 sequence parallelism, FP8 quantization, `torch.compile`, 4-step Lightning LoRA를 테스트하는 스크립트입니다.

## 파일

- `wan_t2v_sp_test.py`: Text-to-Video 벤치마크
- `wan_i2v_sp_test.py`: Image-to-Video 벤치마크
- `samples/i2v/sample_i2v.avif`: I2V 기본 입력 이미지
- `pyproject.toml`: 실행 의존성 선언
- `uv.lock`: 재현 가능한 잠금 의존성

## 설치

```bash
uv sync --locked
```

의존성은 `pyproject.toml`에 선언하고 `uv.lock`으로 고정합니다. 의존성을 변경한 뒤에는 `uv lock`을 실행해 잠금 파일을 갱신하고, 설치/실행 환경에서는 `uv sync --locked`로 잠금 파일과 일치하는 환경을 구성합니다.

의존성 import는 파일 상단에서 바로 수행됩니다. 패키지가 없으면 파이썬 import 에러가 그대로 출력됩니다.

## T2V 실행

```bash
uv run --locked python wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 480p \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --output-dir outputs/t2v
```

## I2V 실행

```bash
uv run --locked python wan_i2v_sp_test.py \
  --image samples/i2v/sample_i2v.avif \
  --prompt "Animate the image with gentle camera motion" \
  --resolution 480p \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --output-dir outputs/i2v
```

## Sequence Parallel 실행

`--sp-backend` 사용 시 `torchrun`으로 실행합니다. USP는 `--ulysses-degree`와 `--ring-degree`를 명시합니다.

```bash
uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A futuristic city flythrough" \
  --resolution 720p \
  --sp-backend usp \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --warmup-runs 1 \
  --benchmark-runs 5
```

## 로그

- 기본 로그는 `outputs/.../*.jsonl`에 저장됩니다.
- `--log-file result.csv`처럼 `.csv` 경로를 주면 CSV로 저장됩니다.
- benchmark run만 latency summary에 포함되고 warmup run은 제외됩니다.

## 구현 원칙

- 각 스크립트는 `parse_args()`와 `main()`만 정의합니다.
- helper function과 대체 실행 경로를 두지 않습니다.
- import 실패나 런타임 실패는 별도 포장 없이 그대로 드러납니다.
