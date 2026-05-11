export CUDA_VISIBLE_DEVICES=0
uv run --locked python wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v


export CUDA_VISIBLE_DEVICES=0,1
uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v \
  --sp-backend ring \
  --ring-degree 2

uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v \
  --sp-backend ulysses \
  --ulysses-degree 2


export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v \
  --sp-backend ring \
  --ring-degree 4

uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v \
  --sp-backend ulysses \
  --ulysses-degree 4

uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 2 \
  --output-dir outputs/t2v \
  --sp-backend usp \
  --ring-degree 2 \
  --ulysses-degree 2
