export CUDA_VISIBLE_DEVICES=1,2,3,4

uv run --locked torchrun --nproc_per_node=4 wan_t2v_sp_test.py \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --resolution 720p \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --output-dir outputs/t2v \
  --sp-backend ulysses \
  --ulysses-degree 4
