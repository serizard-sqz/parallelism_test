#!/usr/bin/env python3
"""Wan 2.2 I2V sequence-parallel Diffusers benchmark script."""

import argparse as argparse_lib
import csv
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import diffusers
import numpy as np
import torch
import torchao
import transformers
from diffusers import AutoencoderKLWan, ContextParallelConfig, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_


def frames_for_export(frames):
    """Wan returns np.stack'd video (B, F, H, W, C); imageio expects a list of (H, W, C) frames."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4:
            return list(frames)
    return frames


def parse_args():
    parser = argparse_lib.ArgumentParser(description=__doc__, formatter_class=argparse_lib.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers", help="Target Wan 2.2 model id.")
    parser.add_argument("--lora-repo", default="lightx2v/Wan2.2-Lightning", help="Repository containing Wan 2.2 Lightning LoRA weights.")
    parser.add_argument("--lora-subfolder", default="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1", help="Subfolder inside --lora-repo containing the LoRA file.")
    parser.add_argument("--lora-weight-name", default="high_noise_model.safetensors", help="4-step Lightning high-noise LoRA safetensors filename.")
    parser.add_argument("--lora-transformer-2-weight-name", default="low_noise_model.safetensors", help="4-step Lightning low-noise LoRA safetensors filename.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength fused into transformer base weights (see fuse_lora).")
    parser.add_argument("--resolution", choices=("480p", "720p"), default="480p", help="Output resolution class.")
    parser.add_argument("--height", type=int, default=None, help="Override output height.")
    parser.add_argument("--width", type=int, default=None, help="Override output width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of video frames to generate.")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="Inference steps; defaults to Lightning LoRA's 4-step path.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed used on every rank.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmup generations before latency logging.")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Timed generations after warmup.")
    parser.add_argument("--sp-backend", choices=("none", "ring", "ulysses", "usp"), default="none", help="Context/sequence parallel attention backend.")
    parser.add_argument("--ulysses-degree", type=int, default=None, help="Ulysses degree for Ulysses/USP sequence parallelism.")
    parser.add_argument("--ring-degree", type=int, default=None, help="Ring degree for Ring/USP sequence parallelism.")
    parser.add_argument("--attention-backend", default=None, help="Optional Diffusers transformer attention backend, e.g. _native_cudnn.")
    parser.add_argument("--compile", dest="compile", action=argparse_lib.BooleanOptionalAction, default=True, help="Enable block-level torch.compile(mode=default, dynamic=True, fullgraph=True) after FP8 quantization.")
    parser.add_argument("--fp8", dest="fp8", action=argparse_lib.BooleanOptionalAction, default=True, help="Enable TorchAO dynamic activation/dynamic weight FP8 quantization before torch.compile.")
    parser.add_argument("--output-dir", default="outputs/i2v", help="Directory for generated videos and default latency log.")
    parser.add_argument("--log-file", default=None, help="Optional JSONL or CSV latency log path. Defaults to JSONL under --output-dir.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", help="Pipeline inference dtype.")
    parser.add_argument("--device", default=None, help="CUDA device string. Defaults to cuda:<LOCAL_RANK>.")
    parser.add_argument("--fps", type=int, default=16, help="FPS used when exporting the generated video.")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed configuration without loading models.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code=True to from_pretrained.")
    parser.add_argument("--image", default="samples/i2v/sample_i2v.avif", help="Input image path for I2V generation.")
    parser.add_argument("--prompt", default="Animate the input image with gentle cinematic camera motion, high quality.", help="Text prompt for I2V generation.")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt.")
    return parser.parse_args()


def context_parallel_degrees(args):
    """Ring/Ulysses degrees for ContextParallelConfig, log paths, and export filenames."""
    ring_degree = 1
    ulysses_degree = 1
    if args.sp_backend != "none":
        if args.sp_backend == "ring":
            ring_degree = args.ring_degree
            ulysses_degree = 1
        elif args.sp_backend == "ulysses":
            ring_degree = 1
            ulysses_degree = args.ulysses_degree
        else:
            ulysses_degree = args.ulysses_degree
            ring_degree = args.ring_degree
    return ring_degree, ulysses_degree


def main():
    args = parse_args()
    script_kind = "i2v"
    height, width = {"480p": (480, 832), "720p": (720, 1280)}[args.resolution]
    if args.height is not None:
        height = args.height
    if args.width is not None:
        width = args.width

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ring_degree, ulysses_degree = context_parallel_degrees(args)
    log_file = (
        Path(args.log_file)
        if args.log_file
        else output_dir / f"{script_kind}_latency_{args.sp_backend}_ring{ring_degree}_ulysses{ulysses_degree}_{args.resolution}.jsonl"
    )
    config = vars(args).copy()
    config.update(
        {
            "script": script_kind,
            "height": height,
            "width": width,
            "log_file": str(log_file),
            "ring_degree": ring_degree,
            "ulysses_degree": ulysses_degree,
        }
    )
    print("CONFIG " + json.dumps(config, sort_keys=True, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete; model loading and generation skipped.")
        return

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device or f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist_initialized_here = False
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            dist_initialized_here = True

    version_info = {
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }
    print("VERSIONS " + json.dumps(version_info, sort_keys=True, ensure_ascii=False))

    load_kwargs = {"torch_dtype": torch_dtype}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_id, vae=vae, **load_kwargs)
    pipe.to(device)
    pipe.load_lora_weights(args.lora_repo, subfolder=args.lora_subfolder, weight_name=args.lora_weight_name)
    pipe.load_lora_weights(args.lora_repo, subfolder=args.lora_subfolder, weight_name=args.lora_transformer_2_weight_name, load_into_transformer_2=True)
    pipe.fuse_lora(components=["transformer", "transformer_2"], lora_scale=args.lora_scale)
    pipe.unload_lora_weights()
    print("LORA " + json.dumps({"repo": args.lora_repo, "subfolder": args.lora_subfolder, "weight_name": args.lora_weight_name, "transformer_2_weight_name": args.lora_transformer_2_weight_name, "scale": args.lora_scale, "fused": True, "components": ["transformer", "transformer_2"]}, sort_keys=True, ensure_ascii=False))

    modules = [("transformer", pipe.transformer), ("transformer_2", pipe.transformer_2)]
    if args.attention_backend:
        for module_name, module in modules:
            module.set_attention_backend(args.attention_backend)

    if args.sp_backend != "none":
        cp_config = ContextParallelConfig(ring_degree=ring_degree, ulysses_degree=ulysses_degree)
        for module_name, module in modules:
            module.enable_parallelism(config=cp_config)
        print("SEQUENCE_PARALLELISM " + json.dumps({"backend": args.sp_backend, "ring_degree": ring_degree, "ulysses_degree": ulysses_degree, "world_size": world_size}, sort_keys=True, ensure_ascii=False))

    if args.fp8:
        fp8_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        for module_name, module in modules:
            quantized_blocks = 0
            for block_index, block in enumerate(module.blocks):
                if block_index >= 3 and block_index < len(module.blocks) - 3:
                    quantize_(block, fp8_config, device="cuda")
                    quantized_blocks += 1
            torch.cuda.empty_cache()
            print("FP8 " + json.dumps({"module": module_name, "enabled": True, "config": "Float8DynamicActivationFloat8WeightConfig(PerTensor)", "device": "cuda", "skip_edge_blocks": 3, "quantized_blocks": quantized_blocks}, sort_keys=True))
    else:
        print("FP8 " + json.dumps({"enabled": False}, sort_keys=True))

    if args.compile:
        for module_name, module in modules:
            compiled_blocks = 0
            for block_index, block in enumerate(module.blocks):
                module.blocks[block_index] = torch.compile(block, mode="default", dynamic=True, fullgraph=False)
                compiled_blocks += 1
            print("COMPILE " + json.dumps({"module": module_name, "enabled": True, "target": "blocks", "compiled_blocks": compiled_blocks, "mode": "default", "dynamic": True, "fullgraph": True}, sort_keys=True))
    else:
        print("COMPILE " + json.dumps({"enabled": False}, sort_keys=True))

    run_records = []
    log_file.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fieldnames = ["event", "script", "model_id", "lora_repo", "lora_subfolder", "lora_weight_name", "sp_backend", "ring_degree", "ulysses_degree", "world_size", "rank", "resolution", "height", "width", "num_frames", "dtype", "compile", "fp8", "run_index", "latency_seconds", "output_path", "seed", "warmup_runs", "measured_runs", "mean_latency_seconds", "median_latency_seconds", "min_latency_seconds", "max_latency_seconds", "log_file", "cuda_peak_memory_allocated_bytes", "cuda_peak_memory_reserved_bytes"]
    if rank == 0:
        if log_file.suffix.lower() == ".csv":
            with log_file.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                if fh.tell() == 0:
                    writer.writeheader()
        else:
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "start", "config": config, "versions": version_info}, ensure_ascii=False) + "\n")

    for run_index in range(args.warmup_runs + args.benchmark_runs):
        phase = "warmup" if run_index < args.warmup_runs else "benchmark"
        torch.cuda.synchronize(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + run_index)
        pipe_kwargs = {"prompt": args.prompt, "negative_prompt": args.negative_prompt, "image": load_image(str(Path(args.image))), "height": height, "width": width, "num_frames": args.num_frames, "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale, "generator": generator}
        start = time.perf_counter()
        result = pipe(**pipe_kwargs)
        torch.cuda.synchronize(device)
        latency = time.perf_counter() - start
        frames = result.frames
        measured_index = run_index - args.warmup_runs
        if phase == "benchmark" and rank == 0:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = output_dir / (
                f"wan_{script_kind}_{args.resolution}_{args.sp_backend}_"
                f"ring{ring_degree}_ulysses{ulysses_degree}_seed{args.seed}_run{measured_index}_{stamp}.mp4"
            )
            export_to_video(frames_for_export(frames), str(output_path), fps=args.fps)
            record = {"event": "latency", "script": script_kind, "model_id": args.model_id, "lora_repo": args.lora_repo, "lora_subfolder": args.lora_subfolder, "lora_weight_name": args.lora_weight_name, "sp_backend": args.sp_backend, "ring_degree": ring_degree, "ulysses_degree": ulysses_degree, "world_size": world_size, "rank": rank, "resolution": args.resolution, "height": height, "width": width, "num_frames": args.num_frames, "dtype": args.dtype, "compile": args.compile, "fp8": args.fp8, "run_index": measured_index, "latency_seconds": latency, "output_path": str(output_path), "seed": args.seed + run_index, "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            run_records.append(record)
            print("LATENCY " + json.dumps(record, sort_keys=True, ensure_ascii=False))
            if log_file.suffix.lower() == ".csv":
                with log_file.open("a", encoding="utf-8", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore").writerow(record)
            else:
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print("RUN " + json.dumps({"phase": phase, "run_index": run_index, "rank": rank, "latency_seconds": latency}, sort_keys=True))

    if rank == 0:
        latencies = [record["latency_seconds"] for record in run_records]
        summary = {"event": "summary", "script": script_kind, "measured_runs": len(latencies), "warmup_runs": args.warmup_runs, "mean_latency_seconds": statistics.fmean(latencies), "median_latency_seconds": statistics.median(latencies), "min_latency_seconds": min(latencies), "max_latency_seconds": max(latencies), "log_file": str(log_file)}
        print("SUMMARY " + json.dumps(summary, sort_keys=True, ensure_ascii=False))
        if log_file.suffix.lower() == ".csv":
            with log_file.open("a", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore").writerow(summary)
        else:
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if dist_initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
