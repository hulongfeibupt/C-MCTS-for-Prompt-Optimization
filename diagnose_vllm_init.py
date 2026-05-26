#!/usr/bin/env python3
"""
Diagnostic script for vLLM engine initialization.
Tests different gpu_memory_utilization and max_model_len combinations.
"""

import os
import sys
import argparse
import subprocess

def get_gpu_memory(gpu_id):
    """Get GPU memory info using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            free, total = result.stdout.strip().split(',')
            return int(free), int(total)
        else:
            print(f"Failed to get GPU memory for GPU {gpu_id}")
            return None, None
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None, None

def test_vllm_init(ckpt_dir, gpu_id, model_type, gpu_memory_utilization, max_model_len):
    """Test vLLM engine initialization with given parameters."""
    try:
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Lazy imports to avoid main process CUDA init
        try:
            from swift.llm import VllmEngine
        except ImportError as e:
            print(f"Failed to import VllmEngine: {e}")
            return False
        
        print(f"Testing VllmEngine init: gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}")
        
        # Try to initialize
        llm_engine = VllmEngine(
            ckpt_dir,
            model_type=model_type,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
        )
        
        # If successful, clean up
        del llm_engine
        import torch
        torch.cuda.empty_cache()
        
        print("SUCCESS")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Diagnose vLLM initialization")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint directory")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to test")
    parser.add_argument("--model_type", default="qwen3", help="Model type")
    parser.add_argument("--check-memory-only", action="store_true", help="Only check GPU memory, don't test vLLM")
    args = parser.parse_args()
    
    print(f"=== vLLM Initialization Diagnostic for GPU {args.gpu} ===")
    
    # Check GPU memory
    free_mem, total_mem = get_gpu_memory(args.gpu)
    if free_mem is not None:
        print(f"GPU {args.gpu} Memory: {free_mem}MB free / {total_mem}MB total")
    else:
        print(f"Could not get memory info for GPU {args.gpu}")
        return
    
    if args.check_memory_only:
        return
    
    # Test different combinations (from low to high memory utilization)
    mem_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_len_candidates = [256, 512, 1024, 2048, 4096, 8192]
    
    successful_configs = []
    
    for mem in mem_candidates:
        for max_len in max_len_candidates:
            if test_vllm_init(args.ckpt, args.gpu, args.model_type, mem, max_len):
                successful_configs.append((mem, max_len))
                print(f"✓ Success with mem={mem}, max_len={max_len}")
            else:
                print(f"✗ Failed with mem={mem}, max_len={max_len}")
    
    print("=== Summary ===")
    if successful_configs:
        print("Successful configurations:")
        for mem, max_len in successful_configs:
            print(f"  gpu_memory_utilization={mem}, max_model_len={max_len}")
    else:
        print("No successful configurations found. Check GPU memory and model requirements.")

if __name__ == "__main__":
    main()