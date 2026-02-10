"""
GPU Memory Cleanup Utility

This utility script helps free up GPU memory when experiencing CUDA memory issues
or GPU overload. It performs Python garbage collection and clears CUDA cache.

Usage:
    Run directly: python gpu_cleanup_utility.py
    Or import: from gpu_cleanup_utility import clear_gpu_memory

Author: OpenPose Project
Date: 2026
"""

import gc
import torch
import sys

def clear_gpu_memory():
    """
    Force clear GPU memory by performing garbage collection and clearing CUDA cache.
    
    This function:
    1. Runs Python's garbage collector to free Python objects
    2. Clears CUDA cache if PyTorch and CUDA are available
    3. Synchronizes CUDA operations
    4. Reports memory status for each available GPU
    
    Returns:
        None
        
    Prints:
        GPU memory statistics including allocated and reserved memory for each GPU device
    """
    print("🧹 Clearing GPU memory...")
    
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache if PyTorch is available
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✅ Cleared CUDA cache")
            
            # Show GPU memory info
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        else:
            print("⚠️ CUDA not available")
    except Exception as e:
        print(f"⚠️ Could not clear CUDA cache: {e}")
    
    print("✅ Cleanup complete")

if __name__ == "__main__":
    clear_gpu_memory()
    
    # Keep window open so you can see the results
    input("\nPress Enter to exit...")
