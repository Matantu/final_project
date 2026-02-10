"""
GPU Memory Cleanup Utility
Run this if you experience GPU overload or memory issues
"""

import gc
import torch
import sys

def clear_gpu_memory():
    """Force clear GPU memory"""
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
