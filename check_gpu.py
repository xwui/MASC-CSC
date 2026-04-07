import torch
import sys

print("Python:", sys.executable)
try:
    if torch.cuda.is_available():
        print("CUDA available. Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            try:
                mem_alloc = torch.cuda.memory_allocated(i)
                mem_res = torch.cuda.memory_reserved(i)
                print(f"Device {i}: {torch.cuda.get_device_name(i)} - Allocated: {mem_alloc/(1024**2):.2f} MB, Reserved: {mem_res/(1024**2):.2f} MB")
            except Exception as e:
                print(f"Device {i} error: {e}")
    else:
        print("CUDA not available.")
except Exception as e:
    print("Error checking torch:", e)
