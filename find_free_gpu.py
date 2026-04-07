import torch

def find_best_gpu():
    best_gpu = -1
    max_free = 0
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: Free {free/(1024**3):.2f} GB / Total {total/(1024**3):.2f} GB")
            if free > max_free:
                max_free = free
                best_gpu = i
        except Exception as e:
            print(f"GPU {i} error:", e)
    
    if best_gpu >= 0 and max_free > 10 * 1024**3: # At least 10GB free
        print(f"\nBest GPU is {best_gpu} with {max_free/(1024**3):.2f} GB free")
    else:
        print("\nNo GPU with enough memory found!")

if __name__ == '__main__':
    find_best_gpu()
