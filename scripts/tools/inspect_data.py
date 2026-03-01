import numpy as np
import os
import argparse

def inspect_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Inspecting: {file_path}")
    
    try:
        if file_path.endswith('.npz'):
            data_file = np.load(file_path, allow_pickle=True)
            print(f"Keys: {list(data_file.keys())}")
            for key in data_file.keys():
                try:
                    val = data_file[key]
                    print(f"Key: {key}, Shape: {val.shape}, Dtype: {val.dtype}")
                    if val.ndim > 0 and len(val) > 0:
                        print(f"  First element type: {type(val[0])}")
                        if isinstance(val[0], dict):
                            print(f"  First element keys: {val[0].keys()}")
                except Exception as e:
                    print(f"  Could not inspect key {key}: {e}")
        
        elif file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
            print(f"Type: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"Shape: {data.shape}")
            if hasattr(data, 'dtype'):
                print(f"Dtype: {data.dtype}")
            
            if len(data) > 0:
                first = data[0]
                print(f"First element type: {type(first)}")
                if isinstance(first, dict):
                    print(f"Keys: {first.keys()}")
                elif hasattr(first, 'shape'):
                    print(f"First element shape: {first.shape}")
                
                # 检查重复
                duplicates = 0
                unique_count = 0
                try:
                    # 尝试检查唯一性
                    if isinstance(first, np.ndarray):
                        # 对于数组数组
                        flat_data = [d.tobytes() for d in data]
                        unique_count = len(set(flat_data))
                    elif isinstance(first, (int, float, str)):
                        unique_count = len(set(data))
                    else:
                        # 通用情况
                        unique_count = len(set([str(d) for d in data]))
                    
                    for i in range(1, len(data)):
                        if np.array_equal(data[i], data[i-1]):
                            duplicates += 1
                    
                    print(f"Total elements: {len(data)}")
                    print(f"Consecutive duplicates: {duplicates}")
                    print(f"Unique elements: {unique_count}")
                    print(f"Duplicate ratio: {(len(data) - unique_count) / len(data):.2%}")
                except Exception as e:
                    print(f"Error during duplication check: {e}")
        
        else:
            print(f"Unsupported file format: {file_path}")

    except Exception as e:
        print(f"Error loading file: {e}")
    
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .npy or .npz data files.")
    parser.add_argument("--path", type=str, help="Path to the file to inspect")
    parser.add_argument("--dir", type=str, help="Directory containing files to inspect")
    args = parser.parse_args()

    if args.path:
        inspect_data(args.path)
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} not found.")
        else:
            files = [f for f in os.listdir(args.dir) if f.endswith('.npy') or f.endswith('.npz')]
            for f in files:
                inspect_data(os.path.join(args.dir, f))
    else:
        # Default fallback
        default_path = r"d:\Axon\ANN\world\data\demos\benchmark_obs.npy"
        if os.path.exists(default_path):
            inspect_data(default_path)
        else:
            parser.print_help()
