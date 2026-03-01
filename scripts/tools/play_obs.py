import numpy as np
import cv2
import os
import argparse

def play_obs(file_path, num_frames=1000, fps=30, width=240, height=135):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    try:
        data_file = np.load(file_path, mmap_mode='r')
        if 'frames' in data_file:
            data = data_file['frames']
            print(f"Found 'frames' in npz. Shape: {data.shape}")
        else:
            data = data_file
            print(f"Loading as raw npy/npz. Shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 检查维度 [N, C, H, W] 或 [N, H, W, C]
    if data.ndim != 4:
        print(f"Unexpected data dimensions: {data.ndim}. Expected 4.")
        return

    # 自动转换 [N, C, H, W] -> [N, H, W, C]
    if data.shape[1] == 3 or data.shape[1] == 1:
        data = np.transpose(data, (0, 2, 3, 1))

    wait_time = int(1000 / fps)
    
    print(f"Playing at {fps} FPS. Target Size: {width}x{height}")
    print("Controls: 'q' to quit, 'r' to toggle RGB/BGR swap")
    
    swap_rb = True # 默认开启，因为原始数据是 RGB，OpenCV 需要 BGR
    
    for i in range(min(num_frames, len(data))):
        frame = data[i].copy()
        
        # 确保是 uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # 颜色空间处理
        if swap_rb:
            # 数据是 RGB，OpenCV 需要 BGR 才能正确显示
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame = frame

        # 缩放处理
        display_frame = cv2.resize(display_frame, (width, height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Sekiro Observation Check', display_frame)
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            swap_rb = not swap_rb
            print(f"RGB/BGR Swap: {swap_rb}")
            
    cv2.destroyAllWindows()
    print("Playback finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play observation frames from .npy or .npz data files.")
    parser.add_argument("--path", type=str, help="Path to the file to play")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to play")
    args = parser.parse_args()

    if args.path:
        play_obs(args.path, num_frames=args.num_frames, fps=args.fps)
    else:
        # Default fallback
        default_path = r"d:\Axon\ANN\World-Studio\data\demos\sekiro_wm_20260301_023500.npz"
        if os.path.exists(default_path):
            play_obs(default_path, num_frames=args.num_frames, fps=args.fps)
        else:
            parser.print_help()
