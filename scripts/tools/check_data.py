import torch
import os
import numpy as np
import argparse

def check_file(file_path):
    print(f"Checking {file_path}...")
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"Data in {file_path} is not a list, it is {type(data)}")
        return

    print(f"Total steps in file: {len(data)}")
    
    # 查找观察键 (observation key)
    obs_key = None
    if len(data) > 0:
        # print(f"Sample data structure: {data[0]}")
        for key in ['obs', 'image', 'state', 'basic']:
            if key in data[0]:
                obs_key = key
                break
    
    if not obs_key:
        print(f"Could not find observation key in {data[0].keys() if len(data) > 0 else 'empty list'}")
        return

    images = []
    telemetries = []
    actions = []
    for i, step in enumerate(data):
        # 深度提取 policy 图像、遥测和动作
        img = step
        tel = step
        act = step
        try:
            if 'basic' in step:
                img = step['basic']
                tel = step['basic']
                act = step['basic']
            
            if 'obs' in img:
                img = img['obs']
                tel = img # telemetry usually in obs
            
            if 'policy' in img:
                img = img['policy']
            
            if 'telemetry' in tel:
                tel = tel['telemetry']
            
            if 'action' in act:
                act = act['action']
        except Exception:
            pass
            
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(tel, torch.Tensor):
            tel = tel.detach().cpu().numpy()
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
            
        images.append(img)
        telemetries.append(tel)
        actions.append(act)
    
    # 检查连续重复
    img_dups = 0
    tel_dups = 0
    act_dups = 0
    for i in range(1, len(images)):
        if np.array_equal(images[i], images[i-1]):
            img_dups += 1
        if np.array_equal(telemetries[i], telemetries[i-1]):
            tel_dups += 1
        if np.array_equal(actions[i], actions[i-1]):
            act_dups += 1
            
    # 检查全局唯一性
    unique_images = len(set([img.tobytes() if hasattr(img, 'tobytes') else str(img) for img in images]))
    unique_tel = len(set([t.tobytes() if hasattr(t, 'tobytes') else str(t) for t in telemetries]))
    unique_act = len(set([a.tobytes() if hasattr(a, 'tobytes') else str(a) for a in actions]))
    
    print(f"Image - Consecutive dups: {img_dups}, Unique: {unique_images}/{len(images)}")
    print(f"Telemetry - Consecutive dups: {tel_dups}, Unique: {unique_tel}/{len(telemetries)}")
    print(f"Action - Consecutive dups: {act_dups}, Unique: {unique_act}/{len(actions)}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check .pt data files for duplication.")
    parser.add_argument("--dir", type=str, default=r"d:\Axon\ANN\world\data\demos", help="Directory containing .pt files")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} not found.")
    else:
        files = [f for f in os.listdir(args.dir) if f.endswith('.pt')]
        for f in files:
            check_file(os.path.join(args.dir, f))
