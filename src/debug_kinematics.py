import torch
import pandas as pd
import numpy as np
import os
import pytorch_kinematics as pk
from dfk_layer import DifferentiableFK

# Configuration
DATA_DIR = 'data/real_evo_ik_samples' 
CSV_PATH = os.path.join(DATA_DIR, 'samples.csv')
URDF_PATH = 'complete.urdf'

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return

    # 1. Load Data
    print(f"[Info] Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Joint order as defined in dataset.py: 6-DoF arm + 2-DoF head (using arm for FK check)
    joint_cols = ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x']
    pos_cols = ['hand_pos_x', 'hand_pos_y', 'hand_pos_z']
    
    joints = torch.tensor(df[joint_cols].values, dtype=torch.float32)
    gt_pos = torch.tensor(df[pos_cols].values, dtype=torch.float32)

    # 3. Initialize Differentiable Forward Kinematics
    dfk = DifferentiableFK(device='cpu', urdf_path=URDF_PATH)

    # 4. Diagnostic 1: Unit Verification (Radian vs Degree)
    print("\n--- [진단 1] Joint Data Range Check ---")
    min_vals = joints.min(dim=0).values
    max_vals = joints.max(dim=0).values
    print(f"Min Joint Values: {min_vals.numpy()}")
    print(f"Max Joint Values: {max_vals.numpy()}")
    
    is_degree = False
    if (max_vals > 3.2).any() or (min_vals < -3.2).any():
        print(">>> Warning: Values exceed PI range. Data is likely in [DEGREES].")
        is_degree = True
    else:
        print(">>> Confirmed: Values are within PI range. Data is in [RADIANS].")

    # 5. Diagnostic 2: Kinematic Mismatch Check
    print("\n--- [Diagnostic 2] Kinematic Mismatch Check ---")
    
    if is_degree:
        print("[Info] Degree -> Radian 변환 후 DFK 계산 시도...")
        joints_input = joints * (3.141592 / 180.0)
    else:
        print("[Info] Raw Data 그대로 DFK 계산 시도...")
        joints_input = joints

    pred_pos = dfk(joints_input)
    
    # Calculate Euclidean error
    errors = torch.norm(pred_pos - gt_pos, dim=1)
    mean_error = errors.mean().item()
    min_error = errors.min().item()
    max_error = errors.max().item()

    print(f"Mean Error: {mean_error * 100:.2f} cm")
    print(f"Min Error : {min_error * 100:.2f} cm")
    print(f"Max Error : {max_error * 100:.2f} cm")

    # 6. Diagnostic 3: TCP (Tool Center Point) Offset Estimation
    # If the error is consistent, it likely indicates the distance between Palm and Fingertip.
    diff_vec = gt_pos - pred_pos
    mean_offset = diff_vec.mean(dim=0)
    print("\n--- [Diagnostic 3] Estimated Offset Check ---")
    print(f"Mean offset vector (GT - DFK): {mean_offset.numpy()}")
    print(f"Offset magnitude (vector norm): {torch.norm(mean_offset).item() * 100:.2f} cm")
    if mean_error > 0.05: # Threshold: 5cm
        print("\n>>> Analysis: Significant Mismatch Detected")
        if torch.norm(mean_offset).item() > 0.05:
            print(f"    Likely Cause: [TCP Offset Missing] DFK is using the palm, but the ground truth seems to be the fingertip (about {torch.norm(mean_offset).item()*100:.1f}cm ahead).")
        else:
            print("    Likely Cause: [Coordinate/Unit Issue] Robot base position may differ or axes may be misaligned.")
    else:
        print("\n>>> Conclusion: DFK and the data match well. (Error less than 5cm)")

if __name__ == "__main__":
    main()