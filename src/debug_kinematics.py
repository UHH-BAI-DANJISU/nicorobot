import torch
import pandas as pd
import numpy as np
import os
import pytorch_kinematics as pk
from dfk_layer import DifferentiableFK

# 설정
DATA_DIR = 'data/real_evo_ik_samples' # 데이터 경로 (필요시 수정)
CSV_PATH = os.path.join(DATA_DIR, 'samples.csv')
URDF_PATH = 'complete.urdf'

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return

    # 1. 데이터 로드
    print(f"[Info] Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 2. Joint와 GT Position 추출
    # dataset.py의 순서: l_shoulder_z, l_shoulder_y, l_arm_x, l_elbow_y, l_wrist_z, l_wrist_x
    joint_cols = ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x']
    pos_cols = ['hand_pos_x', 'hand_pos_y', 'hand_pos_z']
    
    joints = torch.tensor(df[joint_cols].values, dtype=torch.float32)
    gt_pos = torch.tensor(df[pos_cols].values, dtype=torch.float32)

    # 3. DFK 준비
    dfk = DifferentiableFK(device='cpu', urdf_path=URDF_PATH)

    # 4. 진단 1: 단위(Unit) 확인
    print("\n--- [진단 1] Joint Data Range Check ---")
    min_vals = joints.min(dim=0).values
    max_vals = joints.max(dim=0).values
    print(f"Min Joint Values: {min_vals.numpy()}")
    print(f"Max Joint Values: {max_vals.numpy()}")
    
    is_degree = False
    if (max_vals > 3.2).any() or (min_vals < -3.2).any():
        print(">>> ⚠️ 경고: 값의 범위가 3.14를 넘습니다. 데이터가 [DEGREE] 단위일 확률이 높습니다.")
        is_degree = True
    else:
        print(">>> ✅ 확인: 값의 범위가 PI 내외입니다. 데이터는 [RADIAN] 단위입니다.")

    # 5. 진단 2: DFK 오차(Mismatch) 확인
    print("\n--- [진단 2] Kinematic Mismatch Check ---")
    
    # (옵션) 만약 Degree라면 변환해서 테스트
    if is_degree:
        print("[Info] Degree -> Radian 변환 후 DFK 계산 시도...")
        joints_input = joints * (3.141592 / 180.0)
    else:
        print("[Info] Raw Data 그대로 DFK 계산 시도...")
        joints_input = joints

    pred_pos = dfk(joints_input)
    
    # 오차 계산
    errors = torch.norm(pred_pos - gt_pos, dim=1)
    mean_error = errors.mean().item()
    min_error = errors.min().item()
    max_error = errors.max().item()

    print(f"Mean Error: {mean_error * 100:.2f} cm")
    print(f"Min Error : {min_error * 100:.2f} cm")
    print(f"Max Error : {max_error * 100:.2f} cm")

    # 6. 진단 3: 오프셋(TCP Offset) 추정
    # 만약 에러가 일정하다면, 그건 Palm -> Fingertip 사이의 거리입니다.
    diff_vec = gt_pos - pred_pos
    mean_offset = diff_vec.mean(dim=0)
    print("\n--- [진단 3] Estimated Offset Check ---")
    print(f"평균 오프셋 벡터 (GT - DFK): {mean_offset.numpy()}")
    print(f"이 벡터의 길이 (Offset Magnitude): {torch.norm(mean_offset).item() * 100:.2f} cm")
    
    if mean_error > 0.05: # 5cm 이상 차이나면
        print("\n>>> 🚨 결론: 치명적인 불일치 발생!")
        if torch.norm(mean_offset).item() > 0.05:
            print(f"    원인 추정: [TCP Offset 누락] DFK는 손바닥인데, 정답은 손끝(약 {torch.norm(mean_offset).item()*100:.1f}cm 앞)인 것 같습니다.")
        else:
            print("    원인 추정: [좌표계/단위 문제] 로봇 베이스 위치가 다르거나 축이 꼬여 있습니다.")
    else:
        print("\n>>> ✅ 결론: DFK와 데이터가 잘 맞습니다. (오차 5cm 미만)")

if __name__ == "__main__":
    main()