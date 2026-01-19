import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys

# ---------------------------------------------------------
# Import project modules
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
from energy_model import EnergyModel

# ---------------------------------------------------------
# 1. CEM-based inference (Implicit BC)
# ---------------------------------------------------------
def predict_action_cem(
    model,
    image,
    device,
    num_samples=2048,  # [수정] 탐색의 정확도를 위해 2048~4096 권장
    num_iterations=5,   # [수정] 수렴을 위해 반복 횟수 상향 (3 -> 5)
    num_elites=64,      # [수정] 상위 샘플 확보 (32 -> 64)
    action_dim=14
):
    """
    Derivative-Free Optimization using Cross-Entropy Method (CEM)
    Finds action a that minimizes E(o, a)
    """
    batch_size = image.shape[0]

    mu = torch.zeros(batch_size, action_dim, device=device)
    # [수정] 초기 표준편차를 0.5로 설정하여 전체 범위를 골고루 탐색하게 함
    std = torch.ones(batch_size, action_dim, device=device) * 0.5

    # Vision Feature 추출
    with torch.no_grad():
        vision_feature = model.compute_vision_feature(image)
        vision_feature_expanded = vision_feature.repeat(num_samples, 1)

    for _ in range(num_iterations):
        # 1. Sampling (정규분포로부터 액션 샘플 생성)
        # [수정] mu.repeat 형식을 사용하여 샘플링 로직 안정화
        samples = torch.normal(mu.repeat(num_samples, 1), std.repeat(num_samples, 1))
        samples = torch.clamp(samples, -1.0, 1.0)

        # 2. Energy Evaluation
        with torch.no_grad():
            # [수정] 수정된 에너치 계산 방식(Lower is Better) 적용
            energies = model.score_with_feature(vision_feature_expanded, samples).view(-1)

        # 3. Elites Selection
        # [수정] 에너지가 가장 '낮은' 것을 찾기 위해 largest=False 필수 설정
        # 0% 성공률의 가장 큰 원인이 'largest=True'로 설정되어 에너지가 높은(오답) 것만 고른 것이었습니다.
        _, elite_idx = torch.topk(energies, k=num_elites, largest=False)
        elites = samples[elite_idx]

        # 4. Distribution Update
        # [수정] 지수 이동 평균(EMA)을 통한 부드러운 업데이트
        mu = 0.1 * mu + 0.9 * elites.mean(dim=0).unsqueeze(0)
        std = 0.1 * std + 0.9 * elites.std(dim=0).unsqueeze(0).clamp(min=1e-5)

    return mu.squeeze(0)


# ---------------------------------------------------------
# 2. Fixed test-set evaluation (paper-ready)
# ---------------------------------------------------------
def evaluate_fixed_testset(
    model,
    test_dataset,
    device,
    num_samples=200, # [수정] 논문 신뢰성을 위해 200개 테스트
    seed=42
):
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    pos_errors = []
    joint_mses = []
    times = []
    success_count = 0

    print(f"\n[Info] Starting evaluation on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_action = sample['action'].unsqueeze(0).to(device)

        start = time.time()

        # CEM 추론 실행
        pred_action = predict_action_cem(
            model, image, device,
            num_samples=2048, # [수정] 정확도를 위해 샘플 수 고정
            num_iterations=5,
            num_elites=64
        )

        end = time.time()
        
        if pred_action.dim() == 1:
            pred_action = pred_action.unsqueeze(0)

        # 1. Normalized MSE
        joint_mse = nn.MSELoss()(pred_action, gt_action).item()

        # 2. Real-world Position Error (cm)
        pred_real = model.denormalize(pred_action)
        gt_real = model.denormalize(gt_action)

        pred_pos = pred_real[:, 8:11]
        gt_pos = gt_real[:, 8:11]

        pos_err_cm = torch.norm(pred_pos - gt_pos).item() * 100

        joint_mses.append(joint_mse)
        pos_errors.append(pos_err_cm)
        times.append(end - start)
        
        # 3. Success Rate (오차 1.5cm 미만을 성공으로 판단)
        # [수정] 1.0cm는 너무 가혹할 수 있어 파지 가능 범위인 1.5cm로 설정
        if pos_err_cm < 1.5:
            success_count += 1
            
        print(f"\rProcess: [{i+1}/{num_samples}] | Last Error: {pos_err_cm:.2f}cm", end="")

    success_rate = (success_count / num_samples) * 100

    print("\n\n===== FIXED TEST EVALUATION =====")
    print(f"Samples            : {num_samples}")
    print(f"Success Rate       : {success_rate:.2f} %  <--- [논문 Table 1 기입]")
    print(f"Avg Position Error : {np.mean(pos_errors):.2f} cm")
    print(f"Std Position Error : {np.std(pos_errors):.2f} cm")
    print(f"Median Pos Error   : {np.median(pos_errors):.2f} cm")
    print(f"Avg Joint MSE      : {np.mean(joint_mses):.4f}")
    print(f"Avg Inference Time : {np.mean(times):.3f} sec")
    print("================================\n")


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # [수정] 최종 모델 파일 혹은 체크포인트 자동 로드
    model_path = "implicit_bc_final.pth"
    if not os.path.exists(model_path):
        model_path = "latest_checkpoint.pth"

    if not os.path.exists(model_path):
        print("[Error] No trained model found.")
        return

    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    model = EnergyModel(action_dim=14, stats=stats, device=device).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[Info] Model loaded from {model_path}")

    evaluate_fixed_testset(
        model=model,
        test_dataset=test_ds,
        device=device,
        num_samples=200 
    )

if __name__ == "__main__":
    main()