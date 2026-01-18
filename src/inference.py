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
    num_samples=4096,  # [수정] 4096 -> 1024 (CPU 최적화)
    num_iterations=3,
    num_elites=32,     # [수정] 32 -> 64 (상위 샘플 충분히 확보)
    action_dim=14
):
    """
    Derivative-Free Optimization using Cross-Entropy Method (CEM)
    Finds action a that minimizes E(o, a)
    """
    batch_size = image.shape[0]

    mu = torch.zeros(batch_size, action_dim, device=device)
    std = torch.ones(batch_size, action_dim, device=device)

    # [최적화 1] Vision Feature 한 번만 추출! (속도 핵심)
    with torch.no_grad():
        vision_feature = model.compute_vision_feature(image) # [1, 512]
        # 샘플 개수만큼 복제
        vision_feature_expanded = vision_feature.repeat(num_samples, 1) # [1024, 512]

    for _ in range(num_iterations):
        # 1. Sampling
        samples = torch.normal(
            mean=mu.unsqueeze(1).repeat(1, num_samples, 1),
            std=std.unsqueeze(1).repeat(1, num_samples, 1)
        ).view(num_samples, action_dim)
        
        samples = torch.clamp(samples, -1.0, 1.0)

        # 2. Energy Evaluation (Optimized)
        with torch.no_grad():
            # [최적화 2] ResNet 없이 가벼운 Head만 통과
            energies = model.score_with_feature(vision_feature_expanded, samples).view(-1)

        # 3. Elites Selection
        # 에너지가 낮은 순서대로 정렬 (Minimization)
        _, elite_idx = torch.topk(-energies, k=num_elites)
        elites = samples[elite_idx] # [num_elites, 14]

        # 4. Distribution Update
        mu = 0.1 * mu + 0.9 * elites.mean(dim=0).unsqueeze(0)
        std = 0.1 * std + 0.9 * elites.std(dim=0).unsqueeze(0).clamp(min=1e-5)

    return mu.squeeze(0) # [14]


# ---------------------------------------------------------
# 2. Fixed test-set evaluation (paper-ready)
# ---------------------------------------------------------
def evaluate_fixed_testset(
    model,
    test_dataset,
    device,
    num_samples=50,
    seed=42
):
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    pos_errors = []
    joint_mses = []
    times = []
    
    # [추가] 성공률 계산을 위한 리스트
    success_count = 0

    print(f"\n[Info] Starting evaluation on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_action = sample['action'].unsqueeze(0).to(device)

        start = time.time()

        # [수정] 여기서 num_samples를 1024로 전달
        pred_action = predict_action_cem(
            model, image, device,
            num_samples=4096, # CPU 타협점
            num_iterations=3,
            num_elites=32
        )

        end = time.time()
        
        # 차원 맞추기 [14] -> [1, 14]
        if pred_action.dim() == 1:
            pred_action = pred_action.unsqueeze(0)

        joint_mse = nn.MSELoss()(pred_action, gt_action).item()

        pred_real = model.denormalize(pred_action)
        gt_real = model.denormalize(gt_action)

        pred_pos = pred_real[:, 8:11]
        gt_pos = gt_real[:, 8:11]

        # 오차 계산 (cm 단위)
        pos_err_cm = torch.norm(pred_pos - gt_pos).item() * 100

        joint_mses.append(joint_mse)
        pos_errors.append(pos_err_cm)
        times.append(end - start)
        
        # [추가] 성공 여부 판단 (오차 1.0cm 미만 + Orientation 생략)
        if pos_err_cm < 1.0:
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

    model_path = "latest_checkpoint.pth"
    if not os.path.exists(model_path):
        if os.path.exists("latest_checkpoint.pth"):
            print("[Warning] Using checkpoint instead of final model.")
            model_path = "latest_checkpoint.pth"
        else:
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

    print("\n[Inference] Running fixed evaluation for paper...")
    evaluate_fixed_testset(
        model=model,
        test_dataset=test_ds,
        device=device,
        num_samples=200 # 테스트용 샘플 수 (논문 제출용으론 100~200 권장)
    )

if __name__ == "__main__":
    main()