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
from dfk_layer import DifferentiableFK  #  (개념적 참조)

# ---------------------------------------------------------
# 1. CEM-based inference (Implicit BC)
# ---------------------------------------------------------
def predict_action_cem(
    model,
    dfk_layer,
    image,
    device,
    num_samples=2048,
    num_iterations=5,
    num_elites=64,
    action_dim=14
):
    """
    Finds action 'a' that minimizes Total Energy = Neural Energy + Kinematic Error
    """
    batch_size = image.shape[0]

    mu = torch.zeros(batch_size, action_dim, device=device)
    std = torch.ones(batch_size, action_dim, device=device) * 0.5

    # 1. Vision Feature 미리 추출 (속도 최적화)
    with torch.no_grad():
        vision_feature = model.compute_vision_feature(image)
        # [B, 1024] -> [B * Samples, 1024]
        vision_feature_expanded = vision_feature.repeat_interleave(num_samples, dim=0)

    for i in range(num_iterations):
        # --- Sampling ---
        # [B, Samples, Dim] 형태로 생성 후 펼치기
        samples = torch.normal(mu.unsqueeze(1).repeat(1, num_samples, 1), 
                               std.unsqueeze(1).repeat(1, num_samples, 1))
        samples = torch.clamp(samples, -1.0, 1.0)
        
        # Flatten for batch processing: [B * Samples, 14]
        samples_flat = samples.view(-1, action_dim)

        # --- Energy Evaluation ---
        with torch.no_grad():
            # A. Neural Energy (from learned model)
            neural_energy = model.score_with_feature(vision_feature_expanded, samples_flat).view(batch_size, num_samples)
            
            # B. Kinematic Error (Physical Consistency)
            # Denormalize to Real Scale
            raw_samples = model.denormalize(samples_flat)
            
            # Joint (Degree) -> Radian
            joints_deg = raw_samples[:, :6] # Arm joints only for DFK
            joints_rad = joints_deg * (torch.pi / 180.0)
            
            # DFK Forward
            pred_pos = dfk_layer(joints_rad) # [B*S, 3]
            target_pos = raw_samples[:, 8:11] # [B*S, 3] (Hand Pos columns)
            
            # Euclidean Distance Error
            kinematic_error = torch.norm(pred_pos - target_pos, dim=1).view(batch_size, num_samples)

            # C. Total Energy
            # alpha=10.0 (가중치)
            total_energy = neural_energy + (10.0 * kinematic_error)

        # --- Elites Selection ---
        # 에너지가 가장 낮은(Best) 샘플 선택
        _, elite_idx = torch.topk(total_energy, k=num_elites, dim=1, largest=False)
        
        # elite_idx: [B, Elites] -> Gather elites
        # samples: [B, Samples, Dim]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_elites)
        elites = samples[batch_indices, elite_idx, :] # [B, Elites, Dim]

        # --- Distribution Update (Soft Update) ---
        new_mu = elites.mean(dim=1)
        new_std = elites.std(dim=1).clamp(min=1e-5)
        
        mu = 0.1 * mu + 0.9 * new_mu
        std = 0.1 * std + 0.9 * new_std

    return mu # [B, 14]


# ---------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------
def evaluate_fixed_testset(
    model,
    dfk_layer,
    test_dataset,
    device,
    num_samples=200,
    seed=42
):
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random sampling from test set
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    pos_errors = []
    success_count = 0
    times = []

    print(f"\n[Info] Starting evaluation on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device) # [1, 6, 64, 64]
        gt_action = sample['action'].unsqueeze(0).to(device)

        start = time.time()

        # Inference
        pred_action = predict_action_cem(
            model, dfk_layer, image, device,
            num_samples=2048,
            num_iterations=5,
            num_elites=64
        )

        end = time.time()
        times.append(end - start)

        # Real-world scale conversion
        pred_real = model.denormalize(pred_action)
        gt_real = model.denormalize(gt_action)

        # Position Error (columns 8,9,10 are x,y,z)
        pred_pos = pred_real[:, 8:11]
        gt_pos = gt_real[:, 8:11]

        # Meter -> cm
        pos_err_cm = torch.norm(pred_pos - gt_pos).item() * 100
        pos_errors.append(pos_err_cm)

        # Success Check (1.5cm threshold)
        if pos_err_cm < 1.5:
            success_count += 1
            
        print(f"\r[{i+1}/{num_samples}] Error: {pos_err_cm:.2f} cm | Success Rate: {(success_count/(i+1))*100:.1f}%", end="")

    print("\n\n===== EVALUATION RESULTS =====")
    print(f"Total Samples      : {num_samples}")
    print(f"Success Rate       : {(success_count/num_samples)*100:.2f} %")
    print(f"Avg Position Error : {np.mean(pos_errors):.2f} cm")
    print(f"Inference Time     : {np.mean(times):.3f} sec/sample")
    print("==============================\n")


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load Model Path
    model_path = os.path.join(CURRENT_DIR, "best_implicit_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(CURRENT_DIR, "latest_checkpoint.pth")
    
    if not os.path.exists(model_path):
        print("[Error] No model checkpoint found!")
        return

    # Load Stats & Dataset
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    # Init Models
    model = EnergyModel(action_dim=14, stats=stats, device=device).to(device)
    
    # Init DFK (for Kinematic Error calculation during inference)
    # URDF 파일은 src 폴더 내 complete.urdf를 찾음
    dfk_layer = DifferentiableFK(device=device, urdf_path='complete.urdf')

    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"[Info] Model loaded from {model_path}")

    # Run Eval
    evaluate_fixed_testset(
        model=model,
        dfk_layer=dfk_layer,
        test_dataset=test_ds,
        device=device,
        num_samples=100 # 테스트 샘플 수 조정 가능
    )

if __name__ == "__main__":
    main()