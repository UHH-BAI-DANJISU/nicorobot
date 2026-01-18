import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys

# ---------------------------------------------------------
# Import project modules
# ---------------------------------------------------------
# src 디렉토리를 Python path에 추가
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
    num_samples=4096,
    num_iterations=3,
    num_elites=64,
    action_dim=14
):
    """
    Derivative-Free Optimization using Cross-Entropy Method (CEM)
    Finds action a that minimizes E(o, a)
    """
    batch_size = image.shape[0]

    mu = torch.zeros(batch_size, action_dim, device=device)
    std = torch.ones(batch_size, action_dim, device=device)

    for _ in range(num_iterations):
        samples = torch.normal(
            mean=mu.unsqueeze(1).repeat(1, num_samples, 1),
            std=std.unsqueeze(1).repeat(1, num_samples, 1)
        )
        samples = torch.clamp(samples, -1.0, 1.0)

        images_expanded = image.unsqueeze(1).repeat(
            1, num_samples, 1, 1, 1
        ).view(-1, 6, 64, 64)
        flat_samples = samples.view(-1, action_dim)

        with torch.no_grad():
            energies = model(images_expanded, flat_samples)
            energies = energies.view(batch_size, num_samples)

        _, elite_idx = torch.topk(-energies, k=num_elites, dim=1)
        elites = torch.gather(
            samples, 1,
            elite_idx.unsqueeze(-1).expand(-1, -1, action_dim)
        )

        mu = 0.1 * mu + 0.9 * elites.mean(dim=1)
        std = 0.1 * std + 0.9 * elites.std(dim=1).clamp(min=1e-5)

    return mu


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

    for idx in indices:
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_action = sample['action'].unsqueeze(0).to(device)

        start = time.time()

        pred_action = predict_action_cem(
            model, image, device,
            num_samples=4096,
            num_iterations=3,
            num_elites=64
        )

        end = time.time()

        joint_mse = nn.MSELoss()(pred_action, gt_action).item()

        pred_real = model.denormalize(pred_action)
        gt_real = model.denormalize(gt_action)

        pred_pos = pred_real[:, 8:11]
        gt_pos = gt_real[:, 8:11]

        pos_err_cm = torch.norm(pred_pos - gt_pos).item() * 100

        joint_mses.append(joint_mse)
        pos_errors.append(pos_err_cm)
        times.append(end - start)

    print("\n===== FIXED TEST EVALUATION =====")
    print(f"Samples            : {num_samples}")
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

    model_path = "implicit_bc_final.pth"
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
        num_samples=50
    )


if __name__ == "__main__":
    main()
