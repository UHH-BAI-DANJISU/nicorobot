import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys
from torchvision.models import resnet18, ResNet18_Weights

# ---------------------------------------------------------
# Import project modules
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR

# ---------------------------------------------------------
# 1. Model Definition (Same as training)
# ---------------------------------------------------------
class ExplicitBCPolicy(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ---------------------------------------------------------
# 2. Helper Function for Denormalization
# ---------------------------------------------------------
def denormalize(action, stats, device):
    action_min = torch.tensor(stats['min'], dtype=torch.float32).to(device)
    action_max = torch.tensor(stats['max'], dtype=torch.float32).to(device)
    return (action + 1) / 2 * (action_max - action_min) + action_min

# ---------------------------------------------------------
# 3. Evaluation Function
# ---------------------------------------------------------
def evaluate_fixed_testset(
    model,
    test_dataset,
    stats,
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

    print(f"\n[Info] Starting Explicit Evaluation on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device) # [1, 6, 64, 64]
        gt_action = sample['action'].unsqueeze(0).to(device)

        # --- Direct Inference ---
        start = time.time()
        with torch.no_grad():
            pred_action = model(image) # One-step prediction
        end = time.time()
        times.append(end - start)

        # Real-world scale conversion
        pred_real = denormalize(pred_action, stats, device)
        gt_real = denormalize(gt_action, stats, device)

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

    print("\n\n===== EXPLICIT EVALUATION RESULTS =====")
    print(f"Total Samples      : {num_samples}")
    print(f"Success Rate       : {(success_count/num_samples)*100:.2f} %")
    print(f"Avg Position Error : {np.mean(pos_errors):.2f} cm")
    print(f"Inference Time     : {np.mean(times):.5f} sec/sample") # Typically much faster than CEM
    print("==============================\n")

# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load Model Path
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
    model_path = os.path.join(PROJECT_ROOT, "explicit_bc_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found at: {model_path}")
        return

    # Load Stats & Dataset
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    # Init Model
    model = ExplicitBCPolicy(action_dim=14).to(device)

    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    # state_dict만 저장된 경우와 체크포인트 딕셔너리로 저장된 경우 모두 대응
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"[Info] Explicit model loaded from {model_path}")

    # Run Eval
    evaluate_fixed_testset(
        model=model,
        test_dataset=test_ds,
        stats=stats,
        device=device,
        num_samples=200 
    )

if __name__ == "__main__":
    main()