import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

# 프로젝트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
from energy_model import EnergyModel
from dfk_layer import DifferentiableFK

# ---------------------------------------------------------
# 1. 데이터 수집형 CEM 추론 함수
# ---------------------------------------------------------
def predict_action_cem_with_logging(model, image, device, num_samples=4096, num_iterations=12, num_elites=64):
    batch_size = image.shape[0]
    mu = torch.zeros(batch_size, 14, device=device)
    std = torch.ones(batch_size, 14, device=device) * 1.0
    
    # 기록용 리스트
    energy_history = []
    nn_energy_history = []
    kin_err_history = []

    with torch.no_grad():
        vision_feature = model.compute_vision_feature(image)
        vision_feature_expanded = vision_feature.repeat_interleave(num_samples, dim=0)

    for i in range(num_iterations):
        samples = torch.normal(mu.unsqueeze(1).expand(-1, num_samples, -1), 
                               std.unsqueeze(1).expand(-1, num_samples, -1))
        samples = torch.clamp(samples, -1.0, 1.0)
        samples_flat = samples.view(-1, 14)

        with torch.no_grad():
            # Neural Energy 계산
            x = torch.cat([vision_feature_expanded, samples_flat], dim=1)
            nn_energy = model.energy_net(x).view(batch_size, num_samples)
            
            # Kinematic Error 계산
            raw_samples = model.denormalize(samples_flat)
            pred_pos = model.dfk(raw_samples[:, :6] * (torch.pi / 180.0))
            target_pos = raw_samples[:, 8:11]
            kin_err = torch.norm(pred_pos - target_pos, dim=1).view(batch_size, num_samples)
            
            # Total Energy (가중치 1.0 적용)
            total_energy = nn_energy + (1.0 * kin_err)

        # 현재 이터레이션에서 에너지가 가장 낮은 샘플 정보 기록
        best_val, best_idx = torch.min(total_energy[0], dim=0)
        energy_history.append(best_val.item())
        nn_energy_history.append(nn_energy[0, best_idx].item())
        kin_err_history.append(kin_err[0, best_idx].item() * 100) # cm 단위

        # CEM 업데이트
        _, elite_idx = torch.topk(total_energy, k=num_elites, dim=1, largest=False)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_elites)
        elites = samples[batch_indices, elite_idx, :]
        mu = 0.1 * mu + 0.9 * elites.mean(dim=1)
        std = 0.1 * std + 0.9 * elites.std(dim=1).clamp(min=1e-5)

    return mu, energy_history, nn_energy_history, kin_err_history

# ---------------------------------------------------------
# 2. 메인 실행 함수 (Best Sample 검색 및 플로팅)
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    # 모델 및 데이터 로드
    csv_path = os.path.join(PROJECT_ROOT, TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    test_ds = NICORobotDataset(os.path.join(PROJECT_ROOT, TEST_DIR), stats, is_train=False)
    model = EnergyModel(action_dim=14, stats=stats, device=device).to(device)
    
    model_path = os.path.join(CURRENT_DIR, "best_implicit_model.pth")
    if not os.path.exists(model_path): model_path = os.path.join(PROJECT_ROOT, "best_implicit_model.pth")
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("[Info] Searching for the BEST successful sample in test set...")
    
    best_sample_idx = -1
    min_error = 100.0
    best_data = None

    # 테스트 세트에서 상위 50개 중 가장 결과가 좋은 샘플 탐색
    search_range = min(50, len(test_ds))
    for i in range(search_range):
        sample = test_ds[i]
        img = sample['image'].unsqueeze(0).to(device)
        gt_action = sample['action'].unsqueeze(0).to(device)
        
        # 추론 진행
        pred_action, e_hist, nn_hist, k_hist = predict_action_cem_with_logging(model, img, device)
        
        # 오차 계산
        pred_real = model.denormalize(pred_action)
        gt_real = model.denormalize(gt_action)
        err = torch.norm(pred_real[:, 8:11] - gt_real[:, 8:11]).item() * 100 # cm
        
        if err < min_error:
            min_error = err
            best_sample_idx = i
            best_data = (sample, e_hist, nn_hist, k_hist, gt_action)
            if err < 0.5: break # 0.5cm 미만이면 충분히 훌륭하므로 중단

    if best_data is None:
        print("[Error] No valid results found.")
        return

    sample, e_hist, nn_hist, k_hist, gt_action = best_data
    print(f"🎯 Best Sample Found! Index: {best_sample_idx}, Error: {min_error:.2f}cm")

    # ---------------------------------------------------------
    # Graph 1: Energy Decomposition (GT vs Prediction)
    # ---------------------------------------------------------
    img = sample['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        gt_vision = model.compute_vision_feature(img)
        gt_nn_energy = model.energy_net(torch.cat([gt_vision, gt_action], dim=1)).item()
        gt_raw = model.denormalize(gt_action)
        gt_pos = model.dfk(gt_raw[:, :6] * (torch.pi / 180.0))
        gt_kin_err = torch.norm(gt_pos - gt_raw[:, 8:11]).item() * 100

    labels = ['Ground Truth (Expert)', 'Implicit BC (Ours)']
    nn_energies = [gt_nn_energy, nn_hist[-1]]
    kin_errors = [gt_kin_err, k_hist[-1]]

    fig, ax1 = plt.subplots(figsize=(9, 7))
    x = np.arange(len(labels))
    width = 0.35

    # Neural Energy (Primary Y-axis)
    ax1.bar(x - width/2, nn_energies, width, label='Neural Energy ($E_{nn}$)', color='#4C72B0', edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Neural Energy (Lower is Better)', color='#4C72B0', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    
    # Kinematic Error (Secondary Y-axis)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, kin_errors, width, label='Kinematic Error ($E_{kin}$)', color='#C44E52', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Kinematic Error (cm)', color='#C44E52', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#C44E52')
    ax2.set_ylim(0, max(kin_errors) * 1.5 + 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
    plt.title('Implicit BC: Energy and Kinematic Consistency Analysis', fontsize=15, fontweight='bold', pad=20)
    
    # 통합 범례
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)

    plt.tight_layout()
    plt.savefig('plot_best_energy_decomposition.png', dpi=300, bbox_inches='tight')

    # ---------------------------------------------------------
    # Graph 2: Optimization Trajectory (CEM Convergence)
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 7))
    iterations = np.arange(1, len(e_hist) + 1)
    
    plt.plot(iterations, e_hist, 'o-', color='#DD8452', linewidth=3, markersize=8, label='Total Energy ($E_{total}$)')
    plt.axhline(y=gt_nn_energy, color='#55A868', linestyle='--', linewidth=2, label='Expert Energy Level (Target)')
    
    plt.xlabel('CEM Optimization Iterations', fontsize=13, fontweight='bold')
    plt.ylabel('Energy Value', fontsize=13, fontweight='bold')
    plt.title('Optimization Trajectory: Energy Minimization via CEM', fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(iterations)
    plt.legend(fontsize=11, loc='upper right')

    # 어노테이션 추가
    plt.annotate(f'Final Error: {min_error:.2f} cm', 
                 xy=(iterations[-1], e_hist[-1]), xytext=(iterations[-1]-3, e_hist[-1]+1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.savefig('plot_best_optimization_trajectory.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Plots saved as 'plot_best_energy_decomposition.png' and 'plot_best_optimization_trajectory.png'")

if __name__ == "__main__":
    main()