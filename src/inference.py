import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# 자네가 만든 모듈들 임포트
try:
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from energy_model import EnergyModel
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from models.energy_model import EnergyModel

# ---------------------------------------------------------
# 1. CEM (Cross-Entropy Method) 기반 추론 함수
# ---------------------------------------------------------
def predict_action_cem(model, image, device, 
                       num_samples=4096, 
                       num_iterations=3, 
                       num_elites=64,
                       action_dim=14):
    """
    Derivative-Free Optimization (DFO) using CEM.
    에너지 함수 E(o, a)를 최소화하는 행동 a를 찾는다.
    """
    batch_size = image.shape[0]
    
    # 초기 분포: 평균(mu)=0, 표준편차(std)=1 (범위가 -1~1 정규화되었으므로)
    mu = torch.zeros(batch_size, action_dim, device=device)
    std = torch.ones(batch_size, action_dim, device=device)
    
    # 반복 최적화 (Iterative Refinement)
    for i in range(num_iterations):
        # 1. 샘플링: 가우시안 분포 N(mu, std)에서 후보 행동 생성
        # [B, N, 14]
        samples = torch.normal(mean=mu.unsqueeze(1).repeat(1, num_samples, 1), 
                               std=std.unsqueeze(1).repeat(1, num_samples, 1))
        
        # 범위 클리핑 (-1 ~ 1)
        samples = torch.clamp(samples, -1.0, 1.0)
        
        # 2. 이미지 확장: [B, 6, 64, 64] -> [B*N, 6, 64, 64]
        # (주의: 메모리를 많이 먹으므로 배치 처리가 필요할 수 있음. 여기서는 단순화함)
        images_expanded = image.unsqueeze(1).repeat(1, num_samples, 1, 1, 1).view(-1, 6, 64, 64)
        flat_samples = samples.view(-1, action_dim)
        
        # 3. 에너지 평가 (작을수록 좋음)
        with torch.no_grad():
            energies = model(images_expanded, flat_samples).view(batch_size, num_samples)
        
        # 4. 엘리트 선택 (에너지가 가장 낮은 상위 K개)
        # topk는 큰 값을 뽑으므로, 에너지에 마이너스를 붙여서 가장 작은 걸 뽑음
        top_scores, top_indices = torch.topk(-energies, k=num_elites, dim=1)
        
        # [B, K, 14]
        elites = torch.gather(samples, 1, top_indices.unsqueeze(-1).expand(-1, -1, action_dim))
        
        # 5. 분포 업데이트 (다음 반복을 위해 평균과 표준편차 재조정)
        new_mu = elites.mean(dim=1)
        new_std = elites.std(dim=1) + 1e-5 # 0이 되는 것 방지
        
        # Soft Update (이전 분포를 살짝 유지하여 안정성 확보)
        mu = 0.1 * mu + 0.9 * new_mu
        std = 0.1 * std + 0.9 * new_std

    # 최종 결과: 가장 평균에 가까운 값 (혹은 엘리트 중 1등) 반환
    return mu

# ---------------------------------------------------------
# 2. 메인 실행 함수
# ---------------------------------------------------------
def main():
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    model_path = "implicit_bc_final.pth" # 학습된 모델 경로
    if not os.path.exists(model_path):
        # 없으면 체크포인트라도 찾기
        if os.path.exists("latest_checkpoint.pth"):
            print("[Warning] Final model not found. Loading checkpoint instead.")
            checkpoint = torch.load("latest_checkpoint.pth", map_location=device)
            # 체크포인트 구조에 따라 다름 (model_state_dict 키가 있는지 확인 필요)
            # 여기서는 편의상 경로만 바꿈 (아래 로드 로직에서 처리)
            model_path = "latest_checkpoint.pth" 
        else:
            print("[Error] No model file found!")
            return

    # 1. 데이터셋 준비 (통계값 필요)
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    
    # 테스트셋 로드
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True) # 하나씩 보면서 검증

    # 2. 모델 로드
    model = EnergyModel(action_dim=14, stats=stats, device=device).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Info] Model loaded from {model_path}")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    model.eval()

    # 3. 추론 및 평가
    print("\n[Start Inference] Running evaluation on 5 random samples...")
    print("-" * 60)
    print(f"{'Sample ID':<10} | {'Joint MSE':<12} | {'Pos Err (cm)':<12} | {'Inference Time':<15}")
    print("-" * 60)

    total_pos_error = 0.0
    count = 0
    
    # 5개만 테스트해보기
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5: break
            
            image = batch['image'].to(device)
            gt_action = batch['action'].to(device) # 정답 (-1~1)
            
            start_time = time.time()
            
            # --- [핵심] Implicit BC 추론 ---
            # N=4096 샘플, 3번 반복 (논문 수준 설정)
            pred_action_norm = predict_action_cem(model, image, device, 
                                                  num_samples=4096, 
                                                  num_iterations=3, 
                                                  num_elites=64)
            
            end_time = time.time()
            
            # --- 평가 ---
            # 1. Joint MSE (정규화된 공간에서의 오차)
            joint_mse = nn.MSELoss()(pred_action_norm, gt_action).item()
            
            # 2. Cartesian Position Error (실제 m 단위 오차)
            # 복원(Denormalize)
            pred_real = model.denormalize(pred_action_norm)
            gt_real = model.denormalize(gt_action)
            
            # dataset.py 기준: index 8, 9, 10이 x, y, z 좌표
            pred_pos = pred_real[:, 8:11]
            gt_pos = gt_real[:, 8:11]
            
            pos_error_m = torch.norm(pred_pos - gt_pos).item() # 유클리드 거리
            pos_error_cm = pos_error_m * 100 # cm 변환
            
            total_pos_error += pos_error_cm
            count += 1
            
            print(f"{i:<10} | {joint_mse:.6f}     | {pos_error_cm:.2f} cm      | {(end_time - start_time):.3f} sec")

    print("-" * 60)
    print(f"Average Position Error: {total_pos_error / count:.2f} cm")
    
    if (total_pos_error / count) < 2.0:
        print("\n[Conclusion] 2cm 이내 오차! 아주 훌륭한 성능입니다.")
        print("교수 평가: '이 정도면 CoRL 논문감일세. 자네 수고했네.'")
    elif (total_pos_error / count) < 5.0:
        print("\n[Conclusion] 5cm 이내 오차. 준수하지만 정밀 조작엔 아쉬움.")
        print("교수 평가: '조금 더 다듬어보게. 샘플링 횟수를 늘려보는 건 어떤가?'")
    else:
        print("\n[Conclusion] 5cm 이상 오차. 학습이 덜 되었거나 과적합 의심.")
        print("교수 평가: '아까 Loss 0.0000이라더니... 역시 쉬운 문제만 푼 거였군. Hard Negative를 추가하게!'")

if __name__ == "__main__":
    main()