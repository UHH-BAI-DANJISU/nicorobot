import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time

# 모듈 임포트
from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
from energy_model import EnergyModel

# ---------------------------------------------------------
# 1. Hard Negative Sampling
# ---------------------------------------------------------
def generate_negatives(pos_actions, num_negatives, action_dim, device):
    batch_size = pos_actions.shape[0]
    # 정답 복사 후 노이즈 추가
    neg_actions = pos_actions.unsqueeze(1).repeat(1, num_negatives, 1)
    noise = torch.randn_like(neg_actions) * 0.3
    neg_actions = neg_actions + noise
    
    # 50%는 완전 랜덤 노이즈로 교체
    num_random = num_negatives // 2
    random_noise = torch.rand(batch_size, num_random, action_dim, device=device) * 2 - 1
    neg_actions[:, :num_random, :] = random_noise
    
    return torch.clamp(neg_actions.view(-1, action_dim), -1.0, 1.0)

# ---------------------------------------------------------
# 2. Main Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    # 설정값
    action_dim = 14
    num_negatives = 64
    batch_size = 32
    epochs = 100      
    lr = 1e-4
    
    checkpoint_path = "latest_checkpoint.pth"
    best_model_path = "best_implicit_model.pth"

    # 1. 데이터셋 로드
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # 2. 모델 및 옵티마이저
    model = EnergyModel(action_dim=action_dim, stats=stats, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('runs/implicit_experiment')

    # 3. Resume Logic (이어서 하기)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"\n[Info] Found checkpoint: {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print("\n[Info] Starting FRESH training!")

    # 4. 학습 시작
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            pos_actions = batch['action'].to(device)
            
            # (A) Positive Energy
            pos_energy = model(images, pos_actions)
            
            # (B) Negative Energy
            neg_actions = generate_negatives(pos_actions, num_negatives, action_dim, device)
            images_expanded = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
            neg_energy = model(images_expanded, neg_actions).view(batch_size, num_negatives)
            
            # (C) [수정] Loss 계산: 낮은 에너지에 높은 확률(Softmax) 부여
            logits = torch.cat([-pos_energy, -neg_energy], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # 5. Validation (검증 로직 포함)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images, pos_act = batch['image'].to(device), batch['action'].to(device)
                pos_e = model(images, pos_act)
                neg_act = generate_negatives(pos_act, num_negatives, action_dim, device)
                img_exp = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
                neg_e = model(img_exp, neg_act).view(batch_size, num_negatives)
                
                v_logits = torch.cat([-pos_e, -neg_e], dim=1)
                val_loss_sum += criterion(v_logits, labels).item()

        avg_val_loss = val_loss_sum / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")

        # Best Model 저장 및 메시지 출력
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(" -> 🏆 New Best Model!")
        else:
            print("")

        # 매 에폭 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)

    torch.save(model.state_dict(), "implicit_bc_final.pth")
    print("\n[Success] Training Complete!")

if __name__ == "__main__":
    main()