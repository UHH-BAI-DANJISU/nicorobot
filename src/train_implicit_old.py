import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

try:
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from energy_model import EnergyModel
except ImportError:
    # 혹시 src 폴더 밖에서 실행할 경우를 대비
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from models.energy_model import EnergyModel

# ---------------------------------------------------------
# 1. Negative Sampling (오답 노트 만들기)
# ---------------------------------------------------------
def generate_negatives(batch_size, num_negatives, action_dim, device):
    """
    정답(Expert)과 비교할 '오답(Negative)' 행동들을 무작위로 생성함.
    범위: -1 ~ 1 (Normalized Action Space)
    """
    # [B * num_negatives, action_dim]
    random_actions = torch.rand(batch_size * num_negatives, action_dim, device=device) * 2 - 1
    return random_actions

# ---------------------------------------------------------
# 2. Implicit BC 학습 루프 (Main)
# ---------------------------------------------------------
def main():
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    action_dim = 14
    num_negatives = 64
    batch_size = 32
    
    # [수정 1] 에폭 50으로 단축
    epochs = 50        
    lr = 1e-4
    
    # [수정 2] 체크포인트 파일 경로
    checkpoint_path = "latest_checkpoint.pth"

    # 1. 데이터셋 & 통계 준비
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    if not os.path.exists(csv_path):
        print(f"[Error] 데이터 파일이 없습니다: {csv_path}")
        return

    stats = get_normalization_stats(csv_path)
    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # 2. 모델 및 옵티마이저 초기화
    # (device 인자 전달 필수! energy_model.py가 수정되어 있어야 함)
    model = EnergyModel(action_dim=action_dim, stats=stats, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter('runs/implicit_experiment_final')
    
    # ---------------------------------------------------------
    # [수정 3] Resume Logic (중간 저장 불러오기)
    # ---------------------------------------------------------
    start_epoch = 0
    global_step = 0
    
    if os.path.exists(checkpoint_path):
        print(f"\n[Info] Found checkpoint: {checkpoint_path}")
        print("Loading checkpoint to resume training...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 저장된 에폭 다음부터 시작
            global_step = checkpoint['global_step']
            
            print(f" -> Resuming from Epoch {start_epoch+1}")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
            print(" -> Starting from scratch.")
    else:
        print("\n[Info] No checkpoint found. Starting from scratch.")

    print(f"\n[Info] Start Training Implicit BC (Epochs: {start_epoch+1} ~ {epochs})...")

    # 3. 학습 루프
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            pos_actions = batch['action'].to(device)
            
            # (A) Positive Energy
            pos_energy = model(images, pos_actions)
            
            # (B) Negative Energy
            neg_actions = generate_negatives(batch_size, num_negatives, action_dim, device)
            
            # 이미지 확장 (Negative 개수만큼)
            images_expanded = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
            neg_energy = model(images_expanded, neg_actions).view(batch_size, num_negatives)
            
            # (C) Loss (InfoNCE)
            logits = torch.cat([-pos_energy, -neg_energy], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            loss = criterion(logits, labels)
            
            # (D) Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- 검증 (Validation) ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                pos_actions = batch['action'].to(device)
                
                pos_energy = model(images, pos_actions)
                
                neg_actions = generate_negatives(batch_size, num_negatives, action_dim, device)
                images_expanded = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
                neg_energy = model(images_expanded, neg_actions).view(batch_size, num_negatives)
                
                logits = torch.cat([-pos_energy, -neg_energy], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                loss = criterion(logits, labels)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(test_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # ---------------------------------------------------------
        # [수정 4] 매 에폭마다 '이어하기용' 체크포인트 저장
        # ---------------------------------------------------------
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_train_loss,
        }, checkpoint_path)
        
        # (선택) 10 에폭마다 영구 보존용 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"implicit_model_epoch_{epoch+1}.pth")

    # 최종 저장
    torch.save(model.state_dict(), "implicit_bc_final.pth")
    writer.close()
    print("\n[Success] Training Complete! Checkpoints saved.")

if __name__ == "__main__":
    main()