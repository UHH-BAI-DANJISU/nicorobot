import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time # [추가] 시간 측정을 위해

# 경로 문제 방지
try:
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from energy_model import EnergyModel
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR
    from models.energy_model import EnergyModel

# ---------------------------------------------------------
# 1. Hard Negative Sampling
# ---------------------------------------------------------
def generate_negatives(pos_actions, num_negatives, action_dim, device):
    batch_size = pos_actions.shape[0]
    
    # [1] 정답 복사
    neg_actions = pos_actions.unsqueeze(1).repeat(1, num_negatives, 1)
    
    # [2] Hard Negative (50%): 정답에 노이즈 섞기
    noise = torch.randn_like(neg_actions) * 0.1 
    neg_actions = neg_actions + noise
    
    # [3] Random Negative (50%): 완전 무작위
    num_random = num_negatives // 2
    random_noise = torch.rand(batch_size, num_random, action_dim, device=device) * 2 - 1
    
    neg_actions[:, :num_random, :] = random_noise
    
    # [4] 클리핑
    neg_actions = neg_actions.view(-1, action_dim)
    return torch.clamp(neg_actions, -1.0, 1.0)

# ---------------------------------------------------------
# 2. Implicit BC 학습 루프 (Main)
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    action_dim = 14
    num_negatives = 64
    batch_size = 32
    epochs = 20        
    lr = 1e-4
    
    checkpoint_path = "latest_checkpoint.pth"
    best_model_path = "best_implicit_model.pth"

    # 1. 데이터셋 준비
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    if not os.path.exists(csv_path):
        print(f"[Error] 데이터 파일이 없습니다: {csv_path}")
        return

    stats = get_normalization_stats(csv_path)
    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    # [Tip] CPU 사용 시 num_workers가 너무 크면 오히려 느릴 수 있음. 0이나 2 추천.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # 2. 모델 초기화
    model = EnergyModel(action_dim=action_dim, stats=stats, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter('runs/implicit_experiment_best_tracking')

    # 3. Resume Logic
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"\n[Info] Found checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            print(f" -> Resuming from Epoch {start_epoch+1} (Current Best Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
            print(" -> Starting from scratch.")
    else:
        print("\n[Info] Starting FRESH training!")

    print(f"\n[Info] Start Training (Epochs: {start_epoch+1} ~ {epochs})...")
    
    # 총 배치 개수 미리 계산
    total_batches = len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            pos_actions = batch['action'].to(device)
            
            # (A) Positive Energy
            pos_energy = model(images, pos_actions)
            
            # (B) Negative Energy
            neg_actions = generate_negatives(pos_actions, num_negatives, action_dim, device)
            
            images_expanded = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
            neg_energy = model(images_expanded, neg_actions).view(batch_size, num_negatives)
            
            # (C) Loss
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
            
            # [추가됨] 10 배치마다 로그 출력 (살아있음을 알림)
            if (i + 1) % 10 == 0:
                print(f"\rEpoch [{epoch+1}/{epochs}] Step [{i+1}/{total_batches}] Loss: {loss.item():.4f}", end="")

        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        
        # 한 에폭 걸린 시간
        epoch_duration = time.time() - start_time
        print(f"\nTime per Epoch: {epoch_duration:.2f} sec")

        # --- Validation ---
        print("Running Validation...", end="")
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                pos_actions = batch['action'].to(device)
                
                pos_energy = model(images, pos_actions)
                
                neg_actions = generate_negatives(pos_actions, num_negatives, action_dim, device)
                
                images_expanded = images.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1).view(-1, 6, 64, 64)
                neg_energy = model(images_expanded, neg_actions).view(batch_size, num_negatives)
                
                logits = torch.cat([-pos_energy, -neg_energy], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                loss = criterion(logits, labels)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(test_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        print(f"\rEpoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}", end="")

        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" -> 🏆 New Best Model! ({best_val_loss:.4f})")
        else:
            print("") 

        # Checkpoint 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'loss': avg_train_loss,
        }, checkpoint_path)

    # 최종 저장
    torch.save(model.state_dict(), "implicit_bc_final.pth")
    writer.close()
    print("\n[Success] Training Complete!")

if __name__ == "__main__":
    main()