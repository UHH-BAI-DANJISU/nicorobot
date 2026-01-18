import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

# 자네가 짠 dataset.py에서 클래스 임포트
from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR

# ---------------------------------------------------------
# 1. 모델 정의 (Explicit BC)
# ---------------------------------------------------------
class ExplicitBCPolicy(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        # 비전 인코더: ResNet18 (Pretrained)
        self.backbone = resnet18(pretrained=True)
        # 마지막 FC 레이어 교체 -> 특징 벡터(512) 추출
        self.backbone.fc = nn.Identity()

        # Action 예측 헤드 (MLP)
        # 입력: 이미지 특징 (512)
        # 출력: Action(14) -> Joint(8) + Pose(6)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # 마지막엔 Tanh를 쓰지 않음. 
            # (데이터셋에서 이미 -1~1 정규화를 했지만, MSE Loss가 알아서 맞추도록 두는 게 나음)            
        )
    
    def forward(self, x):
        # x: [Batch, 3, 64, 64]
        features = self.backbone(x) # [Batch, 512]
        pred_action = self.head(features) # [Batch, 14]
        return pred_action

# ---------------------------------------------------------
# 2. 학습 루프 (Main)
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    writer = SummaryWriter('runs/explicit_baseline_exp1')

    # 데이터셋 준비
    stats = get_normalization_stats(os.path.join(TRAIN_DIR, 'samples.csv'))
    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = ExplicitBCPolicy(action_dim=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 30
    print(f"\n[Info] Start Training with TensorBoard logging...")

    global_step = 0 # 배치가 돌 때마다 카운트

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = batch['action'].to(device)
            
            preds = model(images)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
            # --- 배치 단위 Loss 기록 ---
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # --- 에폭 단위 Train Loss 기록 ---
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- 검증 (Validation) ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                targets = batch['action'].to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(test_loader)
        
        # --- Validation Loss 기록 ---
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "explicit_bc_model.pth")
    
    # --- Writer 종료 ---
    writer.close()
    print("\n[Done] Training Finished. Run 'tensorboard --logdir=runs' to view results.")

if __name__ == "__main__":
    main()