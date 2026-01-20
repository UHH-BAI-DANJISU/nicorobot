import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

# Importing the dataset class from dataset.py
from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR

# ---------------------------------------------------------
# 1. Model Definition (Explicit BC)
# ---------------------------------------------------------
class ExplicitBCPolicy(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        # Vision Encoder: Pretrained ResNet18
        self.backbone = resnet18(pretrained=True)
        # Replace the final FC layer with Identity to extract the 512-dim feature vector
        self.backbone.fc = nn.Identity()

        # Action Prediction Head (MLP)
        # Input: Image features (512)
        # Output: Predicted Action (14) -> 8 Joints + 6 Pose parameters
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # Final activation is omitted to allow MSE loss to guide the output range        
        )
    
    def forward(self, x):
        # x: [Batch, 3, 64, 64]
        features = self.backbone(x) # [Batch, 512]
        pred_action = self.head(features) # [Batch, 14]
        return pred_action

# ---------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter('runs/explicit_baseline_exp1')

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

    global_step = 0 

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
            
            # Log Loss per training step
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # --- 에폭 단위 Train Loss 기록 ---
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # Validation phase
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
        
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # Save trained model weights
    torch.save(model.state_dict(), "explicit_bc_model.pth")
    
    # Close TensorBoard writer
    writer.close()
    print("\n[Done] Training Finished. Run 'tensorboard --logdir=runs' to view results.")

if __name__ == "__main__":
    main()