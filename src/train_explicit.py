import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

# Custom dataset modules
from dataset import NICORobotDataset, get_normalization_stats, TRAIN_DIR, TEST_DIR

# ---------------------------------------------------------
# 1. Model Definition (Explicit BC with 6-channel support)
# ---------------------------------------------------------
class ExplicitBCPolicy(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        # Load ResNet18 with default ImageNet weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first layer to accept 6-channel stereo input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Replace the fully connected layer with Identity to get features
        self.backbone.fc = nn.Identity()

        # Action Prediction Head (MLP)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        # x: [Batch, 6, 64, 64]
        features = self.backbone(x) # [Batch, 512]
        pred_action = self.head(features) # [Batch, 14]
        return pred_action

# ---------------------------------------------------------
# 2. Main Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Initialize TensorBoard
    writer = SummaryWriter('runs/explicit_baseline')

    # Data Preparation
    csv_path = os.path.join(TRAIN_DIR, 'samples.csv')
    stats = get_normalization_stats(csv_path)
    
    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model, Optimizer, and Loss
    model = ExplicitBCPolicy(action_dim=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 30
    global_step = 0 

    print(f"\n[Info] Start Training (Explicit BC Baseline)...")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device) # Shape: [B, 6, 64, 64]
            targets = batch['action'].to(device)
            
            preds = model(images)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = train_loss_sum / len(train_loader)
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
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model
    save_path = "explicit_bc_model.pth"
    torch.save(model.state_dict(), save_path)
    
    writer.close()
    print(f"\n[Done] Training Finished. Model saved to {save_path}")

if __name__ == "__main__":
    main()