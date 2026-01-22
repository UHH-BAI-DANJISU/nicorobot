import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_kinematics as pk
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------
# [Setup] Project paths and custom module imports
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from dataset import NICORobotDataset, get_normalization_stats
except ImportError:
    raise ImportError("'dataset.py' must be located in the src folder.")

try:
    from model_architecture import VisionEncoder
except ImportError:
    raise ImportError("Please create 'model_architecture.py' in the src folder.")

# ---------------------------------------------------------
# 1. Kinematics Wrapper
# ---------------------------------------------------------
class NICOArmKinematics:
    def __init__(self, urdf_path, device='cpu'):
        self.device = device
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), 
            end_link_name="left_palm:11"
        ).to(device=device)
        
        self.arm_limits_min = torch.tensor([-2.182, -3.124, -1.8675, -1.745, -1.571, -0.872], device=device)
        self.arm_limits_max = torch.tensor([ 1.745,  3.142,  3.002,   1.745,  1.571,  0.0   ], device=device)

    def forward_kinematics(self, arm_joint_angles):
        tg = self.chain.forward_kinematics(arm_joint_angles)
        pos = tg.get_matrix()[:, :3, 3]
        rot_mat = tg.get_matrix()[:, :3, :3]
        euler = pk.matrix_to_euler_angles(rot_mat, "XYZ")
        return torch.cat([pos, euler], dim=1)

# ---------------------------------------------------------
# 2. Energy Model (Optimized for Memory)
# ---------------------------------------------------------
class EnergyModel(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        self.encoder = VisionEncoder() 
        
        # 6-channel input modification
        original_conv = self.encoder.features[0]
        self.encoder.features[0] = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        nn.init.kaiming_normal_(self.encoder.features[0].weight, mode='fan_out', nonlinearity='relu')

        input_dim = self.encoder.output_dim + action_dim
        
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    # Extract vision features separately to avoid redundant CNN passes
    def compute_vision_feature(self, img):
        return self.encoder(img) # [Batch, 1024]

    # Compute energy using pre-computed features and action candidates
    def score_with_feature(self, vision_feature, action):
        # vision_feature: [Batch, 1024]
        # action: [Batch, 14]
        x = torch.cat([vision_feature, action], dim=1)
        return self.energy_net(x)

    def forward(self, img, action):
        visual_emb = self.compute_vision_feature(img)
        return self.score_with_feature(visual_emb, action)

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Path Configuration
    BASE_DIR = 'data/' 
    TRAIN_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples')
    TEST_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples_test')
    URDF_PATH = os.path.join(os.path.dirname(current_dir), 'complete.urdf')

    # Training Hyperparameters
    BATCH_SIZE = 64 
    LR = 1e-4
    EPOCHS = 30
    NUM_NEGATIVES = 64 # Number of negative samples per positive sample

    # ---------------- Data Loading ----------------
    train_csv = os.path.join(TRAIN_DIR, 'samples.csv')
    if not os.path.exists(train_csv):
        print(f"[Error] Training data CSV not found: {train_csv}")
        return
        
    stats = get_normalization_stats(train_csv) 

    train_ds = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
    test_ds = NICORobotDataset(TEST_DIR, stats, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"[Info] Training Samples: {len(train_ds)}")
    print(f"[Info] Testing Samples: {len(test_ds)}")

    # ---------------- Init Model ----------------
    kinematics = NICOArmKinematics(URDF_PATH, device=device)
    model = EnergyModel(action_dim=14).to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter("runs/implicit_bc_experiment")

    stats_min = torch.tensor(stats['min'], device=device, dtype=torch.float32)
    stats_max = torch.tensor(stats['max'], device=device, dtype=torch.float32)
    stats_scale = (stats_max - stats_min) + 1e-6

    # ---------------- Training ----------------
    print("[Info] Start Training (Memory Optimized)...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)       # [B, 6, 64, 64]
            pos_actions = batch['action'].to(device) # [B, 14]
            
            curr_batch_size = images.shape[0]

            # encode image only once per batch
            vision_features = model.compute_vision_feature(images) # [B, 1024]

            # 1. Positive Energy
            pos_energy = model.score_with_feature(vision_features, pos_actions)

            # 2. Negative Sampling
            # A. Arm Joints
            rand_arm = torch.rand(curr_batch_size * NUM_NEGATIVES, 6, device=device)
            rand_arm = rand_arm * (kinematics.arm_limits_max - kinematics.arm_limits_min) + kinematics.arm_limits_min
            
            # B. Head Joints
            head_min = stats_min[6:8]
            head_max = stats_max[6:8]
            rand_head = torch.rand(curr_batch_size * NUM_NEGATIVES, 2, device=device)
            rand_head = rand_head * (head_max - head_min) + head_min
            
            # C. Combine & FK
            rand_joints_raw = torch.cat([rand_arm, rand_head], dim=1) 
            with torch.no_grad():
                rand_cart_raw = kinematics.forward_kinematics(rand_arm) 

            # D. Normalize
            neg_actions_raw = torch.cat([rand_joints_raw, rand_cart_raw], dim=1)
            neg_actions = 2 * (neg_actions_raw - stats_min) / stats_scale - 1.0
            
            # Expand vision features (copying pointers, not re-encoding images)
            # [B, 1024] -> [B * N, 1024]
            vision_features_expanded = vision_features.repeat_interleave(NUM_NEGATIVES, dim=0)
            
            # Only pass through the MLP (lightweight operation)
            neg_energy = model.score_with_feature(vision_features_expanded, neg_actions)
            neg_energy = neg_energy.view(curr_batch_size, NUM_NEGATIVES) 

            # 3. Loss
            logits = torch.cat([-pos_energy, -neg_energy], dim=1)
            labels = torch.zeros(curr_batch_size, dtype=torch.long, device=device)
            
            loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)

        avg_train_loss = train_loss_sum / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                pos_actions = batch['action'].to(device)
                curr_batch_size = images.shape[0]

                vision_features = model.compute_vision_feature(images)
                pos_energy = model.score_with_feature(vision_features, pos_actions)

                # Negative Sampling
                rand_arm = torch.rand(curr_batch_size * NUM_NEGATIVES, 6, device=device)
                rand_arm = rand_arm * (kinematics.arm_limits_max - kinematics.arm_limits_min) + kinematics.arm_limits_min
                
                head_min = stats_min[6:8]
                head_max = stats_max[6:8]
                rand_head = torch.rand(curr_batch_size * NUM_NEGATIVES, 2, device=device)
                rand_head = rand_head * (head_max - head_min) + head_min
                
                rand_joints_raw = torch.cat([rand_arm, rand_head], dim=1)
                rand_cart_raw = kinematics.forward_kinematics(rand_arm)
                neg_actions_raw = torch.cat([rand_joints_raw, rand_cart_raw], dim=1)
                neg_actions = 2 * (neg_actions_raw - stats_min) / stats_scale - 1.0
                
                vision_features_expanded = vision_features.repeat_interleave(NUM_NEGATIVES, dim=0)
                neg_energy = model.score_with_feature(vision_features_expanded, neg_actions)
                neg_energy = neg_energy.view(curr_batch_size, NUM_NEGATIVES)

                logits = torch.cat([-pos_energy, -neg_energy], dim=1)
                labels = torch.zeros(curr_batch_size, dtype=torch.long, device=device)
                
                val_loss = nn.CrossEntropyLoss()(logits, labels)
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(current_dir, "best_implicit_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved! (Val Loss: {best_val_loss:.4f})")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': avg_train_loss
        }
        ckpt_path = os.path.join(current_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, ckpt_path)

    print("[Done] Training Finished.")
    writer.close()

if __name__ == "__main__":
    main()