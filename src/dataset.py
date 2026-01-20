import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. Configuration and Constants
# ---------------------------------------------------------
BASE_DIR = 'data/'
TRAIN_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples')
TEST_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples_test')

# Unified Action Space (14 DoF: 8 Joints + 3 Pos + 3 Rot)
JOINT_COLS = [
    'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x',
    'head_z', 'head_y'  # 8 DoF
]
POS_COLS = ['hand_pos_x', 'hand_pos_y', 'hand_pos_z'] # 3 DoF
ROT_COLS = ['hand_euler_x', 'hand_euler_y', 'hand_euler_z'] # 3 DoF

ACTION_COLS = JOINT_COLS + POS_COLS + ROT_COLS

# ---------------------------------------------------------
# 2. Statistics for Normalization
# ---------------------------------------------------------
def get_normalization_stats(csv_path):
    print(f"[Info] Calculating stats from: {csv_path}")
    df = pd.read_csv(csv_path)

    actions = df[ACTION_COLS].values # shape: (N, 14)

    stats = {
        'min': actions.min(axis=0),
        'max': actions.max(axis=0),
    }

    print(f"[Info] Stats calculated. Action Dim: {actions.shape[1]}")
    return stats

# ---------------------------------------------------------
# 3. Dataset Class
# ---------------------------------------------------------
class NICORobotDataset(Dataset):
    def __init__(self, data_dir, stats, is_train=True, image_size=64):
        self.data_dir = data_dir
        self.stats = stats

        csv_path = os.path.join(data_dir, 'samples.csv')
        full_df = pd.read_csv(csv_path)
        
        # Validate image integrity (Both eyes)
        valid_indices = []
        print(f"[Info] Checking image integrity (Full Load Check) for {len(full_df)} samples...")
        
        broken_count = 0
        for idx, row in full_df.iterrows():
            l_name = row['image_left']
            r_name = row['image_right']
            
            l_path = os.path.join(self.data_dir, 'left_eye', os.path.basename(l_name))
            r_path = os.path.join(self.data_dir, 'right_eye', os.path.basename(r_name))
            
            # 1. Check if the files exist
            if not os.path.exists(l_path) or not os.path.exists(r_path):
                broken_count += 1
                if broken_count <= 5:
                    print(f"Missing file: {l_name} or {r_name}")
                continue

            # 2. Image loading test
            try:
                with Image.open(l_path) as img:
                    img.convert('RGB')
                with Image.open(r_path) as img:
                    img.convert('RGB')
                valid_indices.append(idx)
            except Exception as e:
                broken_count += 1
                if broken_count <= 10:
                    print(f"Corrupt file pair: {l_name}, {r_name} | Error: {e}")
        
        # Keep only valid indices
        self.df = full_df.iloc[valid_indices].reset_index(drop=True)
        
        removed_count = len(full_df) - len(self.df)
        print(f"Summary: {len(self.df)} valid images kept. {removed_count} images removed.")
        
        if len(self.df) == 0:
            raise RuntimeError(f"No valid images found in {data_dir}")
       
       # Image Augmentation/Preprocessing
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # [추가] 조명 변화 대응
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # For Min-Max Scaling (numpy -> tensor)
        self.min_val = torch.tensor(stats['min'], dtype=torch.float32)
        self.max_val = torch.tensor(stats['max'], dtype=torch.float32)
        self.scale = (self.max_val - self.min_val) + 1e-6

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load stereo images
        l_name = row['image_left']
        r_name = row['image_right']
        
        l_path = os.path.join(self.data_dir, 'left_eye', os.path.basename(l_name))
        r_path = os.path.join(self.data_dir, 'right_eye', os.path.basename(r_name))

        try:
            image_l = Image.open(l_path).convert('RGB')
            image_r = Image.open(r_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image pair: {l_path}, {r_path}")
            image_l = Image.new('RGB', (64, 64))
            image_r = Image.new('RGB', (64, 64))
        
        # Concatenate along channel dimension (6, H, W)
        l_tensor = self.transform(image_l)
        r_tensor = self.transform(image_r)
        image_tensor = torch.cat([l_tensor, r_tensor], dim=0)

        # Action Normalization [-1, 1]
        raw_action = row[ACTION_COLS].values.astype(np.float32)
        action_tensor = torch.tensor(raw_action, dtype=torch.float32)

        # Min-Max Normalization to [-1, 1]
        normalized_action = 2 * (action_tensor - self.min_val) / self.scale - 1.0
        
        # Proprioception: First 8 DoF (Joints)
        proprioception = normalized_action[:8]

        return {
            'image': image_tensor,       # (6, 64, 64) 
            'proprio': proprioception,   # (8,)
            'action': normalized_action, # (14,)
            'raw_action': action_tensor  # (14,)
        }

# ---------------------------------------------------------
# 4. Verification
# ---------------------------------------------------------
if __name__ == "__main__":
    train_csv = os.path.join(TRAIN_DIR, 'samples.csv')

    if not os.path.exists(train_csv):
        print(f"Error: File not found. Please check the path: {train_csv}")
    else:
        stats = get_normalization_stats(train_csv)
        train_dataset = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        print("\n[Info] Data Loading Test:")
        sample = next(iter(train_loader))
        
        print(f"Image Shape: {sample['image'].shape}")   # 32, 6, 64, 64]
        print(f"Action Shape: {sample['action'].shape}") # [32, 14]
        print(f"Proprio Shape: {sample['proprio'].shape}") # [32, 8]
        
        print("Data Loading Successful.")