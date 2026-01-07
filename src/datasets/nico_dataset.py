# resize + tensor 변환 전부 포함

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NicoDataset(Dataset):
    """
    Observation o:
      - left RGB image (64x64)
      - right RGB image (64x64)
    Action a:
      - joint angles
      - end-effector cartesian pose
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(root_dir, "data.csv"))

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        self.joint_cols = [c for c in self.csv.columns if c.startswith("joint")]
        self.ee_cols = [c for c in self.csv.columns if c.startswith("ee")]

    def __len__(self):
        return len(self.csv) # 데이터 샘플 수 반환

    def _load_img(self, rel_path):
        img = Image.open(os.path.join(self.root_dir, rel_path)).convert("RGB")
        return self.transform(img)

        # 상대 경로 받아서 실제 파일 열고 RGB로 강제 변환, resize + tensor 변환 전부 포함

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]

        # obsercation 구성, binocular vision 구현
        left = self._load_img(row["left_image_path"])
        right = self._load_img(row["right_image_path"])
        obs = torch.cat([left, right], dim=0)  # [6,64,64]

        # action 구성
        joint = torch.tensor(row[self.joint_cols].values, dtype=torch.float32)
        ee = torch.tensor(row[self.ee_cols].values, dtype=torch.float32)
        action = torch.cat([joint, ee], dim=0) # 관절 각도와 Cartesian pose를 하나의 벡터로 결합
        # a = [a_J, a_C]

        return obs, action # 모델이 받는 최종 입력
