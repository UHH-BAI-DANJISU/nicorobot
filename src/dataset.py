import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 및 상수 정의
# ---------------------------------------------------------
# 데이터셋 경로
BASE_DIR = 'data/'
TRAIN_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples')
TEST_DIR = os.path.join(BASE_DIR, 'real_evo_ik_samples_test')

# CSV에서 사용할 컬럼 정의 (논문의 Unified Action Space)
JOINT_COLS = [
    'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x',
    'head_z', 'head_y'  # 8 DoF
]
POS_COLS = ['hand_pos_x', 'hand_pos_y', 'hand_pos_z'] # 3 DoF
ROT_COLS = ['hand_euler_x', 'hand_euler_y', 'hand_euler_z'] # 3 DoF

ACTION_COLS = JOINT_COLS + POS_COLS + ROT_COLS # 총 14차원 벡터

# ---------------------------------------------------------
# 2. 통계 산출 함수 (Min-Max Normalization용)
# ---------------------------------------------------------
def get_normalization_stats(csv_path):
    """
    학습 데이터 CSV를 읽어서 각 행동 차원(Column)별 Min, Max 값 반환
    이 값은 나중에 [-1, 1] 정규화에 사용
    """
    print(f"[Info] Calculating stats from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 필요 컬럼만 추출
    actions = df[ACTION_COLS].values # shape: (N, 14)

    # Min, Max 계산
    stats = {
        'min': actions.min(axis=0),
        'max': actions.max(axis=0),
    }

    print(f"[Info] Stats calculated. Action Dim: {actions.shape[1]}")
    return stats

# ---------------------------------------------------------
# 3. 데이터셋 클래스 정의
# ---------------------------------------------------------
class NICORobotDataset(Dataset):
    def __init__(self, data_dir, stats, is_train=True, image_size=64):
        """
        Args:
            data_dir (str): 데이터셋 폴더 경로 (예: real_evo_ik_samples)
            stats (dict): get_normalization_stats에서 구한 {'min': .., 'max': ..}
            is_train (bool): 학습 모드 여부 (Augmentation 적용 유무 등)
            image_size (int): 이미지 리사이징 크기 (64, 64)
        """
        self.data_dir = data_dir
        self.stats = stats

        # 1. CSV 로드
        csv_path = os.path.join(data_dir, 'samples.csv')
        full_df = pd.read_csv(csv_path)
        
        # -------------------------------------------------------------------
        # [최종 수정] __init__ 내부: 안전한 이미지 검증 로직 (양쪽 눈 확인)
        # -------------------------------------------------------------------
        valid_indices = []
        print(f"[Info] Checking image integrity (Full Load Check) for {len(full_df)} samples...")
        
        broken_count = 0
        for idx, row in full_df.iterrows():
            # 왼쪽, 오른쪽 이미지 경로 생성
            # CSV에는 'left_eye/...', 'right_eye/...' 형태로 저장되어 있다고 가정
            l_name = row['image_left']
            r_name = row['image_right']
            
            l_path = os.path.join(self.data_dir, 'left_eye', os.path.basename(l_name))
            r_path = os.path.join(self.data_dir, 'right_eye', os.path.basename(r_name))
            
            # 1. 파일 존재 여부 확인 (둘 다 있어야 함)
            if not os.path.exists(l_path) or not os.path.exists(r_path):
                broken_count += 1
                if broken_count <= 5:
                    print(f"Missing file: {l_name} or {r_name}")
                continue

            # 2. 이미지 로딩 테스트
            try:
                # 둘 다 열어보고 문제없으면 통과
                with Image.open(l_path) as img:
                    img.convert('RGB')
                with Image.open(r_path) as img:
                    img.convert('RGB')
                valid_indices.append(idx)
            except Exception as e:
                broken_count += 1
                if broken_count <= 10:
                    print(f"Corrupt file pair: {l_name}, {r_name} | Error: {e}")
        
        # 유효한 인덱스만 남김
        self.df = full_df.iloc[valid_indices].reset_index(drop=True)
        
        # 결과 리포트
        removed_count = len(full_df) - len(self.df)
        print(f"Summary: {len(self.df)} valid images kept. {removed_count} images removed.")
        
        if len(self.df) == 0:
            raise RuntimeError("CRITICAL: 모든 이미지가 제거되었습니다! 경로 설정을 다시 확인하세요.")
        # -------------------------------------------------------------------
        
        # 이미지 전처리 파이프라인
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

        # Min-Max Scaling을 위한 준비 (numpy -> tensor)
        self.min_val = torch.tensor(stats['min'], dtype=torch.float32)
        self.max_val = torch.tensor(stats['max'], dtype=torch.float32)
        # 분모가 0이 되는 것을 방지하기 위해 아주 작은 값(epsilon) 추가
        self.scale = (self.max_val - self.min_val) + 1e-6

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. 이미지 로드 (왼쪽 + 오른쪽) ---
        l_name = row['image_left']
        r_name = row['image_right']
        
        l_path = os.path.join(self.data_dir, 'left_eye', os.path.basename(l_name))
        r_path = os.path.join(self.data_dir, 'right_eye', os.path.basename(r_name))

        try:
            image_l = Image.open(l_path).convert('RGB')
            image_r = Image.open(r_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image pair: {l_path}, {r_path}")
            # 에러 발생 시 0으로 채운 더미 이미지 반환
            image_l = Image.new('RGB', (64, 64))
            image_r = Image.new('RGB', (64, 64))
        
        # 전처리 적용 (3, 64, 64)
        l_tensor = self.transform(image_l)
        r_tensor = self.transform(image_r)
        
        # [핵심 수정] 채널 방향 결합: (3, H, W) + (3, H, W) -> (6, H, W)
        image_tensor = torch.cat([l_tensor, r_tensor], dim=0)

        # --- 2. 행동(Action) 벡터 추출 및 정규화 ---
        raw_action = row[ACTION_COLS].values.astype(np.float32)
        action_tensor = torch.tensor(raw_action, dtype=torch.float32)

        # Min-Max Normalization to [-1, 1]
        normalized_action = 2 * (action_tensor - self.min_val) / self.scale - 1.0
        
        # Proprioception: 현재 로봇의 관절 상태 (Action의 앞부분 8개가 Joint임)
        proprioception = normalized_action[:8]

        return {
            'image': image_tensor,       # (6, 64, 64) -> 모델 입력 크기와 일치!
            'proprio': proprioception,   # (8,)
            'action': normalized_action, # (14,)
            'raw_action': action_tensor  # (14,)
        }

# ---------------------------------------------------------
# 4. 실행 및 검증 코드 (메인)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 학습 데이터 통계 산출 (Train Set 기준!)
    train_csv = os.path.join(TRAIN_DIR, 'samples.csv')

    if not os.path.exists(train_csv):
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인하세요: {train_csv}")
    else:
        stats = get_normalization_stats(train_csv)

        # 2. 데이터셋 인스턴스 생성
        train_dataset = NICORobotDataset(TRAIN_DIR, stats, is_train=True)
        
        # 3. 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 4. 샘플 데이터 확인
        print("\n[Info] Data Loading Test:")
        sample = next(iter(train_loader))
        
        print(f"Image Shape: {sample['image'].shape}")   # 결과 확인: [32, 6, 64, 64] 여야 함!
        print(f"Action Shape: {sample['action'].shape}") # [32, 14]
        print(f"Proprio Shape: {sample['proprio'].shape}") # [32, 8]
        
        print("\n✅ 수정 완료. 이제 6채널 이미지가 정상적으로 공급됩니다.")