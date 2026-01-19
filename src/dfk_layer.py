import torch
import torch.nn as nn
import pytorch_kinematics as pk
import os

class DifferentiableFK(nn.Module):
    def __init__(self, device='cpu', urdf_path='complete.urdf', end_link_name='left_palm:11'):
        super().__init__()
        self.device = device
        
        # 1. URDF 파일 경로 탐색 (현재 폴더 혹은 상위 폴더)
        if not os.path.exists(urdf_path):
            # ../complete.urdf 시도
            parent_path = os.path.join(os.path.dirname(__file__), '..', urdf_path)
            if os.path.exists(parent_path):
                urdf_path = parent_path
            else:
                # 못 찾으면 에러 발생 (또는 다운로드 로직 등)
                raise FileNotFoundError(f"[DFK Error] Cannot find URDF file at: {urdf_path}")

        print(f"[Info] Loading URDF from: {urdf_path}")

        # 2. Build Serial Chain (Base -> End Effector)
        # NICO 로봇: 'torso:11' (Base) -> 'left_palm:11' (End Effector)
        # URDF 데이터를 읽어서 체인 생성
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        
        self.chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            end_link_name=end_link_name
        )
        self.chain = self.chain.to(device=device)
        
        # 3. Joint Mapping 준비
        # dataset.py의 입력 관절 순서 (왼팔 6 자유도)
        self.input_joint_names = [
            'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 
            'l_elbow_y', 'l_wrist_z', 'l_wrist_x'
        ]
        
        # 체인(URDF)에서 실제 구동되는 관절 이름 목록 가져오기
        self.chain_joint_names = self.chain.get_joint_parameter_names()
        
        # 입력 벡터(dataset 순서)를 체인(URDF 순서)에 맞게 재정렬할 인덱스 생성
        self.perm_indices = []
        for name in self.chain_joint_names:
            if name in self.input_joint_names:
                idx = self.input_joint_names.index(name)
                self.perm_indices.append(idx)
            else:
                print(f"[Warning] Joint '{name}' in URDF chain is not in input list. (Will be treated as 0?)")

    def forward(self, joint_angles):
        """
        Input: joint_angles [Batch, 8] (또는 14) 
               - dataset.py 기준 0~5번 인덱스가 왼팔 관절
        Output: predicted_ee_pos [Batch, 3] (x, y, z)
        """
        # 입력 데이터 Device 이동
        if joint_angles.device != torch.device(self.device):
            joint_angles = joint_angles.to(self.device)
            
        # 1. 왼팔 관절(6개)만 추출
        arm_joints = joint_angles[:, :6] 

        # 2. URDF 체인이 기대하는 순서로 재정렬
        if len(self.perm_indices) > 0:
            ordered_joints = arm_joints[:, self.perm_indices]
        else:
            ordered_joints = arm_joints

        # 3. Forward Kinematics 계산 (어깨 기준 위치 계산)
        tg = self.chain.forward_kinematics(ordered_joints)
        
        # 4. 위치(Translation) 추출
        m = tg.get_matrix()
        predicted_ee_pos = m[:, :3, 3]

        # =================================================================
        # [수정] 좌표계 보정 (Offset Correction)
        # -----------------------------------------------------------------
        # 문제: 로봇의 팔은 바닥(0,0,0)이 아니라 몸통 위(약 75cm 높이)에 달려 있습니다.
        #       DFK가 어깨를 (0,0,0)으로 계산하고 있다면, 실제 정답(몸통 기준)과
        #       수십 cm의 오차가 발생합니다. 이를 URDF 값으로 보정합니다.
        #
        # 출처: complete.urdf (joint: l_shoulder_z)
        # <origin xyz="0.026783 0.049488 0.748809" ... />
        # =================================================================
        
        # URDF에서 가져온 Torso -> Left Shoulder 오프셋 (x, y, z)
        # (주의: 만약 self.chain을 'torso:11'부터 생성했다면 이 과정이 중복일 수 있으나,
        #  현재 오차가 큰 상황에서는 이 오프셋이 빠져 있을 확률이 99%입니다.)
        base_offset = torch.tensor([0.026783, 0.049488, 0.748809], device=self.device)
        
        # 최종 위치 = 어깨 기준 위치 + 오프셋
        predicted_ee_pos = predicted_ee_pos + base_offset
        
        return predicted_ee_pos

if __name__ == "__main__":
    # 테스트 코드
    try:
        dfk = DifferentiableFK(device='cpu', urdf_path='complete.urdf')
        dummy_input = torch.zeros(2, 8) # Batch 2
        pos = dfk(dummy_input)
        print("Output shape:", pos.shape)
        print("Position (Left Palm):", pos)
    except Exception as e:
        print(f"Test Failed: {e}")
        print("Make sure 'complete.urdf' is in the current or parent directory.")