import torch
import torch.nn as nn
import pytorch_kinematics as pk
import os

class DifferentiableFK(nn.Module):
    import torch
import torch.nn as nn
import pytorch_kinematics as pk
import os

class DifferentiableFK(nn.Module):
    def __init__(self, device='cpu', urdf_path='complete.urdf', end_link_name='left_palm:11'):
        super().__init__()
        self.device = device
        
        # URDF 경로 찾기
        if not os.path.exists(urdf_path):
            parent_path = os.path.join(os.path.dirname(__file__), '..', urdf_path)
            if os.path.exists(parent_path):
                urdf_path = parent_path
            else:
                raise FileNotFoundError(f"[DFK Error] Cannot find URDF at: {urdf_path}")

        # 체인 생성
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        
        self.chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            end_link_name=end_link_name
        ).to(device=device)
        
        # 관절 매핑 (왼팔 6개)
        self.input_joint_names = [
            'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 
            'l_elbow_y', 'l_wrist_z', 'l_wrist_x'
        ]
        self.chain_joint_names = self.chain.get_joint_parameter_names()
        self.perm_indices = [self.input_joint_names.index(name) for name in self.chain_joint_names if name in self.input_joint_names]

    def forward(self, joint_angles):
        if joint_angles.device != torch.device(self.device):
            joint_angles = joint_angles.to(self.device)
            
        # 1. 왼팔 6개 관절만 사용
        arm_joints = joint_angles[:, :6] 

        # 2. 순서 재정렬
        if len(self.perm_indices) > 0:
            ordered_joints = arm_joints[:, self.perm_indices]
        else:
            ordered_joints = arm_joints

        # 3. FK 계산
        tg = self.chain.forward_kinematics(ordered_joints)
        predicted_ee_pos = tg.get_matrix()[:, :3, 3]

        # [수정] 중복 오프셋 삭제됨!
        # pytorch_kinematics가 이미 URDF의 origin을 처리합니다.
        
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