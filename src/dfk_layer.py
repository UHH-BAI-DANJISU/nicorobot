import torch
import torch.nn as nn

class DifferentiableFK(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # -------------------------------------------------------------------
        # NICO Robot Physical Parameters (from complete.urdf)
        # -------------------------------------------------------------------
        # 1. Upper Arm (어깨~팔꿈치): l_elbow_y의 origin 벡터 크기
        # sqrt((-0.034)^2 + (0.023)^2 + (-0.17)^2) ≈ 0.1748m
        self.L1 = 0.1748 
        
        # 2. Forearm (팔꿈치~손목): l_wrist_z의 origin 벡터 크기
        # sqrt(0^2 + (-0.01)^2 + (-0.125)^2) ≈ 0.1259m
        self.L2 = 0.1259
        
        # (참고) 손목에서 손끝까지의 오프셋이 필요하다면 추가 가능 (보통 5~10cm)
        self.L_HAND = 0.05 

    def compute_transformation_matrix(self, theta, d, a, alpha):
        """
        Standard Denavit-Hartenberg (DH) Matrix Calculation
        theta: z축 회전 (Joint Angle)
        d: z축 이동 (Link Offset)
        a: x축 이동 (Link Length)
        alpha: x축 회전 (Link Twist)
        """
        B = theta.shape[0]
        
        # 삼각함수 계산
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(torch.tensor(alpha, device=self.device))
        sa = torch.sin(torch.tensor(alpha, device=self.device))
        
        # 4x4 변환 행렬 배치 생성
        mat = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        
        # 1행
        mat[:, 0, 0] = ct
        mat[:, 0, 1] = -st * ca
        mat[:, 0, 2] = st * sa
        mat[:, 0, 3] = a * ct
        
        # 2행
        mat[:, 1, 0] = st
        mat[:, 1, 1] = ct * ca
        mat[:, 1, 2] = -ct * sa
        mat[:, 1, 3] = a * st
        
        # 3행
        mat[:, 2, 1] = sa
        mat[:, 2, 2] = ca
        mat[:, 2, 3] = d
        
        return mat

    def forward(self, joint_angles):
        """
        Forward Kinematics
        input: joint_angles [Batch, 8] (NICO 왼쪽 팔 기준 6개 + 헤드 2개)
               **주의: 입력값은 반드시 라디안(Radian) 단위여야 함!**
               (-1~1로 정규화된 값이 들어온다면 Denormalize 해서 넣어야 함)
        output: predicted_ee_pos [Batch, 3] (x, y, z)
        """
        # 입력 데이터가 GPU에 있는지 확인
        if joint_angles.device != torch.device(self.device):
            joint_angles = joint_angles.to(self.device)

        # NICO Left Arm Joints: 
        # 0: l_shoulder_z
        # 1: l_shoulder_y
        # 2: l_arm_x
        # 3: l_elbow_y
        # 4: l_wrist_z
        # 5: l_wrist_x
        
        q = joint_angles
        B = q.shape[0]

        # -----------------------------------------------------
        # Kinematic Chain (Simplified for NICO)
        # NICO는 3D 오프셋이 복잡하지만, 학습을 위해 주요 링크 길이(L1, L2)를 
        # 반영한 Standard DH Chain으로 근사화함.
        # -----------------------------------------------------
        
        # 1. Base -> Shoulder Z (Yaw)
        T0 = self.compute_transformation_matrix(q[:, 0], d=0.0, a=0.0, alpha=1.57)
        
        # 2. Shoulder Z -> Shoulder Y (Pitch)
        # 여기서 L1(Upper Arm) 길이만큼 이동한다고 가정
        T1 = self.compute_transformation_matrix(q[:, 1], d=0.0, a=self.L1, alpha=0.0)
        
        # 3. Shoulder Y -> Arm X (Roll)
        # (단순화를 위해 Pitch와 Roll이 같은 위치에 있다고 가정하거나 작은 오프셋 무시)
        T2 = self.compute_transformation_matrix(q[:, 2], d=0.0, a=0.0, alpha=-1.57)
        
        # 4. Arm X -> Elbow Y (Pitch)
        # 여기서 L2(Forearm) 길이만큼 이동
        T3 = self.compute_transformation_matrix(q[:, 3], d=0.0, a=self.L2, alpha=0.0)
        
        # 5. Elbow -> Wrist Z (Yaw)
        T4 = self.compute_transformation_matrix(q[:, 4], d=0.0, a=0.0, alpha=1.57)

        # 6. Wrist Z -> Wrist X (Roll) -> Hand Tip
        T5 = self.compute_transformation_matrix(q[:, 5], d=0.0, a=self.L_HAND, alpha=0.0)

        # -----------------------------------------------------
        # 행렬 곱셈 (Chain Multiplication)
        # Base에서 시작해 손끝까지 변환 행렬을 누적
        # -----------------------------------------------------
        T_final = T0.matmul(T1).matmul(T2).matmul(T3).matmul(T4).matmul(T5)
        
        # 최종 위치 추출 (Translation Vector)
        # [Batch, 0:3, 3] -> x, y, z 좌표
        predicted_ee_pos = T_final[:, :3, 3]
        
        return predicted_ee_pos

# --- 간단 테스트 코드 ---
if __name__ == "__main__":
    # GPU 사용 가능하면 GPU로, 아니면 CPU로
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dfk = DifferentiableFK(device=device)
    
    # 더미 데이터 (배치 크기 2, 관절 8개)
    # 0도(0.0)일 때 팔을 쭉 뻗은 상태라고 가정
    dummy_joints = torch.zeros(2, 8).to(device)
    
    # Forward Pass
    pred_pos = dfk(dummy_joints)
    
    print(f"Device: {device}")
    print(f"L1(Upper): {dfk.L1}m, L2(Forearm): {dfk.L2}m")
    print("Joint Angles (Rad):", dummy_joints[0, :6])
    print("Predicted Hand Pos (x, y, z):", pred_pos[0])
    
    # 미분 가능 여부 확인 (True여야 학습 가능)
    print("Requires Grad?", pred_pos.requires_grad)