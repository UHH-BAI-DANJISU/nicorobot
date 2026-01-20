import torch
import torch.nn as nn
import pytorch_kinematics as pk
import os

class DifferentiableFK(nn.Module):
    def __init__(self, device='cpu', urdf_path='complete.urdf', end_link_name='left_palm:11'):
        super().__init__()
        self.device = device
        
        # Locate URDF path
        if not os.path.exists(urdf_path):
            parent_path = os.path.join(os.path.dirname(__file__), '..', urdf_path)
            if os.path.exists(parent_path):
                urdf_path = parent_path
            else:
                raise FileNotFoundError(f"[DFK Error] Cannot find URDF at: {urdf_path}")

        # Build kinematic chain from URDF
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        
        self.chain = pk.build_serial_chain_from_urdf(
            urdf_data, 
            end_link_name=end_link_name
        ).to(device=device)
        
        # Define joint mapping
        self.input_joint_names = [
            'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 
            'l_elbow_y', 'l_wrist_z', 'l_wrist_x'
        ]
        self.chain_joint_names = self.chain.get_joint_parameter_names()
        self.perm_indices = [self.input_joint_names.index(name) for name in self.chain_joint_names if name in self.input_joint_names]
        
        # This aligns DFK output with the ground truth hand position
        self.register_buffer('tcp_offset', torch.tensor([-0.02256, -0.02931, -0.02383], device=device))

    def forward(self, joint_angles):
        if joint_angles.device != torch.device(self.device):
            joint_angles = joint_angles.to(self.device)
            
        # 1. Extract arm joints
        arm_joints = joint_angles[:, :6] 

        # 2. Reorder joints to match URDF chain
        if len(self.perm_indices) > 0:
            ordered_joints = arm_joints[:, self.perm_indices]
        else:
            ordered_joints = arm_joints

        # 3. Compute Forward Kinematics
        tg = self.chain.forward_kinematics(ordered_joints)
        predicted_ee_pos = tg.get_matrix()[:, :3, 3]

        # 4. Apply TCP offset correction
        # Transforms DFK (Palm) position to Ground Truth (Hand) position
        return predicted_ee_pos + self.tcp_offset