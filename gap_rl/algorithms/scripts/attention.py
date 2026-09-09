import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenSelfAttention(nn.Module):
    def __init__(self, d_attn: int = 384, use_attn_lstm: bool = False):
        super().__init__()
        self.use_attn_lstm = use_attn_lstm
        self.d_attn = d_attn
        
        if self.use_attn_lstm:
            # Linear projections for each of the 5 tokens
            self.proj_pn = nn.Linear(256, d_attn)
            self.proj_tcp = nn.Linear(6, d_attn)
            self.proj_gripper = nn.Linear(2, d_attn)
            self.proj_action = nn.Linear(7, d_attn)
            self.proj_grasp = nn.Linear(5, d_attn)
            
            # Q, K, V projections for the self-attention
            # The attention operates on d_attn size embeddings
            self.W_q = nn.Linear(d_attn, d_attn)
            self.W_k = nn.Linear(d_attn, d_attn)
            self.W_v = nn.Linear(d_attn, d_attn)

    def forward(self, obs_t, features_extractor_out):
        """
        obs_t: Dictionary of tensors for the current frame
        features_extractor_out: (batch, features_dim) output of the original features extractor
        """
        if not self.use_attn_lstm:
            return features_extractor_out
            
        # Extract components
        # We assume the last 256 dims of features_extractor_out are pn_feature
        pn_feature = features_extractor_out[:, -256:] # (batch, 256)
        tcp_pose = obs_t["tcp_pose"] # (batch, 6)
        gripper_pos = obs_t["gripper_pos"] # (batch, 2)
        action = obs_t["action"] # (batch, 7)
        grasp_exist = obs_t["grasp_exist"] # (batch, 5)
        
        # Project each component to d_attn
        t1 = self.proj_pn(pn_feature) # (batch, d_attn)
        t2 = self.proj_tcp(tcp_pose)
        t3 = self.proj_gripper(gripper_pos)
        t4 = self.proj_action(action)
        t5 = self.proj_grasp(grasp_exist)
        
        # Stack into (batch, n_tokens, d_attn)
        tokens = torch.stack([t1, t2, t3, t4, t5], dim=1) # (batch, 5, d_attn)
        
        # Self-Attention
        Q = self.W_q(tokens) # (batch, 5, d_attn)
        K = self.W_k(tokens) # (batch, 5, d_attn)
        V = self.W_v(tokens) # (batch, 5, d_attn)
        
        # Scaled dot-product: softmax(Q * K^T / sqrt(d_attn)) * V
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_attn ** 0.5) # (batch, 5, 5)
        attn_weights = F.softmax(scores, dim=-1)
        attended_tokens = torch.bmm(attn_weights, V) # (batch, 5, d_attn)
        
        # Flatten back into single vector
        # (batch, 5 * d_attn)
        flattened = attended_tokens.view(attended_tokens.size(0), -1)
        return flattened

