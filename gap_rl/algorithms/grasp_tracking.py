import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation

def qrot(q, v):
    """
    q: (B, 4)
    v: (B, 3)
    """
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qmul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w = -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1
    x = x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1
    y = -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1
    z = x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1
    return torch.stack((w, x, y, z), dim=-1)

def compute_pose_distance(pose1, pose2):
    """
    pose1: (B, 9) [x, y, z, rx1, rx2, rx3, ry1, ry2, ry3]
    pose2: (B, N, 9)
    Reuses the distance logic from pick_single.py but adapted for PyTorch 9D poses
    """
    t1 = pose1[:, :3].unsqueeze(1)
    t2 = pose2[:, :, :3]
    trans_dist = torch.norm(t1 - t2, dim=-1)
    
    # 9D pose -> 3x3 rot -> quat
    # Actually pick_single uses rotation matrix to quat. 
    # Let's compute rotation distance from matrices directly like goal_pred_rotmat_loss
    r1_x, r1_y = pose1[:, 3:6], pose1[:, 6:9]
    r1_z = torch.cross(r1_x, r1_y, dim=-1)
    r1 = torch.stack((r1_x, r1_y, r1_z), dim=-1).unsqueeze(1) # (B, 1, 3, 3)
    
    r2_x, r2_y = pose2[:, :, 3:6], pose2[:, :, 6:9]
    r2_z = torch.cross(r2_x, r2_y, dim=-1)
    r2 = torch.stack((r2_x, r2_y, r2_z), dim=-1) # (B, N, 3, 3)
    
    # For symmetry we might need rotz180 as in pick_single, but a simple trace distance is sufficient for association
    r2_rotz180 = r2.clone()
    r2_rotz180 = r2_rotz180 * torch.tensor([-1, -1, 1], device=r2.device).view(1, 1, 3, 1)
    
    # tr(R1^T R2)
    rot_dist0 = 3 - torch.diagonal(torch.einsum("bijh, bikjh -> bijk", r1, r2.transpose(-2, -1)), dim1=-2, dim2=-1).sum(dim=-1)
    rot_dist1 = 3 - torch.diagonal(torch.einsum("bijh, bikjh -> bijk", r1, r2_rotz180.transpose(-2, -1)), dim1=-2, dim2=-1).sum(dim=-1)
    
    rot_dist = torch.minimum(rot_dist0, rot_dist1)
    return trans_dist + rot_dist

class GraspTracking(nn.Module):
    def __init__(self, pose_dim=9, hidden_size=32):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_size = hidden_size
        
        # Motion predictor GRU
        self.gru = nn.GRUCell(input_size=pose_dim, hidden_size=hidden_size)
        
        # Predictors for forecasted pose and confidence
        self.pose_predictor = nn.Linear(hidden_size, pose_dim)
        self.conf_predictor = nn.Linear(hidden_size, 1)
        
        self.dist_threshold = 1.5 # Distance threshold for track-loss

    def forward(self, candidates, scores, prev_target=None, prev_hidden=None):
        """
        candidates: (B, N, pose_dim)
        scores: (B, N)
        prev_target: (B, pose_dim) or None
        prev_hidden: (B, hidden_size) or None
        
        Returns:
        tracked_pose: (B, pose_dim)
        confidence: (B, 1)
        runner_ups: (B, N-1, pose_dim)
        new_hidden: (B, hidden_size)
        """
        B, N, D = candidates.shape
        device = candidates.device
        
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.hidden_size, device=device)
            
        confidence = torch.zeros(B, 1, device=device)
        tracked_pose = torch.zeros(B, D, device=device)
        runner_ups = torch.zeros(B, N-1, D, device=device)
        
        if prev_target is None:
            # Cold-start
            best_idx = torch.argmax(scores, dim=1) # (B,)
            tracked_pose = candidates[torch.arange(B), best_idx]
            # Emit low/zero confidence for cold start
            confidence = torch.zeros(B, 1, device=device)
            
            for b in range(B):
                mask = torch.arange(N) != best_idx[b].item()
                runner_ups[b] = candidates[b, mask]
                
            new_hidden = self.gru(tracked_pose, prev_hidden)
        else:
            # Match
            dist = compute_pose_distance(prev_target, candidates) # (B, N)
            best_match_idx = torch.argmin(dist, dim=1) # (B,)
            min_dist = dist[torch.arange(B), best_match_idx] # (B,)
            
            # Motion predictor forecast
            forecast_pose = self.pose_predictor(prev_hidden)
            forecast_conf = torch.sigmoid(self.conf_predictor(prev_hidden))
            
            for b in range(B):
                if min_dist[b] > self.dist_threshold:
                    # Track loss -> fallback to motion predictor forecast
                    tracked_pose[b] = forecast_pose[b]
                    confidence[b] = forecast_conf[b] * 0.5 # lower confidence
                    
                    # Runner ups are just top candidates by score
                    best_scored_idx = torch.argmax(scores[b])
                    mask = torch.arange(N) != best_scored_idx.item()
                    runner_ups[b] = candidates[b, mask]
                else:
                    # Association succeeded
                    tracked_pose[b] = candidates[b, best_match_idx[b]]
                    confidence[b] = forecast_conf[b] # or 1.0, but let's use the predictor's conf
                    
                    mask = torch.arange(N) != best_match_idx[b].item()
                    runner_ups[b] = candidates[b, mask]
            
            new_hidden = self.gru(tracked_pose, prev_hidden)
            
        return tracked_pose, confidence, runner_ups, new_hidden
