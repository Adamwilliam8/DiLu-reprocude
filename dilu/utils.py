import os
import yaml
import random
import numpy as np
import torch

def imitation_reward(disc, state, action_expert, action_agent, eps_smooth=1e-8):
    """
    计算模仿奖励：R_imit = -log(1 - D(s, a_agent) + ε)
    
    Args:
        disc: 判别器模型
        state: 当前状态 (numpy array)
        action_expert: LLM 专家动作 (int)
        action_agent: RL 智能体动作 (int)
        eps_smooth: 平滑因子，防止 log(0)
    
    Returns:
        模仿奖励标量
    """
    with torch.no_grad():
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a_tensor = torch.tensor([action_agent], dtype=torch.int64)
        
        # 判别器评分 (相似度)
        d_score = disc(s_tensor, a_tensor).item()
        
        # 计算奖励：越不相似（d_score越小），惩罚越大
        r_imit = -torch.log(torch.tensor(1.0 - d_score + eps_smooth)).item()
    
    return r_imit


def linear_eps(current_step, total_steps, start_eps=1.0, end_eps=0.1):
    """线性衰减 epsilon（探索率）"""
    return start_eps - (start_eps - end_eps) * min(current_step / total_steps, 1.0)