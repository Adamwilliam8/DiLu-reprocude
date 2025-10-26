import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ChannelAttention(nn.Module):
    """通道注意力机制，增强状态特征的重要维度"""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, D]
        w = self.fc(x)  # [B, D]
        return x * w


class AttentionDiscriminator(nn.Module):
    """
    带注意力机制的判别器，用于评估 Agent 动作与 LLM 专家动作的相似度
    输入: (state, action_onehot)
    输出: [0,1] 相似度分数 (1=完全相似, 0=不相似)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        # 状态通道注意力
        self.state_attn = ChannelAttention(state_dim)
        
        # 主判别网络
        layers = []
        last = state_dim + action_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(last, h),
                nn.LayerNorm(h),  # 添加层归一化，稳定训练
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)   # 防止过拟合
            ]
            last = h
        
        # 输出层
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: 状态张量 [B, state_dim]
            a: 动作张量 [B] (int) 或 [B, action_dim] (onehot)
        Returns:
            相似度分数 [B, 1]，范围 [0, 1]
        """
        # 动作编码为 one-hot
        if a.dim() == 1:
            a = F.one_hot(a.long(), num_classes=self._action_dim).float()
        
        # 应用状态注意力
        s = self.state_attn(s)
        
        # 拼接状态和动作
        x = torch.cat([s, a], dim=-1)
        
        # 判别器输出
        logit = self.net(x)
        return torch.sigmoid(logit)  # 输出 [0, 1] 相似度
    
    @property
    def action_dim(self):
        return self._action_dim
    
    @property
    def state_dim(self):
        return self._state_dim


class DiscriminatorBuffer:
    """判别器训练缓冲区，用于批量更新"""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.expert_buffer = []  # (s, a_expert)
        self.agent_buffer = []   # (s, a_agent)
    
    def add(self, state, action_expert, action_agent):
        """添加经验对"""
        self.expert_buffer.append((state, action_expert))
        self.agent_buffer.append((state, action_agent))
        
        # 保持缓冲区大小
        if len(self.expert_buffer) > self.capacity:
            self.expert_buffer.pop(0)
            self.agent_buffer.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样批量数据"""
        import random
        indices = random.sample(range(len(self.expert_buffer)), min(batch_size, len(self.expert_buffer)))
        
        s_ex = torch.stack([self.expert_buffer[i][0] for i in indices])
        a_ex = torch.stack([self.expert_buffer[i][1] for i in indices])
        s_ag = torch.stack([self.agent_buffer[i][0] for i in indices])
        a_ag = torch.stack([self.agent_buffer[i][1] for i in indices])
        
        return s_ex, a_ex, s_ag, a_ag
    
    def __len__(self):
        return len(self.expert_buffer)