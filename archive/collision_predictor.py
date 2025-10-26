## 要换成LLM-based的接口

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class CollisionPredictor:
    """默认用MLP进行二分类，返回碰撞概率。可以替换为LLM-based接口。"""
    def __init__(self, in_dim: int, hidden_sizes: List[int], device="cpu"):
        self.model = MLPBinary(in_dim, hidden_sizes).to(device)
        self.device = device

    @torch.no_grad()
    def predict_prob(self, state_vec, action_id: int):
        # 简单拼接 state 和动作 onehot
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.zeros((1, 5), device=self.device); a[0, action_id] = 1.0
        x = torch.cat([s, a], dim=-1)
        p = self.model(x)
        return float(p.item())

    def train_step(self, batch, optim):
        s, a, y = batch  # y in {0,1}
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        a_oh = torch.nn.functional.one_hot(a, num_classes=5).float()
        x = torch.cat([s, a_oh], dim=-1)
        p = self.model(x)
        loss = F.binary_cross_entropy(p, y)
        optim.zero_grad(); loss.backward(); optim.step()
        return float(loss.item())
