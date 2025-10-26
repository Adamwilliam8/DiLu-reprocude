# ppo_research_template.py
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# --------------------------
# 1) Config
# --------------------------
@dataclass
class PPOConfig:
    policy_kwargs: Dict[str, Any]              # {"net_arch": [256,256]}
    learning_rate: float = 3e-4                # 0.0003
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5                # 论文常用：0.5
    value_clip_range: Optional[float] = 0.2   # None 关闭 value clipping
    device: str = "cpu"                       # or "cuda"
    tensorboard_log: Optional[str] = None
    verbose: int = 1                          # 0/1


# --------------------------
# 2) MLP 构建器
# --------------------------
def build_mlp(input_dim: int, net_arch: List[int], activation=nn.ReLU) -> nn.Sequential:
    layers = []
    last = input_dim
    for h in net_arch:
        layers += [nn.Linear(last, h), activation()]
        last = h
    return nn.Sequential(*layers)


# --------------------------
# 3) Policy / Value
# --------------------------
class DiscretePolicy(nn.Module):
    """离散动作策略：共享干路 MLP + logits 头"""
    def __init__(self, state_dim: int, action_dim: int, net_arch: List[int]):
        super().__init__()
        self.backbone = build_mlp(state_dim, net_arch)
        self.logits = nn.Linear(net_arch[-1], action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.logits(z)  # raw logits

    def dist(self, x: torch.Tensor) -> Categorical:
        logits = self.forward(x)
        return Categorical(logits=logits)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, net_arch: List[int]):
        super().__init__()
        self.backbone = build_mlp(state_dim, net_arch)
        self.v = nn.Linear(net_arch[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.v(z)  # (B,1)


# --------------------------
# 4) PPO Agent
# --------------------------
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        net_arch = cfg.policy_kwargs.get("net_arch", [256, 256])
        self.policy = DiscretePolicy(state_dim, action_dim, net_arch).to(self.device)
        self.value_fn = ValueNet(state_dim, net_arch).to(self.device)

        # 分别优化（也可共享优化器）
        self.optimizer_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.optimizer_v  = torch.optim.Adam(self.value_fn.parameters(), lr=cfg.learning_rate)

        self.writer = None
        if cfg.tensorboard_log and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=cfg.tensorboard_log)

        # rollout 缓存
        self.reset_storage(state_dim, action_dim)

        # 计步
        self.global_step = 0
        self.update_it = 0
        self.current_step = 0  # 当前缓冲区位置

    # ---------- Rollout 存储 ----------
    def reset_storage(self, state_dim: int, action_dim: int):
        n = self.cfg.n_steps
        self.obs_buf = torch.zeros((n, state_dim), dtype=torch.float32, device=self.device)
        self.actions  = torch.zeros((n, 1), dtype=torch.long, device=self.device)
        self.logprobs = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.rewards  = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dones    = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.values   = torch.zeros((n, 1), dtype=torch.float32, device=self.device)

    # ---------- 与环境交互 ----------
    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.policy.dist(obs_t)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value_fn(obs_t)
        return int(action.item()), float(logprob.item()), float(value.item())

    # ---------- 计算 GAE ----------
    @torch.no_grad()
    def compute_gae(self, last_value: float):
        rewards = self.rewards
        values  = self.values
        dones   = self.dones

        advantages = torch.zeros_like(rewards, device=self.device)
        last_adv = 0.0
        for t in reversed(range(self.cfg.n_steps)):
            if t == self.cfg.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t+1].item()
                next_non_terminal = 1.0 - dones[t+1]
            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv

        returns = advantages + values
        # 标准化 advantage（稳定关键）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # ---------- 更新 ----------
    def update(self, advantages: torch.Tensor, returns: torch.Tensor):
        cfg = self.cfg
        n = cfg.n_steps

        # 打平
        obs      = self.obs_buf
        actions  = self.actions
        old_logp = self.logprobs
        values   = self.values

        # 参考旧值用于 value clipping
        old_values = values.detach()

        # 打乱索引做 mini-batch
        idxs = torch.randperm(n, device=self.device)
        num_minibatches = math.ceil(n / cfg.batch_size)

        pi_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for epoch in range(cfg.n_epochs):
            for mb in range(num_minibatches):
                start = mb * cfg.batch_size
                end = min(start + cfg.batch_size, n)
                mb_idx = idxs[start:end]

                mb_obs      = obs[mb_idx]
                mb_actions  = actions[mb_idx].squeeze(-1)
                mb_adv      = advantages[mb_idx]
                mb_returns  = returns[mb_idx]
                mb_old_logp = old_logp[mb_idx].squeeze(-1)
                mb_old_v    = old_values[mb_idx]

                # --- Policy ---
                dist = self.policy.dist(mb_obs)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)       # π/π_old
                surr1 = ratio * mb_adv.squeeze(-1)
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv.squeeze(-1)
                pi_loss = -torch.min(surr1, surr2).mean()

                # --- Value ---
                new_v = self.value_fn(mb_obs)
                if cfg.value_clip_range is not None:
                    v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -cfg.value_clip_range, cfg.value_clip_range)
                    v_loss_unclipped = (new_v - mb_returns).pow(2)
                    v_loss_clipped   = (v_clipped - mb_returns).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (new_v - mb_returns).pow(2).mean()

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                self.optimizer_pi.zero_grad(set_to_none=True)
                self.optimizer_v.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value_fn.parameters()),
                                         cfg.max_grad_norm)
                self.optimizer_pi.step()
                self.optimizer_v.step()

                # 统计
                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float()).item()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())
                clip_fracs.append(clip_frac)

        # 日志
        self.update_it += 1
        if self.writer is not None:
            self.writer.add_scalar("loss/policy", np.mean(pi_losses), self.update_it)
            self.writer.add_scalar("loss/value",  np.mean(v_losses), self.update_it)
            self.writer.add_scalar("stats/entropy", np.mean(entropies), self.update_it)
            self.writer.add_scalar("stats/clip_frac", np.mean(clip_fracs), self.update_it)
        if self.cfg.verbose:
            print(f"[PPO] update #{self.update_it} | pi_loss={np.mean(pi_losses):.4f} "
                  f"v_loss={np.mean(v_losses):.4f} ent={np.mean(entropies):.4f} "
                  f"clip_frac={np.mean(clip_fracs):.3f}")

    def store_transition(self, obs, action, logprob, reward, done, value):
        """手动存储单步经验"""
        if self.current_step >= self.cfg.n_steps:
            raise RuntimeError("Buffer is full, call update() first!")
        
        t = self.current_step
        self.obs_buf[t] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[t, 0] = action
        self.logprobs[t, 0] = logprob
        self.rewards[t, 0] = reward  # 这里存储 r_total
        self.dones[t, 0] = float(done)
        self.values[t, 0] = value
        
        self.current_step += 1
        self.global_step += 1

    def is_ready_to_update(self):
        """检查是否收集了足够的经验"""
        return self.current_step >= self.cfg.n_steps

    def update_from_buffer(self, last_obs):
        """使用缓冲区中的经验更新策略"""
        if not self.is_ready_to_update():
            return
        
        # 计算最后一个状态的值
        with torch.no_grad():
            last_value = self.value_fn(
                torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).item()
        
        # 计算 GAE
        adv, ret = self.compute_gae(last_value)
        
        # 更新策略
        self.update(adv, ret)
        
        # 重置缓冲区
        self.current_step = 0