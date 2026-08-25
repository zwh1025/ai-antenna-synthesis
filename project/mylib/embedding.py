"""词嵌入与解嵌入模块。

提供两套实现：
  - numpy 版（map_num_to_vec_np / map_vec_to_num_np）：
    线性插值，不可导，用于数据预处理（与原代码行为一致）。
  - torch 版（NumEmbedding / NumDeEmbedding）：
    softmax 软化，可微，用于网络层或物理损失回传。

原代码的 map_num_to_vec 在分割点处不可导（线性插值切换）。
本模块用 softmax(-beta * (x - dp)^2) 替代，保证全局可导。
beta 自动根据分割点间距选取，使相邻分割点权重约为 exp(-1)。
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================
#  numpy 版（数据预处理用）
# ============================================================

def map_num_to_vec_np(x, divide_points, dtype=np.float32, padding=True):
    """将实数映射为概率分布向量（线性插值，不可导）。

    与原 CreateDataset.map_num_to_vec 行为一致。
    """
    x = float(x)
    dp = np.asarray(divide_points, dtype=np.float64)
    vec = np.zeros(len(dp), dtype=dtype)

    if x <= dp[0]:
        vec[0] = 1.0
    elif x >= dp[-1]:
        vec[-1] = 1.0
    else:
        left = np.where(x - dp >= 0)[0][-1]
        right = np.where(dp - x >= 0)[0][0]
        if left == right:
            vec[left] = 1.0
        else:
            rl = dp[right] - dp[left]
            vec[left] = (dp[right] - x) / rl
            vec[right] = (x - dp[left]) / rl

    if padding:
        vec = np.hstack([vec, np.array([0.0], dtype=dtype)])
    return vec


def map_vec_to_num_np(vec, divide_points):
    """将概率分布向量映射回实数（加权平均）。

    与原 CreateDataset.map_vec_to_num 行为一致。
    输入 vec 可含 separator 位（自动截断）。
    """
    dp = np.asarray(divide_points, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    if len(vec) == len(dp) + 1:
        vec = vec[:-1]
    return float(np.sum(vec * dp))


# ============================================================
#  torch 版（可微，网络层用）
# ============================================================

class NumEmbedding(nn.Module):
    """可微词嵌入层。

    将实数 x 映射为 (n+1) 维概率分布向量（前 n 维为 softmax 权重，
    最后一维为 separator 位，默认 0）。

    权重计算：w_i = softmax(-beta * (x - dp_i)^2)
    beta 自动选取使相邻分割点权重约为 exp(-1)。
    """

    def __init__(self, divide_points, beta=None):
        super().__init__()
        dp = np.asarray(divide_points, dtype=np.float32)
        self.register_buffer('divide_points', torch.as_tensor(dp))
        if beta is None:
            spacing = (dp[-1] - dp[0]) / (len(dp) - 1)
            beta = 1.0 / (spacing ** 2)
        self.beta = float(beta)

    def forward(self, x):
        """x: (...) 实数 → (..., n+1) 词向量。"""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().unsqueeze(-1)
        diff = x - self.divide_points
        weights = torch.softmax(-self.beta * diff ** 2, dim=-1)
        zeros = torch.zeros(weights.shape[:-1] + (1,),
                            dtype=weights.dtype, device=weights.device)
        return torch.cat([weights, zeros], dim=-1)

    def separator(self, batch_shape=()):
        """返回 separator 向量 [0, ..., 0, 1]。"""
        n = len(self.divide_points)
        vec = torch.zeros(batch_shape + (n + 1,), dtype=torch.float32)
        vec[..., -1] = 1.0
        return vec


class NumDeEmbedding(nn.Module):
    """解嵌入层。

    将 (n+1) 维概率分布向量映射回实数。
    取前 n 维归一化后与分割点加权平均。
    """

    def __init__(self, divide_points):
        super().__init__()
        dp = np.asarray(divide_points, dtype=np.float32)
        self.register_buffer('divide_points', torch.as_tensor(dp))

    def forward(self, vec):
        """vec: (..., n+1) → (...,) 实数。"""
        if not isinstance(vec, torch.Tensor):
            vec = torch.as_tensor(vec, dtype=torch.float32)
        weights = vec[..., :-1]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        return torch.sum(weights * self.divide_points, dim=-1)
