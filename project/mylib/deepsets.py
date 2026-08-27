"""DeepSets 排列等变网络：曲面阵列坐标→权值 AI 综合。

架构（排列等变 DeepSets）：
  1. Phi：逐阵元 MLP 编码器
     输入 (batch, N, input_dim)：
       (x, y, z, w_baseline_re, w_baseline_im, u0, v0, w0, sll_norm)
     输出 (batch, N, hidden_dim)
  2. 池化：mean + max over N → (batch, hidden_dim*2)
  3. Rho：全局 MLP → (batch, hidden_dim)
  4. 全局特征回传：concat(element_features, global_features)
  5. Output：逐阵元 MLP → (batch, N, 2) 即 (Δw_re, Δw_im)
  6. 最终权值：W = W_baseline + ΔW_AI

排列等变性：打乱输入阵元顺序 → 输出对应打乱 → 方向图不变。
"""

import torch
import torch.nn as nn


class DeepSetsModel(nn.Module):
    """排列等变 DeepSets 网络。

    Args:
        input_dim:  逐阵元输入特征维度（默认 9）
        hidden_dim: 隐藏层维度（默认 128）
        output_dim: 逐阵元输出维度（默认 2：Δw_re, Δw_im）
        n_phi:      Phi 网络的 MLP 层数
        n_rho:      Rho 网络的 MLP 层数
        n_output:   Output 网络的 MLP 层数
    """

    def __init__(self, input_dim=9, hidden_dim=128, output_dim=2,
                 n_phi=3, n_rho=2, n_output=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.phi = self._build_mlp(input_dim, hidden_dim, n_phi)
        self.rho = self._build_mlp(hidden_dim * 2, hidden_dim, n_rho)
        self.output_net = self._build_mlp(
            hidden_dim + hidden_dim, hidden_dim, n_output, out_dim=output_dim)

    @staticmethod
    def _build_mlp(in_dim, hidden_dim, n_layers, out_dim=None):
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim if out_dim is not None else hidden_dim))
        if out_dim is None:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。

        Args:
            x: (batch, N, input_dim) 逐阵元输入特征

        Returns:
            delta_w: (batch, N, output_dim) 逐阵元权值残差
        """
        h = self.phi(x)                              # (B, N, H)
        mean_pool = h.mean(dim=1)                    # (B, H)
        max_pool = h.max(dim=1)[0]                   # (B, H)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2H)
        global_feat = self.rho(pooled)               # (B, H)
        global_expanded = global_feat.unsqueeze(1).expand(
            -1, h.size(1), -1)                       # (B, N, H)
        combined = torch.cat([h, global_expanded], dim=-1)  # (B, N, 2H)
        delta_w = self.output_net(combined)           # (B, N, out)
        return delta_w


def apply_ai_correction(model, features, w_baseline_re, w_baseline_im):
    """应用 AI 修正：W = W_baseline + ΔW_AI。

    Args:
        model:        DeepSetsModel
        features:     (batch, N, input_dim) 逐阵元输入
        w_baseline_re: (batch, N) 基线权值实部
        w_baseline_im: (batch, N) 基线权值虚部

    Returns:
        w_re: (batch, N) AI 修正后权值实部
        w_im: (batch, N) AI 修正后权值虚部
    """
    delta = model(features)  # (B, N, 2)
    w_re = w_baseline_re + delta[..., 0]
    w_im = w_baseline_im + delta[..., 1]
    return w_re, w_im


def count_parameters(model):
    """可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
