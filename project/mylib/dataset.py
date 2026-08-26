"""数据集生成模块。

对应原 CreateDataset.py，修正项：
  1. 相位用弧度（beam_steering_phase），分割点 linspace(0, 2π, L_Y)
  2. 使用 mylib.antenna_calc 的修正激励生成
  3. 使用 mylib.embedding 的 map_num_to_vec_np
  4. SLL 不在输入序列中（与原代码一致——SLL_vec 行被注释）
"""

import numpy as np

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    chebyshev_excitation,
    taylor_excitation,
    beam_steering_phase,
)
from mylib.embedding import map_num_to_vec_np


def create_single_data(N_units, theta0, SLL,
                       theta0_range, SLL_range, pos_range, L_X,
                       amp_range, phase_range, L_Y,
                       N_units_max,
                       reference='Taylor', dtype=np.float32):
    """生成单组训练数据。

    输入序列: [sep] + [SLL] + [sep] + [theta0] + [sep] + [pos...] + [sep] + [pad...]
    输出序列: [sep] + [amp/phase...] + [sep] + [pad...]

    返回 (X, Y):
      X: (enc_seq, L_X+1)            输入序列
      Y: (dec_seq, L_Y+1, 2)         输出序列（幅值/相位两路）
    """
    pos = uniform_linear_array_pos(N_units)
    pos_norm = pos / np.max(np.abs(pos)) if np.max(np.abs(pos)) > 0 else pos

    dp_theta0 = np.linspace(theta0_range[0], theta0_range[1], L_X)
    dp_sll = np.linspace(SLL_range[0], SLL_range[1], L_X)
    dp_pos = np.linspace(pos_range[0], pos_range[1], L_X)
    dp_amp = np.linspace(amp_range[0], amp_range[1], L_Y)
    dp_phase = np.linspace(phase_range[0], phase_range[1], L_Y)

    sep_X = np.hstack([np.zeros(L_X, dtype=dtype), [1.0]])
    sep_Y = np.hstack([np.zeros(L_Y, dtype=dtype), [1.0]])
    pad_X = np.zeros(L_X + 1, dtype=dtype)
    pad_Y = np.zeros_like(sep_Y)

    if isinstance(theta0, (int, float)):
        theta0 = [theta0]

    sll_vec = map_num_to_vec_np(float(SLL), dp_sll, dtype=dtype)
    theta0_vecs = [map_num_to_vec_np(t, dp_theta0, dtype=dtype) for t in theta0]
    pos_vecs = [map_num_to_vec_np(p, dp_pos, dtype=dtype) for p in pos_norm]

    X = np.stack(
        [sep_X] + [sll_vec] + [sep_X] +
        theta0_vecs + [sep_X] +
        pos_vecs + [sep_X] +
        (N_units_max - N_units) * [pad_X], axis=0)

    if reference == 'Chebyshev':
        amp = chebyshev_excitation(N_units, SLL)
    elif reference == 'Taylor':
        amp = taylor_excitation(N_units * 0.5, pos, SLL)
    else:
        raise ValueError(f"unknown reference: {reference}")

    if len(theta0) == 1:
        phase = beam_steering_phase(pos, theta0[0]) % (2 * np.pi)
    else:
        amps = amp.reshape(1, -1)
        phases = np.stack([
            beam_steering_phase(pos, t) for t in theta0
        ], axis=0)
        complex_exc = np.sum(amps * np.exp(1j * phases), axis=0)
        amp = np.abs(complex_exc)
        amp = amp / np.max(amp)
        phase = np.angle(complex_exc) % (2 * np.pi)

    amp_vecs = np.stack(
        [sep_Y] +
        [map_num_to_vec_np(a, dp_amp, dtype=dtype) for a in amp] + [sep_Y] +
        (N_units_max - N_units + 1) * [pad_Y], axis=0)
    phase_vecs = np.stack(
        [sep_Y] +
        [map_num_to_vec_np(p, dp_phase, dtype=dtype) for p in phase] + [sep_Y] +
        (N_units_max - N_units + 1) * [pad_Y], axis=0)

    Y = np.stack([amp_vecs, phase_vecs], axis=2)
    return X, Y


def create_dataset(N_list, theta0_list, SLL_list,
                   reference='Taylor', dtype=np.float32):
    """批量生成训练数据集。

    返回 (X, Y):
      X: (n_samples, enc_seq, L_X+1)
      Y: (n_samples, dec_seq, L_Y+1, 2)
    """
    N_units_max = int(np.max(N_list)) + 1
    theta0_range = (0.0, 180.0)
    SLL_range = (0.0, 50.0)
    # 固定位置编码范围，不随 N_list 变化
    pos_range = (-1.0, 1.0)
    L_X = 31
    amp_range = (0.0, 1.0)
    phase_range = (0.0, 2 * np.pi)
    L_Y = 31

    results = [
        create_single_data(
            int(N), float(t), int(s),
            theta0_range, SLL_range, pos_range, L_X,
            amp_range, phase_range, L_Y,
            N_units_max, reference, dtype)
        for N in N_list for t in theta0_list for s in SLL_list
    ]
    X = np.stack([r[0] for r in results], axis=0)
    Y = np.stack([r[1] for r in results], axis=0)
    return X, Y


def prepare_training_data(X, Y):
    """准备 teacher forcing 训练数据。

    decoder_output = decoder_input 右移一位（首位去掉，末位补零）。

    返回 (encoder_input, decoder_input, decoder_output)。
    """
    encoder_input = X.astype(np.float32)
    decoder_input = Y.astype(np.float32)
    decoder_output = np.pad(
        decoder_input[:, 1:, :, :],
        pad_width=((0, 0), (0, 1), (0, 0), (0, 0)),
        mode='constant', constant_values=0)
    return encoder_input, decoder_input, decoder_output


def get_dataset_config(N_list):
    """返回数据集配置参数（供推理使用）。"""
    N_units_max = int(np.max(N_list)) + 1
    return {
        'N_units_max': N_units_max,
        'L_X': 31,
        'L_Y': 31,
        'input_dim': 32,
        'output_dim': 32,
        'amp_range': (0.0, 1.0),
        'phase_range': (0.0, 2 * np.pi),
        'theta0_range': (0.0, 180.0),
        'pos_range': ((0.5 - N_units_max / 2) * 0.5,
                      (N_units_max / 2 - 0.5) * 0.5),
    }
