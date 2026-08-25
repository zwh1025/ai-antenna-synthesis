"""Seq2Seq LSTM 模型（PyTorch 版）。

对应原 ModelDefinition.py 的 Encoder / Decoder / build_train_model。

结构：
  Encoder: 主干 LSTM → 分支1 LSTM + 分支2 LSTM → concat
  Decoder: 主干 LSTM + 分支1/2 LSTM → Dense → softmax 输出（幅值/相位）
  训练: teacher forcing
  推理: 逐时间步生成，separator 终止

Keras → PyTorch 映射：
  LSTM(return_sequences=True, return_state=True) → nn.LSTM(batch_first=True)
  lstm(x, initial_state=[h,c]) → lstm(x, (h.unsqueeze(0), c.unsqueeze(0)))
  Dense(activation='relu', he_normal) → Linear + relu + kaiming_normal_
  Dense(activation='softmax') → Linear + softmax(dim=-1)
  Concatenate(axis=-1) → torch.cat(dim=-1)
  Lambda(tf.stack, axis=-1) → torch.stack(dim=-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """编码器：主干 LSTM + 双分支 LSTM。

    输入  (batch, seq_len, input_dim)
    输出  (output, states)
      output: (batch, seq_len, branch1[-1]+branch2[-1])
      states: [[(h,c),...], [(h,c),...], [(h,c),...]]  三组 LSTM 状态
    """

    def __init__(self, input_dim, main_neurons, branch1_neurons, branch2_neurons):
        super().__init__()
        self.main_neurons = main_neurons
        self.branch1_neurons = branch1_neurons
        self.branch2_neurons = branch2_neurons

        self.main_lstm = nn.ModuleList()
        for i, h in enumerate(main_neurons):
            in_dim = input_dim if i == 0 else main_neurons[i - 1]
            self.main_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

        self.branch1_lstm = nn.ModuleList()
        for i, h in enumerate(branch1_neurons):
            in_dim = main_neurons[-1] if i == 0 else branch1_neurons[i - 1]
            self.branch1_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

        self.branch2_lstm = nn.ModuleList()
        for i, h in enumerate(branch2_neurons):
            in_dim = main_neurons[-1] if i == 0 else branch2_neurons[i - 1]
            self.branch2_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

    def forward(self, x):
        states = [[], [], []]

        for lstm in self.main_lstm:
            x, (h, c) = lstm(x)
            states[0].append((h.squeeze(0), c.squeeze(0)))

        for i, lstm in enumerate(self.branch1_lstm):
            inp = x if i == 0 else x1
            x1, (h, c) = lstm(inp)
            states[1].append((h.squeeze(0), c.squeeze(0)))

        for i, lstm in enumerate(self.branch2_lstm):
            inp = x if i == 0 else x2
            x2, (h, c) = lstm(inp)
            states[2].append((h.squeeze(0), c.squeeze(0)))

        output = torch.cat([x1, x2], dim=-1)
        return output, states


class Decoder(nn.Module):
    """解码器：主干 LSTM + 双分支 LSTM + Dense + softmax 输出。

    输入  (x, states)
      x:      (batch, seq_len, output_dim*2)
      states: 编码器输出的三组 LSTM 状态
    输出  (output, new_states)
      output:    (batch, seq_len, output_dim, 2)
      new_states: 更新后的三组 LSTM 状态
    """

    def __init__(self, output_dim, main_neurons, branch1_neurons, branch2_neurons,
                 branch1_dense_neurons=None, branch2_dense_neurons=None):
        super().__init__()
        self.output_dim = output_dim
        input_dim = output_dim * 2

        self.main_lstm = nn.ModuleList()
        for i, h in enumerate(main_neurons):
            in_dim = input_dim if i == 0 else main_neurons[i - 1]
            self.main_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

        self.branch1_lstm = nn.ModuleList()
        for i, h in enumerate(branch1_neurons):
            in_dim = main_neurons[-1] if i == 0 else branch1_neurons[i - 1]
            self.branch1_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

        self.branch2_lstm = nn.ModuleList()
        for i, h in enumerate(branch2_neurons):
            in_dim = main_neurons[-1] if i == 0 else branch2_neurons[i - 1]
            self.branch2_lstm.append(nn.LSTM(in_dim, h, batch_first=True))

        self.branch1_dense = nn.ModuleList()
        self.branch2_dense = nn.ModuleList()
        if branch1_dense_neurons is not None:
            for i, h in enumerate(branch1_dense_neurons):
                in_dim = branch1_neurons[-1] if i == 0 else branch1_dense_neurons[i - 1]
                self.branch1_dense.append(nn.Linear(in_dim, h))
        if branch2_dense_neurons is not None:
            for i, h in enumerate(branch2_dense_neurons):
                in_dim = branch2_neurons[-1] if i == 0 else branch2_dense_neurons[i - 1]
                self.branch2_dense.append(nn.Linear(in_dim, h))

        out1_in = (branch1_dense_neurons[-1] if branch1_dense_neurons
                   else branch1_neurons[-1])
        out2_in = (branch2_dense_neurons[-1] if branch2_dense_neurons
                   else branch2_neurons[-1])
        self.output1 = nn.Linear(out1_in, output_dim)
        self.output2 = nn.Linear(out2_in, output_dim)

        self._init_dense_weights()

    def _init_dense_weights(self):
        for m in list(self.branch1_dense) + list(self.branch2_dense):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x, states):
        new_states = [[], [], []]

        for i, lstm in enumerate(self.main_lstm):
            h0 = states[0][i][0].unsqueeze(0)
            c0 = states[0][i][1].unsqueeze(0)
            x, (h, c) = lstm(x, (h0, c0))
            new_states[0].append((h.squeeze(0), c.squeeze(0)))

        for i, lstm in enumerate(self.branch1_lstm):
            inp = x if i == 0 else x1
            h0 = states[1][i][0].unsqueeze(0)
            c0 = states[1][i][1].unsqueeze(0)
            x1, (h, c) = lstm(inp, (h0, c0))
            new_states[1].append((h.squeeze(0), c.squeeze(0)))

        for i, lstm in enumerate(self.branch2_lstm):
            inp = x if i == 0 else x2
            h0 = states[2][i][0].unsqueeze(0)
            c0 = states[2][i][1].unsqueeze(0)
            x2, (h, c) = lstm(inp, (h0, c0))
            new_states[2].append((h.squeeze(0), c.squeeze(0)))

        for dense in self.branch1_dense:
            x1 = F.relu(dense(x1))
        for dense in self.branch2_dense:
            x2 = F.relu(dense(x2))

        out1 = F.softmax(self.output1(x1), dim=-1)
        out2 = F.softmax(self.output2(x2), dim=-1)

        output = torch.stack([out1, out2], dim=-1)
        return output, new_states


class Seq2SeqModel(nn.Module):
    """训练用完整模型（teacher forcing）。

    forward(encoder_input, decoder_input) → decoder_output
      encoder_input: (batch, enc_seq, input_dim)
      decoder_input: (batch, dec_seq, output_dim, 2)
      decoder_output: (batch, dec_seq, output_dim, 2)
    """

    def __init__(self, input_dim, output_dim, num_neurons):
        super().__init__()
        self.encoder = Encoder(
            input_dim, num_neurons[0], num_neurons[1], num_neurons[2])
        self.decoder = Decoder(
            output_dim, num_neurons[0], num_neurons[1], num_neurons[2],
            num_neurons[3] if len(num_neurons) > 3 else None,
            num_neurons[4] if len(num_neurons) > 4 else None)

    def forward(self, encoder_input, decoder_input):
        batch, dec_seq = decoder_input.shape[:2]
        dec_input = decoder_input.reshape(batch, dec_seq, -1)

        _, states = self.encoder(encoder_input)
        output, _ = self.decoder(dec_input, states)
        return output


def predict_sequence(model, encoder_input, output_dim,
                     amp_range, phase_range, max_steps=50,
                     separator_threshold=0.5):
    """逐时间步推理。

    返回 (generated, raw_outputs)
      generated: [(amp, phase), ...]  已解码的数值对
      raw_outputs: [(output_dim, 2), ...]  原始网络输出向量
    """
    import numpy as np
    from mylib.embedding import map_vec_to_num_np

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        encoder_input = encoder_input.to(device)
        _, states = model.encoder(encoder_input)

        dec_input = torch.zeros(1, 1, output_dim, 2, device=device)
        dec_input[0, 0, output_dim - 1, :] = 1.0

        amp_dp = np.linspace(amp_range[0], amp_range[1], output_dim - 1)
        phase_dp = np.linspace(phase_range[0], phase_range[1], output_dim - 1)

        generated = []
        raw_outputs = []

        for step in range(max_steps):
            dec_in = dec_input.reshape(1, 1, -1)
            output, states = model.decoder(dec_in, states)
            raw = output[0, 0].cpu().numpy()
            raw_outputs.append(raw)

            if raw[-1, 0] > separator_threshold or raw[-1, 1] > separator_threshold:
                break

            amp = map_vec_to_num_np(raw[:, 0], amp_dp)
            phase = map_vec_to_num_np(raw[:, 1], phase_dp)
            generated.append((amp, phase))

            from mylib.embedding import map_num_to_vec_np
            amp_vec = map_num_to_vec_np(amp, amp_dp)
            phase_vec = map_num_to_vec_np(phase, phase_dp)
            dec_input = torch.from_numpy(
                np.stack([amp_vec, phase_vec], axis=0).reshape(1, 1, output_dim, 2)
            ).float().to(device)

        return generated, raw_outputs


def count_parameters(model):
    """可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
