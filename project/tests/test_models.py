"""阶段 1.6 Encoder/Decoder 模型测试。"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mylib.models import Encoder, Decoder, Seq2SeqModel, count_parameters


def _small_config():
    """小规模网络配置，用于快速测试。"""
    input_dim = 32
    output_dim = 32
    main_neurons = [64, 64]
    branch_neurons = [32, 32, 16, 16]
    dense_neurons = [16, 8, 4]
    num_neurons = [main_neurons, branch_neurons, branch_neurons,
                   dense_neurons, dense_neurons]
    return input_dim, output_dim, num_neurons


def test_encoder_output_shape():
    """Encoder 输出形状和状态数量正确。"""
    input_dim, _, num_neurons = _small_config()
    enc = Encoder(input_dim, num_neurons[0], num_neurons[1], num_neurons[2])

    batch, seq = 4, 30
    x = torch.randn(batch, seq, input_dim)
    output, states = enc(x)

    expected_out = num_neurons[1][-1] + num_neurons[2][-1]
    assert output.shape == (batch, seq, expected_out), \
        f"output {output.shape} != ({batch}, {seq}, {expected_out})"
    assert len(states[0]) == len(num_neurons[0])
    assert len(states[1]) == len(num_neurons[1])
    assert len(states[2]) == len(num_neurons[2])
    for i, h in enumerate(num_neurons[0]):
        assert states[0][i][0].shape == (batch, h)
    for i, h in enumerate(num_neurons[1]):
        assert states[1][i][0].shape == (batch, h)

    print("PASS: test_encoder_output_shape")


def test_decoder_output_shape():
    """Decoder 输出形状正确。"""
    input_dim, output_dim, num_neurons = _small_config()
    enc = Encoder(input_dim, num_neurons[0], num_neurons[1], num_neurons[2])
    dec = Decoder(output_dim, num_neurons[0], num_neurons[1], num_neurons[2],
                  num_neurons[3], num_neurons[4])

    batch, enc_seq, dec_seq = 4, 30, 35
    enc_x = torch.randn(batch, enc_seq, input_dim)
    _, states = enc(enc_x)

    dec_x = torch.randn(batch, dec_seq, output_dim * 2)
    output, new_states = dec(dec_x, states)

    assert output.shape == (batch, dec_seq, output_dim, 2), \
        f"output {output.shape} != ({batch}, {dec_seq}, {output_dim}, 2)"
    assert len(new_states[0]) == len(num_neurons[0])

    sums = output[:, :, :, :].sum(dim=2)
    assert torch.allclose(sums, torch.ones(batch, dec_seq, 2), atol=1e-4), \
        "softmax outputs should sum to 1 along output_dim"

    print("PASS: test_decoder_output_shape")


def test_seq2seq_forward():
    """完整模型前向传播。"""
    input_dim, output_dim, num_neurons = _small_config()
    model = Seq2SeqModel(input_dim, output_dim, num_neurons)

    batch, enc_seq, dec_seq = 4, 30, 35
    enc_x = torch.randn(batch, enc_seq, input_dim)
    dec_x = torch.randn(batch, dec_seq, output_dim, 2)

    output = model(enc_x, dec_x)
    assert output.shape == (batch, dec_seq, output_dim, 2)

    print("PASS: test_seq2seq_forward")


def test_gradient_flow():
    """梯度可以回传到 encoder。"""
    input_dim, output_dim, num_neurons = _small_config()
    model = Seq2SeqModel(input_dim, output_dim, num_neurons)

    batch, enc_seq, dec_seq = 2, 20, 25
    enc_x = torch.randn(batch, enc_seq, input_dim)
    dec_x = torch.randn(batch, dec_seq, output_dim, 2)
    target = torch.randn(batch, dec_seq, output_dim, 2)

    output = model(enc_x, dec_x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum().item() > 0:
                has_grad = True
                break
    assert has_grad, "no gradients found in any parameter"

    enc_first_weight = list(model.encoder.main_lstm[0].parameters())[0]
    assert enc_first_weight.grad is not None
    assert enc_first_weight.grad.abs().sum().item() > 0, \
        "encoder gradients are zero"

    print("PASS: test_gradient_flow")


def test_state_independence():
    """decoder 不应修改 encoder 的原始 states（每次 forward 创建新 states）。"""
    input_dim, output_dim, num_neurons = _small_config()
    enc = Encoder(input_dim, num_neurons[0], num_neurons[1], num_neurons[2])
    dec = Decoder(output_dim, num_neurons[0], num_neurons[1], num_neurons[2],
                  num_neurons[3], num_neurons[4])

    batch, seq = 2, 20
    x = torch.randn(batch, seq, input_dim)
    _, states = enc(x)

    orig_h0 = states[0][0][0].clone()
    dec_x = torch.randn(batch, 25, output_dim * 2)
    _, new_states = dec(dec_x, states)

    assert torch.equal(states[0][0][0], orig_h0), \
        "decoder modified encoder states in-place"

    print("PASS: test_state_independence")


def test_parameter_count():
    """参数量统计。"""
    input_dim, output_dim, num_neurons = _small_config()
    model = Seq2SeqModel(input_dim, output_dim, num_neurons)
    n = count_parameters(model)
    assert n > 0

    full_config = {
        'main': [1024, 1024],
        'branch': [512, 512, 256, 256],
        'dense': [128, 64, 32],
    }
    full_num = [full_config['main'], full_config['branch'], full_config['branch'],
                full_config['dense'], full_config['dense']]
    full_model = Seq2SeqModel(32, 32, full_num)
    full_n = count_parameters(full_model)
    print(f"PASS: test_parameter_count (small={n:,}, full={full_n:,})")


def test_decoder_single_step():
    """decoder 处理单时间步（推理模式基础）。"""
    input_dim, output_dim, num_neurons = _small_config()
    enc = Encoder(input_dim, num_neurons[0], num_neurons[1], num_neurons[2])
    dec = Decoder(output_dim, num_neurons[0], num_neurons[1], num_neurons[2],
                  num_neurons[3], num_neurons[4])

    batch, seq = 1, 30
    enc_x = torch.randn(batch, seq, input_dim)
    _, states = enc(enc_x)

    dec_x = torch.zeros(batch, 1, output_dim * 2)
    output, new_states = dec(dec_x, states)
    assert output.shape == (1, 1, output_dim, 2)

    output2, _ = dec(dec_x, new_states)
    assert output2.shape == (1, 1, output_dim, 2)

    print("PASS: test_decoder_single_step")


if __name__ == '__main__':
    tests = [
        test_encoder_output_shape,
        test_decoder_output_shape,
        test_seq2seq_forward,
        test_gradient_flow,
        test_state_independence,
        test_parameter_count,
        test_decoder_single_step,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("=== ALL TESTS PASSED ===")
