"""v4 服务器补测：NPU 基准 + 1024/4096 质量复核（服务器运行版）。

用途：在 Ascend 910 服务器上 clone 仓库后运行，补齐 v4 的两块数据尾巴：
  1) NPU 计时（256 维 v4，1024 与 4096 阵元，含端到端/批量/一致性）
  2) 官方评估器下 v4 三臂验收（273 方向，对照 v3 的 69/73、61/73）
  3) 4096 阵元 NPU 推理计时（此前 1.23 ms 为 128 维旧模型口径）

前置：git clone 后需要 v4 权重。获取方式二选一：
  a) git fetch origin && git checkout deepsets-models -- <权重路径>（若 tag 包含）
  b) 从本地 outputs/ 目录手动拷贝 deepsets_model_v4_256.pt

用法（服务器）：
  cd project
  python run_server_v4_benchmark.py                     # 全部
  python run_server_v4_benchmark.py --only npu          # 只跑计时
  python run_server_v4_benchmark.py --only acceptance   # 只跑官方口径验收

输出: outputs/server_v4_benchmark.json, outputs/server_v4_acceptance.json
"""

import os, sys, time, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
)
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.sum_diff import capon_nulling_2d
from mylib.official_evaluator import evaluate_official_case
from run_curved_verify import coordinate_taylor_3d
from run_generate_teacher import normalize_weights
from run_acceptance_v2 import get_scan_directions, get_null_dirs
from run_random_validation import random_null_dirs
from run_scale_fix_v4 import _features

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'deepsets_model_v4_256.pt')
BASELINE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results', 'stage2_strict_closure',
    'baseline')

HIDDEN = 256
COORD_NORM = 8.0
SLL_NORM = 50.0


def predict_w(model, px, py, pz, amp_x, amp_y, theta0, phi0):
    n = len(px)
    scale = float(n)
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    w_t = normalize_weights(w_t, px, py, pz, u0, v0, w0c)
    feat = _features(px, py, pz, w_t.real, w_t.imag, u0, v0, w0c, scale)
    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    return w_t + (delta[:, 0] + 1j * delta[:, 1]) / scale, w_t


def sync(dev):
    if 'npu' in str(dev):
        torch.npu.synchronize()


def run_npu_benchmark(model_npu, dev):
    report = {}
    for n_elem in [1024, 4096]:
        x = torch.randn(1, n_elem, 9, device=dev)
        for _ in range(20):
            with torch.no_grad():
                _ = model_npu(x)
        sync(dev)
        times = []
        for _ in range(1000):
            sync(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model_npu(x)
            sync(dev)
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        report[f'pure_{n_elem}'] = {
            'mean': float(arr.mean()), 'p50': float(np.percentile(arr, 50)),
            'p99': float(np.percentile(arr, 99)),
            'throughput_sps': float(1000 / (arr.sum() / 1000)),
            'reference_v3_1024': 0.499,
        }
        print(f'  pure {n_elem}: mean={arr.mean():.3f}ms '
              f'P99={np.percentile(arr, 99):.3f}ms '
              f'throughput={1000 / (arr.sum() / 1000):.0f}/s', flush=True)

    # 端到端 (1024)
    for _ in range(10):
        x_cpu = torch.randn(1, 1024, 9)
        _ = model_npu(x_cpu.to(dev)).to('cpu')
    sync(dev)
    e2e = []
    for _ in range(200):
        t0 = time.perf_counter()
        x_cpu = torch.randn(1, 1024, 9)
        out = model_npu(x_cpu.to(dev)).to('cpu')
        e2e.append((time.perf_counter() - t0) * 1000)
    arr = np.array(e2e)
    report['end_to_end_1024'] = {
        'mean': float(arr.mean()), 'p99': float(np.percentile(arr, 99)),
        'reference_v3': 0.632,
    }
    print(f'  e2e 1024: mean={arr.mean():.3f}ms '
          f'P99={np.percentile(arr, 99):.3f}ms', flush=True)

    # 批量吞吐 (1024)
    for bs in [8, 16, 64]:
        x = torch.randn(bs, 1024, 9, device=dev)
        for _ in range(10):
            with torch.no_grad():
                _ = model_npu(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = model_npu(x)
        sync(dev)
        tps = 200 * bs / (time.perf_counter() - t0)
        report[f'batch_{bs}_1024_sps'] = float(tps)
        print(f'  batch {bs}: {tps:.0f} samples/s', flush=True)
    return report


def run_consistency(model, dev):
    x = torch.randn(1, 1024, 9)
    with torch.no_grad():
        y_cpu = model(x).numpy().flatten()
        y_npu = model(x.to(dev)).to('cpu').numpy().flatten()
    max_err = float(np.max(np.abs(y_cpu - y_npu)))
    cos = float(np.dot(y_cpu, y_npu) /
                (np.linalg.norm(y_cpu) * np.linalg.norm(y_npu) + 1e-30))
    print(f'  consistency: max_err={max_err:.3e} cos={cos:.8f}')
    return {'max_abs_err': max_err, 'cos_sim': cos}


def run_acceptance(model):
    """v4 三臂官方口径验收（273 方向, 复用冻结 case_manifest）。"""
    import json
    manifest = json.load(open(os.path.join(
        BASELINE, 'case_manifest.json'), encoding='utf-8'))
    cases = manifest['regular'] + manifest['random']

    posx = uniform_linear_array_pos(32)
    posy = uniform_linear_array_pos(32)
    amp_x, amp_y = taylor_2d_separable(32, 32, 35)
    px = np.tile(posx[:, None], (1, 32)).ravel()
    py = np.tile(posy[None, :], (32, 1)).ravel()
    pz = np.zeros(1024)

    direct_rows, lcmv_rows = [], []
    t0 = time.time()
    for i, case in enumerate(cases):
        theta0 = case['theta0_deg']
        phi0 = case['phi0_deg']
        null_dirs = [tuple(nd) for nd in case['null_dirs']]

        w_ai, _ = predict_w(model, px, py, pz, amp_x, amp_y, theta0, phi0)
        amp_a = np.abs(w_ai).reshape(32, 32)
        ph_a = np.angle(w_ai).reshape(32, 32)
        r_d = evaluate_official_case(amp_a, ph_a, posx, posy, theta0, phi0,
                                     null_dirs=null_dirs)
        direct_rows.append(r_d['sum']['sll_db'])

        al, pl = capon_nulling_2d(posx, posy, amp_a, ph_a, theta0, phi0,
                                  null_dirs)
        r_l = evaluate_official_case(al, pl, posx, posy, theta0, phi0,
                                     null_dirs=null_dirs)
        lcmv_rows.append(r_l['sum']['sll_db'])

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(cases)} ({time.time()-t0:.0f}s)', flush=True)

    d = np.array(direct_rows)
    l = np.array(lcmv_rows)
    reg = np.array([1 if c['set'] == 'regular' else 0 for c in cases])
    out = {
        'model': 'deepsets_model_v4_256.pt',
        'official_metric_version': '1.0.0',
        'ai_direct': {
            'regular_pass': int(np.sum(d[reg == 1] <= -35)),
            'regular_total': int(np.sum(reg == 1)),
            'random_pass': int(np.sum(d[reg == 0] <= -35)),
            'random_total': int(np.sum(reg == 0)),
            'worst_db': float(d.max()),
        },
        'ai_lcmv': {
            'regular_pass': int(np.sum(l[reg == 1] <= -35)),
            'random_pass': int(np.sum(l[reg == 0] <= -35)),
            'worst_db': float(l.max()),
        },
        'reference_v3': {
            'ai_direct': {'regular': '69/73', 'random': '196/200',
                          'worst': -34.726},
            'ai_lcmv': {'regular': '61/73', 'random': '186/200',
                        'worst': -34.626},
        },
    }
    print(f'\n  v4 direct: {out["ai_direct"]["regular_pass"]}/'
          f'{out["ai_direct"]["regular_total"]} regular, '
          f'{out["ai_direct"]["random_pass"]}/{out["ai_direct"]["random_total"]}'
          f' random, worst {out["ai_direct"]["worst_db"]:.3f}')
    print(f'  v4 +LCMV: {out["ai_lcmv"]["regular_pass"]}/73 regular, '
          f'{out["ai_lcmv"]["random_pass"]}/200 random, '
          f'worst {out["ai_lcmv"]["worst_db"]:.3f}')
    print(f'  (v3 对照: direct 69/73+196/200, lcmv 61/73+186/200)')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['npu', 'acceptance', 'all'],
                        default='all')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f'v4 模型不存在: {MODEL_PATH}\n'
                         '请从本地 outputs/ 拷贝 deepsets_model_v4_256.pt '
                         '或从 deepsets-models tag 获取')

    import torch_npu
    dev = torch.device('npu:0')

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu',
                                      weights_only=True))
    model.eval()
    print(f'model: {MODEL_PATH} ({count_parameters(model):,} params)')

    if args.only in ('npu', 'all'):
        print('\n=== 1) NPU benchmark (v4 256-dim) ===', flush=True)
        model_npu = model.to(dev)
        report = run_npu_benchmark(model_npu, dev)
        report['consistency'] = run_consistency(model_npu, dev)
        report['platform'] = 'Ascend 910 server, torch_npu'
        with open(os.path.join(OUTPUT_DIR, 'server_v4_benchmark.json'),
                  'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print('  saved: outputs/server_v4_benchmark.json')

    if args.only in ('acceptance', 'all'):
        print('\n=== 2) v4 official-caliber acceptance (273 dirs) ===',
              flush=True)
        model_cpu = model.to('cpu')
        out = run_acceptance(model_cpu)
        with open(os.path.join(OUTPUT_DIR, 'server_v4_acceptance.json'),
                  'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print('  saved: outputs/server_v4_acceptance.json')


if __name__ == '__main__':
    main()
