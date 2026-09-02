"""v3 模型 NPU 基准补测：加载真实 v3 权重的推理基准。

背景：v3（平面增广修复捷径学习）此前仅在本地 CPU 训练验证。
本脚本在昇腾 NPU 上加载真实 v3 权重，补齐：
  1. NPU 纯推理基准（1024 阵元，batch=1，1000 轮，P50/P95/P99）
  2. NPU 端到端延迟（CPU 输入 -> H2D -> NPU 推理 -> D2H，200 轮）
  3. CPU 同口径对照（1000 轮）
  4. NPU/CPU 数值一致性（真实权重，max_err / cos_sim）
  5. 批量吞吐（batch=1/8/16/64）

用法（服务器）：
  python run_benchmark_v3.py --model_path outputs/deepsets_model_v3_256.pt

输出：outputs/benchmark_v3.json
"""

import sys, os, time, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel, count_parameters

OUTPUT = os.path.join(os.path.dirname(__file__), 'outputs', 'benchmark_v3.json')
N_ELEMENTS = 1024
INPUT_DIM = 9


def sync(dev):
    if 'npu' in str(dev):
        torch.npu.synchronize()


def bench_pure_inference(model, dev, n_iter=1000, tag=''):
    """纯推理：输入常驻设备，逐次计时。"""
    x = torch.randn(1, N_ELEMENTS, INPUT_DIM, device=dev)
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)
    sync(dev)

    times = []
    for _ in range(n_iter):
        sync(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        sync(dev)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    stats = {
        'config': f'v3 256-dim, {N_ELEMENTS} elements, batch=1, {n_iter} iters',
        'mean': float(arr.mean()), 'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)), 'p99': float(np.percentile(arr, 99)),
        'min': float(arr.min()), 'max': float(arr.max()), 'std': float(arr.std()),
        'throughput_sps': float(n_iter / (arr.sum() / 1000)),
    }
    print(f"\n  {tag} pure inference ({n_iter} iters):")
    print(f"    mean={stats['mean']:.3f}ms P50={stats['p50']:.3f}ms "
          f"P95={stats['p95']:.3f}ms P99={stats['p99']:.3f}ms "
          f"throughput={stats['throughput_sps']:.0f}/s")
    return stats


def bench_end_to_end(model, dev_npu, n_iter=200):
    """端到端：CPU 生成输入 -> H2D -> NPU 推理 -> D2H。"""
    for _ in range(10):
        x_cpu = torch.randn(1, N_ELEMENTS, INPUT_DIM)
        out = model(x_cpu.to(dev_npu))
        _ = out.to('cpu')
    sync(dev_npu)

    e2e, h2d, inf, d2h = [], [], [], []
    for _ in range(n_iter):
        x_cpu = torch.randn(1, N_ELEMENTS, INPUT_DIM)

        sync(dev_npu)
        t0 = time.perf_counter()
        x_npu = x_cpu.to(dev_npu)
        sync(dev_npu)
        t1 = time.perf_counter()
        h2d.append((t1 - t0) * 1000)

        with torch.no_grad():
            out_npu = model(x_npu)
        sync(dev_npu)
        t2 = time.perf_counter()
        inf.append((t2 - t1) * 1000)

        _ = out_npu.to('cpu')
        t3 = time.perf_counter()
        d2h.append((t3 - t2) * 1000)
        e2e.append((t3 - t0) * 1000)

    arr = np.array(e2e)
    stats = {
        'config': f'v3 256-dim, {N_ELEMENTS} elements, batch=1, {n_iter} iters',
        'total': {'mean': float(arr.mean()), 'p50': float(np.percentile(arr, 50)),
                  'p95': float(np.percentile(arr, 95)), 'p99': float(np.percentile(arr, 99))},
        'h2d_mean': float(np.mean(h2d)),
        'npu_infer_mean': float(np.mean(inf)),
        'd2h_mean': float(np.mean(d2h)),
    }
    print(f"\n  End-to-end ({n_iter} iters):")
    print(f"    total mean={stats['total']['mean']:.3f}ms "
          f"P50={stats['total']['p50']:.3f}ms P99={stats['total']['p99']:.3f}ms")
    print(f"    H2D={stats['h2d_mean']:.3f}ms infer={stats['npu_infer_mean']:.3f}ms "
          f"D2H={stats['d2h_mean']:.3f}ms")
    return stats


def bench_batch_throughput(model, dev, batch_sizes=(1, 8, 16, 64), n_iter=200):
    """批量吞吐。"""
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, N_ELEMENTS, INPUT_DIM, device=dev)
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        sync(dev)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            with torch.no_grad():
                _ = model(x)
        sync(dev)
        dt = time.perf_counter() - t0
        tps = n_iter * bs / dt
        results[f'bs{bs}'] = {'throughput_sps': float(tps),
                              'ms_per_sample': float(dt / (n_iter * bs) * 1000)}
        print(f"    batch={bs:2d}: {tps:8.0f} samples/s")
    return results


def check_consistency(model, dev_npu):
    """NPU/CPU 数值一致性（真实 v3 权重）。"""
    x = torch.randn(1, N_ELEMENTS, INPUT_DIM)
    with torch.no_grad():
        y_cpu = model(x).numpy().flatten()
        y_npu = model(x.to(dev_npu)).to('cpu').numpy().flatten()
    max_err = float(np.max(np.abs(y_cpu - y_npu)))
    cos = float(np.dot(y_cpu, y_npu) /
                (np.linalg.norm(y_cpu) * np.linalg.norm(y_npu) + 1e-30))
    print(f"\n  Consistency (real v3 weights): max_err={max_err:.3e} cos_sim={cos:.8f}")
    return {'max_abs_err': max_err, 'cos_sim': cos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=os.path.join('outputs', 'deepsets_model_v3_256.pt'))
    args = parser.parse_args()

    import torch_npu
    dev_npu = torch.device('npu:0')

    model = DeepSetsModel(INPUT_DIM, 256, 2)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu',
                                      weights_only=True))
    model.eval()
    print(f"Loaded v3 model: {args.model_path} ({count_parameters(model):,} params)")

    report = {'model': os.path.basename(args.model_path), 'n_elements': N_ELEMENTS}

    print('=' * 78)
    print('1) NPU pure inference')
    report['npu_pure'] = bench_pure_inference(model.to(dev_npu), dev_npu, tag='NPU')

    print('\n2) NPU end-to-end')
    report['npu_end_to_end'] = bench_end_to_end(model, dev_npu)

    print('\n3) CPU comparison')
    model_cpu = DeepSetsModel(INPUT_DIM, 256, 2)
    model_cpu.load_state_dict(torch.load(args.model_path, map_location='cpu',
                                         weights_only=True))
    model_cpu.eval()
    report['cpu_pure'] = bench_pure_inference(model_cpu, 'cpu', tag='CPU')

    print('\n4) NPU/CPU consistency')
    report['consistency'] = check_consistency(model, dev_npu)

    print('\n5) NPU batch throughput')
    print('    batch  : throughput')
    report['npu_batch'] = bench_batch_throughput(model, dev_npu)

    n_mean = report['npu_pure']['mean']
    c_mean = report['cpu_pure']['mean']
    print('\n' + '=' * 78)
    print('SUMMARY')
    print(f"  NPU vs CPU (mean): {c_mean/n_mean:.1f}x")
    print(f"  NPU vs CPU (P99):  {report['cpu_pure']['p99']/report['npu_pure']['p99']:.1f}x")
    print(f"  NPU vs SOCP 23s:   {23000/n_mean:,.0f}x")
    print(f"  (对照 v2 128-dim: NPU mean 0.504ms, CPU P50 1.332ms)")

    with open(OUTPUT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUTPUT}")
    print('=' * 78)


if __name__ == '__main__':
    main()
