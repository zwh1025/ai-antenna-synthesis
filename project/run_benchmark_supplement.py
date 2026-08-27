"""补充测试：端到端延迟 + 1000轮推理统计。

补做内容：
  1. 端到端延迟：CPU生成输入 -> H2D -> NPU推理 -> D2H -> CPU拿到结果
  2. 1000轮推理计时：128维/1024阵元/batch=1，稳定P99
  3. 1024阵元主结果表（batch=1/16/64）
"""

import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel

OUTPUT = os.path.join(os.path.dirname(__file__), 'outputs', 'benchmark_supplement.json')


def sync(dev):
    if 'npu' in str(dev):
        torch.npu.synchronize()


def main():
    import torch_npu
    dev_npu = torch.device('npu:0')
    dev_cpu = torch.device('cpu')

    report = {}

    # ============================================================
    # 1. 端到端延迟测试
    # ============================================================
    print('=' * 80)
    print('Test 1: End-to-End Latency')
    print('  CPU generate input -> H2D -> NPU inference -> D2H -> CPU result')
    print('=' * 80)

    model = DeepSetsModel(9, 128, 2).to(dev_npu)
    model.eval()

    n_warmup = 10
    n_bench = 200

    # 预热
    for _ in range(n_warmup):
        x_cpu = torch.randn(1, 1024, 9)
        x_npu = x_cpu.to(dev_npu)
        with torch.no_grad():
            out_npu = model(x_npu)
        _ = out_npu.to('cpu')
    sync(dev_npu)

    # 计时：完整端到端
    e2e_times = []
    h2d_times = []
    infer_times = []
    d2h_times = []

    for _ in range(n_bench):
        # Step 1: CPU生成输入
        x_cpu = torch.randn(1, 1024, 9)

        # Step 2: H2D
        sync(dev_npu)
        t0 = time.perf_counter()
        x_npu = x_cpu.to(dev_npu)
        sync(dev_npu)
        t1 = time.perf_counter()
        h2d_times.append((t1 - t0) * 1000)

        # Step 3: NPU推理
        t2 = time.perf_counter()
        with torch.no_grad():
            out_npu = model(x_npu)
        sync(dev_npu)
        t3 = time.perf_counter()
        infer_times.append((t3 - t2) * 1000)

        # Step 4: D2H
        t4 = time.perf_counter()
        out_cpu = out_npu.to('cpu')
        t5 = time.perf_counter()
        d2h_times.append((t5 - t4) * 1000)

        # 总计
        e2e_times.append((t5 - t0) * 1000)

    e2e = np.array(e2e_times)
    h2d = np.array(h2d_times)
    inf = np.array(infer_times)
    d2h = np.array(d2h_times)

    report['end_to_end'] = {
        'config': '128-dim, 1024 elements, batch=1, 200 iterations',
        'total': {'mean': float(e2e.mean()), 'p50': float(np.percentile(e2e, 50)),
                  'p95': float(np.percentile(e2e, 95)), 'p99': float(np.percentile(e2e, 99))},
        'h2d': {'mean': float(h2d.mean()), 'p50': float(np.percentile(h2d, 50)),
                'p95': float(np.percentile(h2d, 95))},
        'npu_infer': {'mean': float(inf.mean()), 'p50': float(np.percentile(inf, 50)),
                      'p95': float(np.percentile(inf, 95))},
        'd2h': {'mean': float(d2h.mean()), 'p50': float(np.percentile(d2h, 50)),
                'p95': float(np.percentile(d2h, 95))},
    }

    print(f'\n  End-to-End ({n_bench} iterations):')
    print(f'    Total:    mean={e2e.mean():.3f}ms  P50={np.percentile(e2e,50):.3f}ms  '
          f'P95={np.percentile(e2e,95):.3f}ms  P99={np.percentile(e2e,99):.3f}ms')
    print(f'    H2D:      mean={h2d.mean():.3f}ms  P50={np.percentile(h2d,50):.3f}ms  '
          f'P95={np.percentile(h2d,95):.3f}ms')
    print(f'    NPU infer: mean={inf.mean():.3f}ms  P50={np.percentile(inf,50):.3f}ms  '
          f'P95={np.percentile(inf,95):.3f}ms')
    print(f'    D2H:      mean={d2h.mean():.3f}ms  P50={np.percentile(d2h,50):.3f}ms  '
          f'P95={np.percentile(d2h,95):.3f}ms')
    print(f'    Inference占比: {inf.mean()/e2e.mean()*100:.1f}%')

    # ============================================================
    # 2. 1000轮推理计时（稳定P99）
    # ============================================================
    print('\n' + '=' * 80)
    print('Test 2: 1000-Iteration Inference (stable P99)')
    print('  128-dim, 1024 elements, batch=1, input pre-loaded on NPU')
    print('=' * 80)

    x_npu = torch.randn(1, 1024, 9, device=dev_npu)
    model = DeepSetsModel(9, 128, 2).to(dev_npu)
    model.eval()

    # 预热
    for _ in range(20):
        with torch.no_grad():
            _ = model(x_npu)
    sync(dev_npu)

    # 1000轮
    n_total = 1000
    times = []
    for _ in range(n_total):
        sync(dev_npu)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_npu)
        sync(dev_npu)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    arr = np.array(times)
    report['long_inference'] = {
        'config': f'128-dim, 1024 elements, batch=1, {n_total} iterations, input on NPU',
        'n_iterations': n_total,
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'std': float(arr.std()),
        'throughput_sps': float(n_total / (arr.sum() / 1000)),
    }

    print(f'\n  {n_total} iterations:')
    print(f'    mean={arr.mean():.3f}ms  P50={np.percentile(arr,50):.3f}ms  '
          f'P95={np.percentile(arr,95):.3f}ms  P99={np.percentile(arr,99):.3f}ms')
    print(f'    min={arr.min():.3f}ms  max={arr.max():.3f}ms  std={arr.std():.3f}ms')
    print(f'    throughput={n_total/(arr.sum()/1000):.0f} samples/sec')

    # ============================================================
    # 3. CPU端到端对比（同样1000轮）
    # ============================================================
    print('\n' + '=' * 80)
    print('Test 3: CPU 1000-Iteration Inference (comparison)')
    print('=' * 80)

    model_cpu = DeepSetsModel(9, 128, 2)
    model_cpu.eval()
    x_cpu = torch.randn(1, 1024, 9)

    # 预热
    for _ in range(20):
        with torch.no_grad():
            _ = model_cpu(x_cpu)

    times_cpu = []
    for _ in range(n_total):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(x_cpu)
        t1 = time.perf_counter()
        times_cpu.append((t1 - t0) * 1000)

    arr_cpu = np.array(times_cpu)
    report['cpu_long_inference'] = {
        'config': f'128-dim, 1024 elements, batch=1, {n_total} iterations',
        'n_iterations': n_total,
        'mean': float(arr_cpu.mean()),
        'p50': float(np.percentile(arr_cpu, 50)),
        'p95': float(np.percentile(arr_cpu, 95)),
        'p99': float(np.percentile(arr_cpu, 99)),
        'min': float(arr_cpu.min()),
        'max': float(arr_cpu.max()),
        'std': float(arr_cpu.std()),
        'throughput_sps': float(n_total / (arr_cpu.sum() / 1000)),
    }

    print(f'\n  CPU {n_total} iterations:')
    print(f'    mean={arr_cpu.mean():.3f}ms  P50={np.percentile(arr_cpu,50):.3f}ms  '
          f'P95={np.percentile(arr_cpu,95):.3f}ms  P99={np.percentile(arr_cpu,99):.3f}ms')
    print(f'    min={arr_cpu.min():.3f}ms  max={arr_cpu.max():.3f}ms  std={arr_cpu.std():.3f}ms')
    print(f'    throughput={n_total/(arr_cpu.sum()/1000):.0f} samples/sec')

    # ============================================================
    # 4. 对比汇总
    # ============================================================
    print('\n' + '=' * 80)
    print('COMPARISON SUMMARY')
    print('=' * 80)

    npu_mean = report['long_inference']['mean']
    cpu_mean = report['cpu_long_inference']['mean']
    e2e_mean = report['end_to_end']['total']['mean']
    e2e_p99 = report['end_to_end']['total']['p99']
    npu_p99 = report['long_inference']['p99']
    cpu_p99 = report['cpu_long_inference']['p99']

    print(f'\n  Pure inference (1000 iterations, input on NPU):')
    print(f'    NPU: mean={npu_mean:.3f}ms P50={report["long_inference"]["p50"]:.3f}ms '
          f'P95={report["long_inference"]["p95"]:.3f}ms P99={npu_p99:.3f}ms')
    print(f'    CPU: mean={cpu_mean:.3f}ms P50={report["cpu_long_inference"]["p50"]:.3f}ms '
          f'P95={report["cpu_long_inference"]["p95"]:.3f}ms P99={cpu_p99:.3f}ms')
    print(f'    Speedup: {cpu_mean/npu_mean:.1f}x (mean), {cpu_p99/npu_p99:.1f}x (P99)')

    print(f'\n  End-to-end (CPU->H2D->NPU->D2H->CPU, 200 iterations):')
    print(f'    mean={e2e_mean:.3f}ms P50={report["end_to_end"]["total"]["p50"]:.3f}ms '
          f'P95={report["end_to_end"]["total"]["p95"]:.3f}ms P99={e2e_p99:.3f}ms')
    print(f'    vs pure NPU inference: overhead={e2e_mean-npu_mean:.3f}ms ({(e2e_mean-npu_mean)/e2e_mean*100:.1f}%)')

    print(f'\n  vs SOCP (13,000ms):')
    print(f'    Pure NPU inference: {npu_mean:.3f}ms = {13000/npu_mean:.0f}x')
    print(f'    End-to-end:        {e2e_mean:.3f}ms = {13000/e2e_mean:.0f}x')

    # ============================================================
    # 5. 1024阵元主结果表
    # ============================================================
    print(f'\n  1024阵元主结果 (128维模型):')
    print(f'    batch=1:  NPU {report["long_inference"]["mean"]:.3f}ms '
          f'({report["long_inference"]["throughput_sps"]:.0f}/s)')
    print(f'    E2E:      {e2e_mean:.3f}ms (含数据传输)')
    print(f'    vs SOCP:  {13000/npu_mean:.0f}x (纯推理), {13000/e2e_mean:.0f}x (端到端)')

    # 保存
    with open(OUTPUT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n  Saved: {OUTPUT}')
    print('=' * 80)


if __name__ == '__main__':
    main()
