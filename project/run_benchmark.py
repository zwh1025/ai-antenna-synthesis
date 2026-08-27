"""CPU/NPU 标准基准程序。

在同一台昇腾服务器上，用相同模型、相同数据、相同随机种子，
一次性生成以下完整报告：

  A. 硬件信息（CPU型号、NPU型号、内存）
  B. 精度一致性（CPU vs NPU 输出最大误差）
  C. 训练速度（不同batch_size + 不同模型宽度）
  D. 推理速度（不同batch_size + 不同阵元数）
  E. 延迟分布（P50/P95/P99/min/max + 直方图数据）
  F. 吞吐量（连续推理 samples/sec）
  G. 数据传输开销（H2D/D2H）
  H. 内存占用

用法: python run_benchmark.py [--quick]
输出: outputs/benchmark_report.json + 控制台表格
"""

import sys, os, time, json, argparse, platform
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel, count_parameters

OUTPUT = os.path.join(os.path.dirname(__file__), 'outputs', 'benchmark_report.json')


# ============================================================
#  工具函数
# ============================================================
def sync(device):
    if 'npu' in str(device):
        torch.npu.synchronize()
    elif 'cuda' in str(device):
        torch.cuda.synchronize()


def timer(fn, device, n_warmup=5, n_iter=30):
    """返回 {mean, p50, p95, p99, min, max, std, raw}。"""
    for _ in range(n_warmup):
        fn(); sync(device)
    times = []
    for _ in range(n_iter):
        sync(device)
        t0 = time.perf_counter()
        fn(); sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'std': float(arr.std()),
        'raw': arr.tolist(),
    }


def fmt_ms(v):
    return f'{v:.2f}ms'


# ============================================================
#  A. 硬件信息
# ============================================================
def collect_hardware():
    info = {'platform': platform.platform()}

    # CPU
    info['cpu'] = {
        'architecture': platform.machine(),
        'cores': os.cpu_count(),
        'threads': torch.get_num_threads(),
    }
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('CPU implementer'):
                    info['cpu']['implementer'] = line.split(':')[1].strip()
                elif line.startswith('CPU part'):
                    info['cpu']['part'] = line.split(':')[1].strip()
                elif line.startswith('cpu MHz'):
                    info['cpu']['mhz'] = float(line.split(':')[1].strip())
                    break
    except:
        pass

    # 内存
    try:
        import subprocess
        mem = subprocess.check_output(['free', '-h']).decode()
        info['memory'] = mem.strip()
    except:
        info['memory'] = 'N/A'

    # NPU
    info['npu'] = {}
    try:
        import torch_npu
        if torch.npu.is_available():
            info['npu']['name'] = torch.npu.get_device_name(0)
            prop = torch.npu.get_device_properties(0)
            info['npu']['total_memory_mb'] = prop.total_memory // (1024*1024)
            info['npu']['cube_cores'] = prop.cube_core_num
            info['npu']['vector_cores'] = prop.vector_core_num
            info['npu']['l2_cache_mb'] = prop.L2_cache_size // (1024*1024)
            info['npu']['available'] = True
        else:
            info['npu']['available'] = False
    except ImportError:
        info['npu']['available'] = False

    return info


# ============================================================
#  B. 精度一致性
# ============================================================
def check_consistency(devices, hidden=128, n_elem=1024, bs=16):
    """CPU vs NPU 输出一致性（相同权重和输入）。"""
    print('\n[B] 精度一致性测试')
    print('-' * 70)

    torch.manual_seed(42)
    model_cpu = DeepSetsModel(9, hidden, 2)
    x = torch.randn(bs, n_elem, 9)

    results = {}
    outputs = {}
    for dev_name, dev in devices:
        m = DeepSetsModel(9, hidden, 2).to(dev)
        m.load_state_dict(model_cpu.state_dict())
        m.eval()
        with torch.no_grad():
            out = m(x.to(dev))
        outputs[dev_name] = out.cpu()
        print(f'  {dev_name}: output shape={list(out.shape)}, '
              f'mean={out.mean().item():.6f}, std={out.std().item():.6f}')

    # 对比
    if len(outputs) >= 2:
        dev_names = list(outputs.keys())
        for i in range(len(dev_names)):
            for j in range(i+1, len(dev_names)):
                a, b = outputs[dev_names[i]], outputs[dev_names[j]]
                max_err = (a - b).abs().max().item()
                mean_err = (a - b).abs().mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
                key = f'{dev_names[i]}_vs_{dev_names[j]}'
                results[key] = {
                    'max_abs_error': max_err,
                    'mean_abs_error': mean_err,
                    'cosine_similarity': cos_sim,
                }
                print(f'  {key}: max_err={max_err:.2e}, mean_err={mean_err:.2e}, '
                      f'cos_sim={cos_sim:.8f}')

    return results


# ============================================================
#  C. 训练速度 (batch_size x model_width)
# ============================================================
def bench_train(devices, batch_sizes, hidden_sizes, n_elem=1024):
    print('\n[C] 训练速度测试')
    print(f'{"batch":>6} {"hidden":>8} {"device":>5} '
          f'{"mean":>8} {"p50":>8} {"p95":>8} {"p99":>8} {"min":>8} {"max":>8}')
    print('-' * 75)

    results = {}
    for h in hidden_sizes:
        for bs in batch_sizes:
            for dev_name, dev in devices:
                torch.manual_seed(42)
                model = DeepSetsModel(9, h, 2).to(dev)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                crit = torch.nn.MSELoss()
                x = torch.randn(bs, n_elem, 9, device=dev)
                y = torch.randn(bs, n_elem, 2, device=dev)

                def fn():
                    opt.zero_grad()
                    out = model(x)
                    loss = crit(out, y)
                    loss.backward()
                    opt.step()

                t = timer(fn, dev)
                key = f'h{h}_bs{bs}_{dev_name}'
                results[key] = t
                print(f'{bs:>6} {h:>8} {dev_name:>5} '
                      f'{fmt_ms(t["mean"]):>8} {fmt_ms(t["p50"]):>8} '
                      f'{fmt_ms(t["p95"]):>8} {fmt_ms(t["p99"]):>8} '
                      f'{fmt_ms(t["min"]):>8} {fmt_ms(t["max"]):>8}')

    return results


# ============================================================
#  D. 推理速度 (batch_size x element_count)
# ============================================================
def bench_infer(devices, batch_sizes, elem_sizes, hidden=128):
    print('\n[D] 推理速度测试')
    print(f'{"batch":>6} {"n_elem":>8} {"device":>5} '
          f'{"mean":>8} {"p50":>8} {"p95":>8} {"throughput":>12}')
    print('-' * 70)

    results = {}
    for ne in elem_sizes:
        for bs in batch_sizes:
            for dev_name, dev in devices:
                model = DeepSetsModel(9, hidden, 2).to(dev)
                model.eval()
                x = torch.randn(bs, ne, 9, device=dev)

                def fn():
                    with torch.no_grad():
                        _ = model(x)

                t = timer(fn, dev)
                tp = bs / (t['mean'] / 1000)
                key = f'ne{ne}_bs{bs}_{dev_name}'
                t['throughput'] = tp
                results[key] = t
                print(f'{bs:>6} {ne:>8} {dev_name:>5} '
                      f'{fmt_ms(t["mean"]):>8} {fmt_ms(t["p50"]):>8} '
                      f'{fmt_ms(t["p95"]):>8} {tp:>10.0f}/s')

    return results


# ============================================================
#  E. 连续吞吐量
# ============================================================
def bench_throughput(devices, hidden=128, n_elem=1024, n_samples=1000):
    print('\n[E] 连续推理吞吐量')
    print('-' * 70)

    results = {}
    for dev_name, dev in devices:
        model = DeepSetsModel(9, hidden, 2).to(dev)
        model.eval()
        x = torch.randn(1, n_elem, 9, device=dev)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        sync(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_samples):
                _ = model(x)
        sync(dev)
        t1 = time.perf_counter()

        total = (t1 - t0) * 1000
        per = total / n_samples
        tp = n_samples / (t1 - t0)
        results[dev_name] = {
            'n_samples': n_samples,
            'total_ms': total,
            'per_sample_ms': per,
            'throughput_sps': tp,
        }
        print(f'  {dev_name}: {n_samples} samples in {total:.1f}ms '
              f'= {per:.3f}ms/sample = {tp:.0f} samples/sec')

    return results


# ============================================================
#  F. 数据传输
# ============================================================
def bench_transfer(devices):
    if not any('npu' in str(d) for _, d in devices):
        return {}

    print('\n[F] 数据传输开销 (CPU <-> NPU)')
    print('-' * 70)

    dev = torch.device('npu:0')
    sizes = {
        '1K': 1024,
        '9K (单样本输入)': 1024 * 9,
        '147K (batch16输入)': 16 * 1024 * 9,
        '2M (batch16特征)': 16 * 1024 * 128,
        '16M (大张量)': 16 * 1024 * 1024,
    }

    results = {}
    for name, sz in sizes.items():
        data_cpu = torch.randn(sz)
        h2d = timer(lambda: data_cpu.to(dev), dev, n_iter=20)
        d_npu = torch.randn(sz, device=dev)
        d2h = timer(lambda: d_npu.to('cpu'), dev, n_iter=20)
        results[name] = {'h2d': h2d, 'd2h': d2h}
        print(f'  {name}: H2D={fmt_ms(h2d["mean"])} D2H={fmt_ms(d2h["mean"])} '
              f'total={fmt_ms(h2d["mean"]+d2h["mean"])}')

    return results


# ============================================================
#  G. 内存占用
# ============================================================
def bench_memory(devices, hidden=128, bs=16, n_elem=1024):
    print('\n[G] 内存占用')
    print('-' * 70)

    results = {}
    for dev_name, dev in devices:
        model = DeepSetsModel(9, hidden, 2).to(dev)
        x = torch.randn(bs, n_elem, 9, device=dev)
        y = torch.randn(bs, n_elem, 2, device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()

        # 训练一步
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        sync(dev)

        param_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
        io_mb = (x.nelement()*x.element_size() + y.nelement()*y.element_size()) / 1e6

        alloc_mb, reserved_mb = 0, 0
        if 'npu' in str(dev):
            try:
                alloc_mb = torch.npu.memory_allocated() / 1e6
                reserved_mb = torch.npu.memory_reserved() / 1e6
            except:
                pass

        results[dev_name] = {
            'param_mb': param_mb,
            'io_mb': io_mb,
            'alloc_mb': alloc_mb,
            'reserved_mb': reserved_mb,
        }
        print(f'  {dev_name}: params={param_mb:.2f}MB I/O={io_mb:.2f}MB '
              f'alloc={alloc_mb:.1f}MB reserved={reserved_mb:.1f}MB')

    return results


# ============================================================
#  主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式(减少迭代)')
    args = parser.parse_args()

    n_iter = 10 if args.quick else 30
    batch_sizes = [1, 8, 16] if args.quick else [1, 4, 8, 16, 32, 64]
    hidden_sizes = [128, 256] if args.quick else [64, 128, 256, 512]
    elem_sizes = [1024] if args.quick else [256, 512, 1024, 2048, 4096]

    print('=' * 80)
    print('CPU/NPU 标准基准程序')
    print(f'迭代次数: warmup=5, benchmark={n_iter}')
    print(f'Batch sizes: {batch_sizes}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Element sizes: {elem_sizes}')
    print('=' * 80)

    # 设备列表
    devices = [('CPU', torch.device('cpu'))]
    try:
        import torch_npu
        if torch.npu.is_available():
            devices.append(('NPU', torch.device('npu:0')))
    except ImportError:
        pass

    report = {}
    report['config'] = {
        'n_warmup': 5, 'n_benchmark': n_iter,
        'batch_sizes': batch_sizes, 'hidden_sizes': hidden_sizes,
        'elem_sizes': elem_sizes,
    }

    # A. 硬件
    report['hardware'] = collect_hardware()
    hw = report['hardware']
    print(f'\n[A] 硬件信息')
    print(f'  CPU: {hw["cpu"].get("implementer","?")} part={hw["cpu"].get("part","?")}'
          f' {hw["cpu"]["cores"]}cores {hw["cpu"].get("mhz","?")}MHz')
    if hw['npu'].get('available'):
        print(f'  NPU: {hw["npu"]["name"]} '
              f'{hw["npu"]["total_memory_mb"]//1024}GB HBM '
              f'cube={hw["npu"]["cube_cores"]} vector={hw["npu"]["vector_cores"]}')
    print(f'  PyTorch: {torch.__version__}')

    # B. 精度一致性
    report['consistency'] = check_consistency(devices)

    # C. 训练速度
    report['train'] = bench_train(devices, batch_sizes, hidden_sizes)

    # D. 推理速度
    report['infer'] = bench_infer(devices, batch_sizes, elem_sizes)

    # E. 吞吐量
    report['throughput'] = bench_throughput(devices)

    # F. 数据传输
    report['transfer'] = bench_transfer(devices)

    # G. 内存
    report['memory'] = bench_memory(devices)

    # ============================================================
    #  汇总报告
    # ============================================================
    print('\n' + '=' * 80)
    print('基准报告汇总')
    print('=' * 80)

    # 训练加速比表
    print('\n训练加速比:')
    print(f'{"hidden":>8} {"batch":>6} {"CPU(ms)":>10} {"NPU(ms)":>10} {"加速比":>8}')
    print('-' * 50)
    for h in hidden_sizes:
        for bs in batch_sizes:
            cpu_key = f'h{h}_bs{bs}_CPU'
            npu_key = f'h{h}_bs{bs}_NPU'
            if cpu_key in report['train'] and npu_key in report['train']:
                ct = report['train'][cpu_key]['mean']
                nt = report['train'][npu_key]['mean']
                print(f'{h:>8} {bs:>6} {ct:>10.2f} {nt:>10.2f} {ct/nt:>7.1f}x')

    # 推理加速比表
    print(f'\n推理加速比 (128维):')
    print(f'{"n_elem":>8} {"batch":>6} {"CPU(ms)":>10} {"NPU(ms)":>10} {"加速比":>8} {"NPU吞吐":>12}')
    print('-' * 62)
    for ne in elem_sizes:
        for bs in batch_sizes:
            cpu_key = f'ne{ne}_bs{bs}_CPU'
            npu_key = f'ne{ne}_bs{bs}_NPU'
            if cpu_key in report['infer'] and npu_key in report['infer']:
                ct = report['infer'][cpu_key]['mean']
                nt = report['infer'][npu_key]['mean']
                tp = report['infer'][npu_key]['throughput']
                print(f'{ne:>8} {bs:>6} {ct:>10.2f} {nt:>10.2f} {ct/nt:>7.1f}x {tp:>10.0f}/s')

    # 端到端对比
    print(f'\n端到端对比 (128维, batch=16, 1024阵元):')
    for dev_name in ['CPU', 'NPU']:
        if dev_name in report['throughput']:
            t = report['throughput'][dev_name]
            print(f'  {dev_name}: {t["per_sample_ms"]:.3f}ms/sample, {t["throughput_sps"]:.0f} samples/sec')
    if 'NPU' in report['throughput'] and 'CPU' in report['throughput']:
        sp = report['throughput']['CPU']['throughput_sps'] / report['throughput']['NPU']['throughput_sps']
        print(f'  吞吐量提升: {report["throughput"]["NPU"]["throughput_sps"]/report["throughput"]["CPU"]["throughput_sps"]:.1f}x')

    # vs SOCP
    if 'NPU' in report['throughput']:
        npu_infer = report['throughput']['NPU']['per_sample_ms']
        print(f'\n  vs SOCP (13000ms): NPU推理{npu_infer:.2f}ms = {13000/npu_infer:.0f}x加速')

    # 保存
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n完整报告已保存: {OUTPUT}')
    print(f'文件大小: {os.path.getsize(OUTPUT)/1024:.0f} KB')
    print('=' * 80)


if __name__ == '__main__':
    main()
