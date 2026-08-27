"""NPU vs CPU 详细性能对比基准。

全面测量：
  1. 不同batch_size (1/4/8/16/32/64) 下的训练和推理速度
  2. 不同阵元数 (256/512/1024/2048/4096) 的影响
  3. 各子模块耗时分解 (Phi/Pool/Rho/Output)
  4. 数据传输开销 (H2D/D2H)
  5. 内存占用对比
  6. 能效比（吞吐量）
"""

import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel, count_parameters

N_WARMUP = 5
N_BENCH = 30
OUTPUT = os.path.join(os.path.dirname(__file__), 'outputs', 'npu_cpu_detailed.json')

def sync(device):
    if 'npu' in str(device):
        torch.npu.synchronize()
    elif 'cuda' in str(device):
        torch.cuda.synchronize()

def bench(fn, device, n_warmup=N_WARMUP, n_bench=N_BENCH):
    """通用计时函数。"""
    for _ in range(n_warmup):
        fn()
        sync(device)
    times = []
    for _ in range(n_bench):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'std': float(arr.std()),
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    import torch_npu
    npu_ok = torch.npu.is_available()
    devices = [('CPU', torch.device('cpu'))]
    if npu_ok:
        devices.append(('NPU', torch.device('npu:0')))
    
    all_results = {}
    
    # ============================================================
    # 测试1: 不同batch_size
    # ============================================================
    print('=' * 90)
    print('Test 1: Batch Size Scaling (1024 elements, 128-dim model)')
    print('=' * 90)
    print(f'{"batch":>6} {"device":>5} {"train(ms)":>10} {"infer(ms)":>10} {"train_std":>10} {"infer_std":>10}')
    print('-' * 90)
    
    batch_sizes = [1, 4, 8, 16, 32, 64]
    n_elem = 1024
    hidden = 128
    
    r1 = {}
    for bs in batch_sizes:
        r1[bs] = {}
        for dev_name, dev in devices:
            model = DeepSetsModel(9, hidden, 2).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.MSELoss()
            x = torch.randn(bs, n_elem, 9, device=dev)
            y = torch.randn(bs, n_elem, 2, device=dev)
            xs = x[:1]
            
            def train_fn():
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
            
            def infer_fn():
                with torch.no_grad():
                    _ = model(xs)
            
            t = bench(train_fn, dev)
            i = bench(infer_fn, dev)
            r1[bs][dev_name] = {'train': t, 'infer': i}
            print(f'{bs:>6} {dev_name:>5} {t["mean"]:>10.2f} {i["mean"]:>10.2f} {t["std"]:>10.2f} {i["std"]:>10.2f}')
    
    all_results['batch_scaling'] = r1
    
    # ============================================================
    # 测试2: 不同阵元数
    # ============================================================
    print('\n' + '=' * 90)
    print('Test 2: Element Count Scaling (batch=16, 128-dim model)')
    print('=' * 90)
    print(f'{"N_elem":>8} {"device":>5} {"train(ms)":>10} {"infer(ms)":>10} {"throughput":>12}')
    print('-' * 90)
    
    elem_sizes = [256, 512, 1024, 2048, 4096]
    bs = 16
    
    r2 = {}
    for ne in elem_sizes:
        r2[ne] = {}
        for dev_name, dev in devices:
            model = DeepSetsModel(9, hidden, 2).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.MSELoss()
            x = torch.randn(bs, ne, 9, device=dev)
            y = torch.randn(bs, ne, 2, device=dev)
            xs = x[:1]
            
            def train_fn():
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
            
            def infer_fn():
                with torch.no_grad():
                    _ = model(xs)
            
            t = bench(train_fn, dev, n_bench=20)
            i = bench(infer_fn, dev, n_bench=20)
            throughput = bs / (t['mean'] / 1000)  # samples/sec
            r2[ne][dev_name] = {'train': t, 'infer': i, 'throughput': throughput}
            print(f'{ne:>8} {dev_name:>5} {t["mean"]:>10.2f} {i["mean"]:>10.2f} {throughput:>10.1f}/s')
    
    all_results['element_scaling'] = r2
    
    # ============================================================
    # 测试3: 不同模型维度
    # ============================================================
    print('\n' + '=' * 90)
    print('Test 3: Model Width Scaling (batch=16, 1024 elements)')
    print('=' * 90)
    print(f'{"hidden":>8} {"params":>10} {"device":>5} {"train(ms)":>10} {"infer(ms)":>10} {"speedup":>8}')
    print('-' * 90)
    
    hidden_sizes = [64, 128, 256, 512]
    
    r3 = {}
    for h in hidden_sizes:
        r3[h] = {}
        for dev_name, dev in devices:
            model = DeepSetsModel(9, h, 2).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.MSELoss()
            x = torch.randn(bs, n_elem, 9, device=dev)
            y = torch.randn(bs, n_elem, 2, device=dev)
            xs = x[:1]
            n_params = count_parameters(model)
            
            def train_fn():
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
            
            def infer_fn():
                with torch.no_grad():
                    _ = model(xs)
            
            t = bench(train_fn, dev)
            i = bench(infer_fn, dev)
            r3[h][dev_name] = {'train': t, 'infer': i, 'params': n_params}
            print(f'{h:>8} {n_params:>10,} {dev_name:>5} {t["mean"]:>10.2f} {i["mean"]:>10.2f}', end='')
            if dev_name == 'NPU':
                print()
            else:
                print(f' {"":>8}')
    
    all_results['model_scaling'] = r3
    
    # 打印加速比
    print('\n' + '=' * 90)
    print('Speedup Summary')
    print('=' * 90)
    print(f'{"hidden":>8} {"params":>10} {"CPU train":>12} {"NPU train":>12} {"train x":>8} {"CPU infer":>12} {"NPU infer":>12} {"infer x":>8}')
    print('-' * 90)
    for h in hidden_sizes:
        d = r3[h]
        cpu_t = d['CPU']['train']['mean']
        npu_t = d.get('NPU',{}).get('train',{}).get('mean',0)
        cpu_i = d['CPU']['infer']['mean']
        npu_i = d.get('NPU',{}).get('infer',{}).get('mean',0)
        params = d['CPU']['params']
        tx = cpu_t/npu_t if npu_t > 0 else 0
        ix = cpu_i/npu_i if npu_i > 0 else 0
        print(f'{h:>8} {params:>10,} {cpu_t:>10.2f}ms {npu_t:>10.2f}ms {tx:>7.1f}x {cpu_i:>10.2f}ms {npu_i:>10.2f}ms {ix:>7.1f}x')
    
    # ============================================================
    # 测试4: 子模块耗时分解 (128-dim, batch=16, 1024 elem)
    # ============================================================
    print('\n' + '=' * 90)
    print('Test 4: Sub-module Breakdown (128-dim, batch=16, 1024 elements)')
    print('=' * 90)
    
    h = 128
    r4 = {}
    for dev_name, dev in devices:
        model = DeepSetsModel(9, h, 2).to(dev)
        x = torch.randn(16, 1024, 9, device=dev)
        
        # Phi
        def phi_fn():
            _ = model.phi(x)
        
        # Phi + pool
        def pool_fn():
            h_out = model.phi(x)
            mp = h_out.mean(dim=1)
            xp = h_out.max(dim=1)[0]
            _ = torch.cat([mp, xp], dim=-1)
        
        # Full
        def full_fn():
            with torch.no_grad():
                _ = model(x)
        
        phi_t = bench(phi_fn, dev, n_bench=20)
        pool_t = bench(pool_fn, dev, n_bench=20)
        full_t = bench(full_fn, dev, n_bench=20)
        
        r4[dev_name] = {'phi': phi_t, 'phi_pool': pool_t, 'full_infer': full_t}
        
        print(f'\n  {dev_name}:')
        print(f'    Phi (per-element MLP):    {phi_t["mean"]:.3f} ms  (P50={phi_t["p50"]:.3f}, P95={phi_t["p95"]:.3f})')
        print(f'    Phi + Pool (mean+max):    {pool_t["mean"]:.3f} ms  (P50={pool_t["p50"]:.3f}, P95={pool_t["p95"]:.3f})')
        print(f'    Full inference:           {full_t["mean"]:.3f} ms  (P50={full_t["p50"]:.3f}, P95={full_t["p95"]:.3f})')
    
    all_results['module_breakdown'] = r4
    
    # ============================================================
    # 测试5: 数据传输开销
    # ============================================================
    if npu_ok:
        print('\n' + '=' * 90)
        print('Test 5: Data Transfer Overhead (CPU <-> NPU)')
        print('=' * 90)
        
        dev = torch.device('npu:0')
        sizes = [1024, 1024*9, 1024*128, 16*1024*9, 16*1024*128]
        
        r5 = {}
        for sz in sizes:
            data_cpu = torch.randn(sz)
            
            def h2d():
                _ = data_cpu.to(dev)
            
            d_npu = torch.randn(sz, device=dev)
            def d2h():
                _ = d_npu.to('cpu')
            
            h2d_t = bench(h2d, dev, n_bench=20)
            d2h_t = bench(d2h, dev, n_bench=20)
            r5[sz] = {'h2d': h2d_t, 'd2h': d2h_t}
            
            print(f'  size={sz:>10,}: H2D={h2d_t["mean"]:.3f}ms  D2H={d2h_t["mean"]:.3f}ms  total={h2d_t["mean"]+d2h_t["mean"]:.3f}ms')
        
        all_results['data_transfer'] = r5
    
    # ============================================================
    # 测试6: 内存占用
    # ============================================================
    print('\n' + '=' * 90)
    print('Test 6: Memory Usage')
    print('=' * 90)
    
    r6 = {}
    for dev_name, dev in devices:
        model = DeepSetsModel(9, 128, 2).to(dev)
        x = torch.randn(16, 1024, 9, device=dev)
        y = torch.randn(16, 1024, 2, device=dev)
        
        # 推理内存
        with torch.no_grad():
            _ = model(x)
        sync(dev)
        
        if 'npu' in str(dev):
            # NPU memory
            try:
                mem_alloc = torch.npu.memory_allocated() / 1024 / 1024
                mem_reserved = torch.npu.memory_reserved() / 1024 / 1024
            except:
                mem_alloc, mem_reserved = 0, 0
        else:
            mem_alloc = 0
            mem_reserved = 0
        
        # 模型参数内存
        param_mem = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        # 输入输出内存
        io_mem = (x.nelement() * x.element_size() + y.nelement() * y.element_size()) / 1024 / 1024
        
        r6[dev_name] = {
            'param_mb': param_mem,
            'io_mb': io_mem,
            'alloc_mb': mem_alloc,
            'reserved_mb': mem_reserved,
        }
        
        print(f'  {dev_name}: params={param_mem:.2f}MB  I/O={io_mem:.2f}MB  alloc={mem_alloc:.1f}MB  reserved={mem_reserved:.1f}MB')
    
    all_results['memory'] = r6
    
    # ============================================================
    # 测试7: 连续推理吞吐量 (模拟在线部署)
    # ============================================================
    print('\n' + '=' * 90)
    print('Test 7: Continuous Inference Throughput (simulating online deployment)')
    print('=' * 90)
    
    r7 = {}
    for dev_name, dev in devices:
        model = DeepSetsModel(9, 128, 2).to(dev)
        model.eval()
        x = torch.randn(1, 1024, 9, device=dev)
        
        # 连续1000次推理
        n_total = 1000
        sync(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_total):
                _ = model(x)
        sync(dev)
        t1 = time.perf_counter()
        
        total_ms = (t1 - t0) * 1000
        per_sample = total_ms / n_total
        throughput = n_total / (t1 - t0)
        
        r7[dev_name] = {
            'n_samples': n_total,
            'total_ms': total_ms,
            'per_sample_ms': per_sample,
            'throughput_sps': throughput,
        }
        
        print(f'  {dev_name}: {n_total} samples in {total_ms:.1f}ms = {per_sample:.3f}ms/sample = {throughput:.0f} samples/sec')
    
    all_results['throughput'] = r7
    
    # ============================================================
    # 最终汇总
    # ============================================================
    print('\n' + '=' * 90)
    print('FINAL SUMMARY')
    print('=' * 90)
    
    # 核心对比表
    for h in [128, 256]:
        d = r3[h]
        cpu_t = d['CPU']['train']['mean']
        npu_t = d.get('NPU',{}).get('train',{}).get('mean',0)
        cpu_i = d['CPU']['infer']['mean']
        npu_i = d.get('NPU',{}).get('infer',{}).get('mean',0)
        params = d['CPU']['params']
        
        # 全训练时间
        n_ep = 200
        n_batch = 13
        cpu_train_total = cpu_t * n_ep * n_batch / 1000
        npu_train_total = npu_t * n_ep * n_batch / 1000
        
        # vs SOCP
        socp_ms = 13000
        vs_socp = socp_ms / npu_i if npu_i > 0 else 0
        
        print(f'\n  {h}-dim ({params:,} params):')
        print(f'    Training:  CPU {cpu_t:.2f}ms/ep  ->  NPU {npu_t:.2f}ms/ep  =  {cpu_t/npu_t:.1f}x speedup')
        print(f'    Inference: CPU {cpu_i:.2f}ms     ->  NPU {npu_i:.2f}ms     =  {cpu_i/npu_i:.1f}x speedup')
        print(f'    Full train (200ep): CPU {cpu_train_total:.1f}s -> NPU {npu_train_total:.1f}s')
        print(f'    vs SOCP (13s): NPU inference {npu_i:.2f}ms = {vs_socp:.0f}x faster')
        
        # 吞吐量
        t7 = r7
        if dev_name in t7:
            for dn in ['CPU', 'NPU']:
                if dn in t7:
                    print(f'    Throughput ({dn}): {t7[dn]["throughput_sps"]:.0f} samples/sec')
    
    # 保存
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Detailed results saved: {OUTPUT}')
    print('=' * 90)


if __name__ == '__main__':
    main()
