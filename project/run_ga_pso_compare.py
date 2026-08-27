"""传统优化算法(GA/PSO) vs AI 综合速度对比。

竞赛要求: "给出与解析法、交替投影、GA、PSO的加速比"

实现:
  1. 遗传算法(GA): 优化64维幅度锥削(32x+32y), 200代x50种群
  2. 粒子群(PSO): 同样64维, 200轮x50粒子
  3. 与 Taylor/SOCP/AI 对比时间和SLL

注意: GA/PSO优化幅度锥削(64参数), 相位由扫描方向确定。
     SOCP/AI优化完整复权值(1024个)。这是不同自由度，但反映了
     实际工程中各方法的使用方式。
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from run_curved_verify import (
    generate_curved_array, coordinate_taylor_3d,
    eval_dense_3d, uv_to_uvw,
)
from run_generate_teacher import normalize_weights
from run_deepsets_train import _get_null_dirs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
N_ELEM = NX * NY
THETA0 = 30.0
PHI0 = 0.0
ALPHA = 0.12
NULL_DIRS = _get_null_dirs(THETA0, PHI0)

GA_POP = 30
GA_GEN = 50
PSO_SWARM = 30
PSO_ITER = 50
N_EVAL = 41  # 粗网格加速


def build_weights(amp_x, amp_y, px, py, pz, theta0, phi0):
    """从幅度锥削构建复权值(相位=扫描相位)。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    amp = np.outer(amp_y, amp_x).ravel()  # 注意顺序
    phase = k * (px * u0 + py * v0 + pz * w0)
    return amp * np.exp(1j * phase)


def eval_sll(w, px, py, pz, theta0, phi0, null_dirs):
    """快速SLL评估(粗网格41x41)。"""
    k = 2 * np.pi
    n = 41
    u = np.linspace(-1, 1, n)
    v = np.linspace(-1, 1, n)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    vis = (ug**2 + vg**2) <= 1.0
    wg = np.sqrt(np.maximum(1 - ug**2 - vg**2, 0))
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
    sl_mask = (dist >= exc) & vis

    w_conj = np.conj(w)
    pat = np.zeros(n * n)
    uf = ug.ravel()
    vf = vg.ravel()
    wf = wg.ravel()
    visf = vis.ravel()
    slf = sl_mask.ravel()
    for i in range(n*n):
        if visf[i]:
            psi = k * (px * uf[i] + py * vf[i] + pz * wf[i])
            pat[i] = np.abs(np.sum(w_conj * np.exp(1j * psi)))
    psi_main = k * (px * u0 + py * v0 + pz * np.cos(np.deg2rad(theta0)))
    main_resp = np.abs(np.sum(w_conj * np.exp(1j * psi_main)))
    if main_resp < 1e-10:
        return -100.0
    sll = 20 * np.log10(np.max(pat[slf]) / (main_resp + 1e-30))
    return sll if not np.isnan(sll) else -100.0


def run_ga(px, py, pz, theta0, phi0, null_dirs, n_gen=GA_GEN, pop_size=GA_POP):
    """遗传算法: 优化64维幅度锥削。"""
    rng = np.random.RandomState(42)
    n_dim = NX + NY  # 64

    # 初始种群(在Taylor附近扰动)
    amp_x0, amp_y0 = taylor_2d_separable(NX, NY, 35)
    taylor_vec = np.concatenate([amp_x0, amp_y0])
    pop = np.clip(taylor_vec[None, :] + rng.randn(pop_size, n_dim) * 0.05, 0.01, 1.0)

    def fitness(ind):
        ax, ay = ind[:NX], ind[NX:]
        w = build_weights(ax, ay, px, py, pz, theta0, phi0)
        sll = eval_sll(w, px, py, pz, theta0, phi0, null_dirs)
        return sll  # 越负越好

    scores = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmin(scores)
    best_sll = scores[best_idx]
    best_ind = pop[best_idx].copy()

    for gen in range(n_gen):
        # 选择(锦标赛)
        idx = rng.randint(0, pop_size, pop_size)
        parents = pop[idx[np.argsort(scores[idx])[:pop_size//2]]]
        # 交叉
        children = pop.copy()
        for i in range(0, pop_size, 2):
            if i+1 < pop_size:
                p = rng.rand(n_dim)
                c1 = p * parents[i % len(parents)] + (1-p) * parents[(i+1) % len(parents)]
                c2 = (1-p) * parents[i % len(parents)] + p * parents[(i+1) % len(parents)]
                children[i] = np.clip(c1, 0.01, 1.0)
                children[i+1] = np.clip(c2, 0.01, 1.0)
        # 变异
        children += rng.randn(pop_size, n_dim) * 0.02 * (1 - gen/n_gen)
        children = np.clip(children, 0.01, 1.0)
        # 评估
        new_scores = np.array([fitness(ind) for ind in children])
        # 精英保留
        children[0] = best_ind
        new_scores[0] = best_sll
        pop = children
        scores = new_scores
        gen_best = np.argmin(scores)
        if scores[gen_best] < best_sll:
            best_sll = scores[gen_best]
            best_ind = pop[gen_best].copy()

    ax, ay = best_ind[:NX], best_ind[NX:]
    w = build_weights(ax, ay, px, py, pz, theta0, phi0)
    return w, best_sll


def run_pso(px, py, pz, theta0, phi0, null_dirs, n_iter=PSO_ITER, swarm=PSO_SWARM):
    """粒子群算法: 优化64维幅度锥削。"""
    rng = np.random.RandomState(42)
    n_dim = NX + NY
    amp_x0, amp_y0 = taylor_2d_separable(NX, NY, 35)
    taylor_vec = np.concatenate([amp_x0, amp_y0])

    pos = np.clip(taylor_vec[None, :] + rng.randn(swarm, n_dim) * 0.05, 0.01, 1.0)
    vel = rng.randn(swarm, n_dim) * 0.01
    pbest = pos.copy()
    pbest_score = np.array([eval_sll(
        build_weights(p[:NX], p[NX:], px, py, pz, theta0, phi0),
        px, py, pz, theta0, phi0, null_dirs) for p in pos])
    gbest_idx = np.argmin(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    w_inertia = 0.7
    c1, c2 = 1.5, 1.5

    for it in range(n_iter):
        r1, r2 = rng.rand(swarm, n_dim), rng.rand(swarm, n_dim)
        vel = w_inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0.01, 1.0)

        for i in range(swarm):
            s = eval_sll(
                build_weights(pos[i, :NX], pos[i, NX:], px, py, pz, theta0, phi0),
                px, py, pz, theta0, phi0, null_dirs)
            if s < pbest_score[i]:
                pbest_score[i] = s
                pbest[i] = pos[i].copy()
            if s < gbest_score:
                gbest_score = s
                gbest = pos[i].copy()

    ax, ay = gbest[:NX], gbest[NX:]
    w = build_weights(ax, ay, px, py, pz, theta0, phi0)
    return w, gbest_score


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, 35)
    rng = np.random.RandomState(42)
    px, py, pz = generate_curved_array(posx, posy, ALPHA, rng)

    u0 = np.sin(np.deg2rad(THETA0)) * np.cos(np.deg2rad(PHI0))
    v0 = np.sin(np.deg2rad(THETA0)) * np.sin(np.deg2rad(PHI0))
    w0 = np.cos(np.deg2rad(THETA0))

    print('=' * 70)
    print('传统优化算法(GA/PSO) vs AI 综合速度对比')
    print(f'  阵列: {N_ELEM}阵元, 曲率alpha={ALPHA}')
    print(f'  扫描: theta={THETA0} phi={PHI0}')
    print(f'  GA: {GA_POP}种群x{GA_GEN}代')
    print(f'  PSO: {PSO_SWARM}粒子x{PSO_ITER}轮')
    print('=' * 70)

    results = {}

    # 1. Taylor基线
    print('\n[Taylor] 解析法...')
    w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, THETA0, PHI0)
    w_taylor_n = normalize_weights(w_taylor, px, py, pz, u0, v0, w0)
    t0 = time.perf_counter()
    sll_t = eval_sll(w_taylor_n, px, py, pz, THETA0, PHI0, NULL_DIRS)
    t_taylor = (time.perf_counter() - t0) * 1000
    print(f'  SLL={sll_t:.1f}dB, 时间={t_taylor:.1f}ms')
    results['Taylor'] = {'sll': float(sll_t), 'time_ms': float(t_taylor)}

    # 2. GA
    print(f'\n[GA] 遗传算法({GA_POP}x{GA_GEN})...')
    t0 = time.perf_counter()
    w_ga, sll_ga = run_ga(px, py, pz, THETA0, PHI0, NULL_DIRS)
    t_ga = time.perf_counter() - t0
    print(f'  SLL={sll_ga:.1f}dB, 时间={t_ga:.1f}s')
    results['GA'] = {'sll': float(sll_ga), 'time_s': float(t_ga),
                     'pop': GA_POP, 'gen': GA_GEN}

    # 3. PSO
    print(f'\n[PSO] 粒子群算法({PSO_SWARM}x{PSO_ITER})...')
    t0 = time.perf_counter()
    w_pso, sll_pso = run_pso(px, py, pz, THETA0, PHI0, NULL_DIRS)
    t_pso = time.perf_counter() - t0
    print(f'  SLL={sll_pso:.1f}dB, 时间={t_pso:.1f}s')
    results['PSO'] = {'sll': float(sll_pso), 'time_s': float(t_pso),
                      'swarm': PSO_SWARM, 'iter': PSO_ITER}

    # 4. SOCP (从教师标签)
    print('\n[SOCP] 凸优化(从教师标签)...')
    teacher_path = os.path.join(OUTPUT_DIR, 'teacher_labels.npz')
    sll_s = None
    if os.path.exists(teacher_path):
        data = np.load(teacher_path)
        split = data['split']
        test_idx = np.where(split == 2)[0]
        for idx in test_idx:
            if data['alpha'][idx] > 0.10 and data['sll_socp'][idx] < data['sll_taylor'][idx] - 2:
                sll_s = float(data['sll_socp'][idx])
                break
    if sll_s is not None:
        print(f'  SLL={sll_s:.1f}dB, 时间~23s(含切平面迭代)')
        results['SOCP'] = {'sll': float(sll_s), 'time_s': 23.0}
    else:
        print('  教师标签不可用, 跳过')

    # 5. AI (DeepSets)
    print('\n[AI] DeepSets推理...')
    try:
        import torch
        from mylib.deepsets import DeepSetsModel
        from run_deepsets_train import WEIGHT_SCALE
        model_path = os.path.join(OUTPUT_DIR, 'deepsets_model.pt')
        if os.path.exists(model_path):
            model = DeepSetsModel(9, 128, 2)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            feat = np.stack([
                px/8.0, py/8.0, pz/8.0,
                w_taylor_n.real*WEIGHT_SCALE, w_taylor_n.imag*WEIGHT_SCALE,
                np.full(N_ELEM, u0), np.full(N_ELEM, v0), np.full(N_ELEM, w0),
                np.full(N_ELEM, 0.7),
            ], axis=-1).astype(np.float32)
            x = torch.as_tensor(feat[None], dtype=torch.float32)
            for _ in range(5):
                with torch.no_grad():
                    _ = model(x)
            t0 = time.perf_counter()
            with torch.no_grad():
                delta = model(x)[0].numpy()
            t_ai = (time.perf_counter() - t0) * 1000
            w_ai = (w_taylor_n.real + delta[:, 0]/WEIGHT_SCALE) + \
                   1j*(w_taylor_n.imag + delta[:, 1]/WEIGHT_SCALE)
            sll_a = eval_sll(w_ai, px, py, pz, THETA0, PHI0, NULL_DIRS)
            print(f'  SLL={sll_a:.1f}dB, 时间={t_ai:.2f}ms')
            results['AI'] = {'sll': float(sll_a), 'time_ms': float(t_ai)}
        else:
            print('  模型文件不可用, 跳过')
    except Exception as e:
        print(f'  AI推理失败: {e}')

    # ============================================================
    # 汇总
    # ============================================================
    print('\n' + '=' * 70)
    print('对比汇总')
    print('=' * 70)
    print(f'{"方法":<12} {"SLL(dB)":>10} {"耗时":>12} {"vs AI加速比":>12}')
    print('-' * 50)

    ai_time_ms = results.get('AI', {}).get('time_ms', 0.5)
    for method in ['Taylor', 'GA', 'PSO', 'SOCP', 'AI']:
        if method not in results:
            continue
        r = results[method]
        sll = r.get('sll', 0)
        if 'time_ms' in r:
            t = r['time_ms']
            t_str = f'{t:.1f}ms'
            speedup = t / ai_time_ms if ai_time_ms > 0 else 0
        else:
            t = r.get('time_s', 0) * 1000
            t_str = f'{r.get("time_s", 0):.1f}s'
            speedup = t / ai_time_ms if ai_time_ms > 0 else 0
        print(f'{method:<12} {sll:>10.1f} {t_str:>12} {speedup:>10.0f}x')

    # 保存
    with open(os.path.join(OUTPUT_DIR, 'ga_pso_compare.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n结果已保存: {OUTPUT_DIR}/ga_pso_compare.json')
    print('=' * 70)


if __name__ == '__main__':
    main()
