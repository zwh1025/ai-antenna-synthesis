"""v3 收敛速度指标：副瓣逼近目标值 0.5 dB 以内所需 Epochs 与训练总耗时。

竞赛要求(方案第3页, 效率指标2a): "AI算法达到指定精度(如副瓣电平逼近
目标值0.5dB以内)所需的迭代次数或训练轮数(Epochs)以及训练总耗时"。

口径定义(与竞赛措辞对齐):
  - 目标值 := 本训练 run 的最终最优验证 SLL(收敛极限);
  - 达标   := 验证 SLL 进入 [最终最优值, 最终最优值+0.5dB] 区间,
              且此后不再离开超过 0.5 dB;
  - 记录   := 达标 epoch、总训练耗时、对照 Stage5A 的 90%-gain epoch。

训练配置与 run_planar_fix_train.py 完全一致(SEED 456, 400 样本,
hidden 256, batch 16, Adam 1e-3, 最多 300 epochs); 每 20 epochs 在
固定曲面测试子集(10 样本)上评估 SLL。

输出: outputs/convergence_v3.json
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.train import EarlyStopping
from run_curved_verify import eval_dense_3d
from run_deepsets_train import _get_null_dirs
from run_planar_fix_train import (
    build_dataset, make_features, COORD_NORM, SLL_NORM, WEIGHT_SCALE,
    EPOCHS, BATCH, LR, HIDDEN, N_PLANAR, SEED,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
EVAL_EVERY = 20
EVAL_SUBSET = 10
CONV_TOL_DB = 0.5


def eval_curved_subset(model, data, idx):
    model.eval()
    slls = []
    for i in idx:
        px, py, pz = data['px'][i], data['py'][i], data['pz'][i]
        w_re, w_im = data['w_taylor_re'][i], data['w_taylor_im'][i]
        feat = make_features(px, py, pz, w_re, w_im,
                             data['u0'][i], data['v0'][i], data['w0'][i])
        with torch.no_grad():
            delta = model(torch.as_tensor(feat[None]))[0].numpy()
        w_ai = (w_re + 1j * w_im) + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE
        th0, ph0 = float(data['theta0'][i]), float(data['phi0'][i])
        sll, _, _, _ = eval_dense_3d(w_ai, px, py, pz, th0, ph0,
                                     _get_null_dirs(th0, ph0))
        slls.append(sll)
    return float(np.mean(slls))


def main():
    print('=' * 74, flush=True)
    print('v3 Convergence Tracking (SLL within 0.5 dB of final best)', flush=True)
    print(f'config: SEED={SEED}, samples=400(280curved+{N_PLANAR}planar), '
          f'hidden={HIDDEN}, epochs<={EPOCHS}', flush=True)
    print('=' * 74, flush=True)

    data, n_curved = build_dataset()
    split = data['split']
    va_idx = np.where(split == 1)[0]
    te_idx = np.where(split == 2)[0]
    eval_idx = te_idx[:EVAL_SUBSET]

    def pack(idx):
        px, py, pz = data['px'][idx], data['py'][idx], data['pz'][idx]
        w_re, w_im = data['w_taylor_re'][idx], data['w_taylor_im'][idx]
        feats, targets = [], []
        for j in range(len(idx)):
            feats.append(make_features(px[j], py[j], pz[j], w_re[j], w_im[j],
                                       data['u0'][idx[j]], data['v0'][idx[j]],
                                       data['w0'][idx[j]]))
            d_re = (data['w_socp_re'][idx][j] - w_re[j]) * WEIGHT_SCALE
            d_im = (data['w_socp_im'][idx][j] - w_im[j]) * WEIGHT_SCALE
            targets.append(np.stack([d_re, d_im], axis=-1).astype(np.float32))
        return np.array(feats), np.array(targets)

    tr_idx = np.where(split == 0)[0]
    tr_f, tr_y = pack(tr_idx)
    va_f, va_y = pack(va_idx)

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=8)
    stop = EarlyStopping(patience=30)
    crit = nn.MSELoss()

    ds = TensorDataset(torch.as_tensor(tr_f), torch.as_tensor(tr_y))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    vx = torch.as_tensor(va_f)
    vy = torch.as_tensor(va_y)

    curve = []
    t_train_start = time.time()
    reached_epoch = None
    best_val = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(vx), vy).item()
        sched.step(vl)
        if vl < best_val:
            best_val = vl

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            t_now = time.time() - t_train_start
            sll = eval_curved_subset(model, data, eval_idx)
            curve.append({'epoch': epoch + 1, 'val_loss': float(vl),
                          'curved_sll_mean_db': sll,
                          'elapsed_s': float(t_now)})
            print(f'  ep{epoch + 1:3d}: val_loss={vl:.6f} '
                  f'curved_SLL={sll:.2f} dB ({t_now:.0f}s)', flush=True)

        if stop.step(vl):
            print(f'  early stop at ep{epoch + 1}', flush=True)
            break

    t_total = time.time() - t_train_start

    # 最终再评估一次
    final_sll = eval_curved_subset(model, data, eval_idx)
    best_sll = min(c['curved_sll_mean_db'] for c in curve + [
        {'curved_sll_mean_db': final_sll}])

    # 达标定义: SLL 进入 best+0.5dB 内且此后不再离开
    in_band = False
    for c in curve:
        if not in_band and c['curved_sll_mean_db'] <= best_sll + CONV_TOL_DB:
            in_band = True
            reached_epoch = c['epoch']
            reached_time = c['elapsed_s']
        elif in_band and c['curved_sll_mean_db'] > best_sll + CONV_TOL_DB:
            in_band = False
            reached_epoch = None  # 离开区间则重置

    result = {
        'description': 'v3 convergence: epochs/time to reach SLL within '
                       f'{CONV_TOL_DB} dB of final best (competition 2a)',
        'config': {'seed': SEED, 'epochs_max': EPOCHS, 'batch': BATCH,
                   'lr': LR, 'hidden': HIDDEN,
                   'samples': {'curved': int(n_curved),
                               'planar': N_PLANAR},
                   'eval_every': EVAL_EVERY,
                   'eval_subset': EVAL_SUBSET,
                   'params': count_parameters(model)},
        'final_curved_sll_mean_db': final_sll,
        'best_curved_sll_mean_db': best_sll,
        'epoch_to_0.5dB_of_best': reached_epoch,
        'time_to_0.5dB_of_best_s': (curve[[c['epoch'] for c in curve]
                                          .index(reached_epoch)]['elapsed_s']
                                     if reached_epoch else None),
        'total_train_time_s': float(t_total),
        'early_stopped': bool(stop.should_stop),
        'sll_curve': curve,
        'stage5a_reference': {
            'fixed_task_90pct_gain_epoch': {'4p': 70, '6p': 55, '8p': 55},
            'fixed_task_runtime_s': {'4p': 106.7, '6p': 133.4, '8p': 152.7},
            'note': 'Stage5A为固定曲面任务、不同训练协议, 仅作参照',
        },
        'inference_convergence': {
            'iterations_per_synthesis': 1,
            'note': 'AI推理为单次前向, 无迭代收敛过程',
        },
    }

    print('\n' + '=' * 74)
    print(f'  最终曲面SLL(子集均值): {final_sll:.2f} dB '
          f'(历史最优 {best_sll:.2f})')
    ep_str = f'epoch {reached_epoch}' if reached_epoch else '未在曲线采样点内稳定进入'
    print(f'  进入最优值+0.5dB: {ep_str}')
    print(f'  总训练耗时: {t_total:.0f}s ({EPOCHS} epochs 上限)')
    print('=' * 74)

    with open(os.path.join(OUTPUT_DIR, 'convergence_v3.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'保存: {OUTPUT_DIR}/convergence_v3.json')


if __name__ == '__main__':
    main()
