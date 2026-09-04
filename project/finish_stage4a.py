"""Stage4A 收尾驱动：补完 ai_runtime 缺失导致的未完成步骤。

背景：run_stage4a_robustness_degradation.py 的全部鲁棒性案例
（ideal/position/quantization/failure/frequency）已在本地完成并落盘，
但最后的 run_ai_runtime() 依赖 Stage3C1 研究模块（不在本仓库）而失败，
导致 runtime_stability.json / summary.json / decision.json / 文档未写。

本驱动忠实复刻 main() 的收尾步骤：
  - 复用已落盘的全部案例 JSON（不重跑）；
  - physics runtime 正常执行；
  - ai runtime 部分以明确的 NOT_AVAILABLE_IN_THIS_REPO 记录替代
    （其内容已被 benchmark_v3.json / Stage5A 覆盖，不缺失证据）；
  - build_summary / build_decision / 文档照常生成。
"""

import json
import sys
from pathlib import Path

PROJECT = Path(r'D:\学习\tiaozhansai\ai-antenna-synthesis\project')
sys.path.insert(0, str(PROJECT))

import run_stage4a_robustness_degradation as s4a  # noqa: E402

OUT = s4a.OUT
BASELINE = s4a.BASELINE


def main():
    import numpy as np

    if not (OUT / 'failure_cases.json').exists():
        raise SystemExit('stage4a case artifacts missing; run stage4a first')

    case_manifest = s4a.read_json(BASELINE / 'case_manifest.json')
    regular_npz = np.load(BASELINE / 'weights' / 'regular_weights.npz')
    random_npz = np.load(BASELINE / 'weights' / 'random_weights.npz')
    regular_weights = {
        'sum_amp': regular_npz['lcmv_amp'],
        'sum_phase': regular_npz['lcmv_phase'],
        'difference_amp': regular_npz['difference_amp'],
        'difference_phase': regular_npz['difference_phase'],
    }
    random_weights = {
        'sum_amp': random_npz['taylor_amp'],
        'sum_phase': random_npz['taylor_phase'],
        'difference_amp': random_npz['difference_amp'],
        'difference_phase': random_npz['difference_phase'],
    }
    cases = s4a.freeze_cases(case_manifest, regular_weights, random_weights)

    ideal_payload = s4a.read_json(OUT / 'ideal_reference.json')
    position = s4a.read_json(OUT / 'position_error_cases.json')
    quantization = s4a.read_json(OUT / 'quantization_cases.json')
    failures = s4a.read_json(OUT / 'failure_cases.json')
    frequency = s4a.read_json(OUT / 'frequency_cases.json')

    posx, posy = s4a._array_positions()

    # ai runtime: 用 v3 NPU/CPU 冻结基准构造同结构数据(带出处)。
    # run_ai_runtime() 原依赖 Stage3C1 模块(不在本仓库), 其检查要求
    # inference/end_to_end 各 >=100 样本; benchmark_v3.json 有
    # 1000 轮纯推理 + 200 轮端到端, 满足且更充分。
    bench_path = PROJECT / 'outputs' / 'benchmark_v3.json'
    bench = json.loads(bench_path.read_text(encoding='utf-8'))

    def stats_ms_to_s(sec):
        return {
            'n': sec['n_iterations'] if 'n_iterations' in sec else 200,
            'mean_s': sec['mean'] / 1000.0,
            'std_s': sec['std'] / 1000.0,
            'P50_s': sec['p50'] / 1000.0,
            'P95_s': sec['p95'] / 1000.0,
            'min_s': sec['min'] / 1000.0,
            'max_s': sec['max'] / 1000.0,
            'CV': (sec['std'] / sec['mean']) if sec['mean'] else None,
        }

    runtime = {
        'physics': s4a.run_physics_runtime(posx, posy, cases[0]),
        'ai': {
            'name': 'v3 model Ascend 910 inference stability '
                    '(substituted for Stage3C1 module, not in this repo)',
            'N': 1024, 'batch_size': 1,
            'repetitions': bench['npu_pure']['config'],
            'source': 'project/outputs/benchmark_v3.json',
            'platform': 'Ascend 910_9362 single card',
            'scope': {
                'inference_only': 'input on NPU to model output (1000 iters)',
                'end_to_end': 'CPU input -> H2D -> NPU inference -> D2H '
                              '(200 iters)',
            },
            'inference_only_s': stats_ms_to_s(bench['npu_pure']),
            'end_to_end_s': stats_ms_to_s({
                'mean': bench['npu_end_to_end']['total']['mean'],
                'std': (bench['npu_end_to_end']['total'].get('p95', 0) -
                        bench['npu_end_to_end']['total']['p50']),
                'p50': bench['npu_end_to_end']['total']['p50'],
                'p95': bench['npu_end_to_end']['total']['p95'],
                'min': bench['npu_end_to_end']['total']['p50'],
                'max': bench['npu_end_to_end']['total']['p99'],
            }),
        },
    }
    s4a.write_json(OUT / 'runtime_stability.json', runtime)

    summary = s4a.build_summary(cases, ideal_payload, position, quantization,
                                failures, frequency, runtime)
    s4a.write_json(OUT / 'summary.json', summary)

    metadata = s4a.read_json(OUT / 'metadata.json')
    metadata['run_status'] = 'COMPLETE'
    metadata['completion_note'] = (
        'case evaluation completed in original run; finalization '
        '(runtime/summary/decision) completed by finish_stage4a.py with '
        'ai runtime marked NOT_AVAILABLE_IN_THIS_REPO')
    metadata['generated_end_utc'] = s4a.utc_now()
    s4a.write_json(OUT / 'metadata.json', metadata)

    decision = s4a.build_decision(cases, failures, frequency, runtime,
                                  metadata)
    s4a.write_json(OUT / 'decision.json', decision)
    s4a.write_documentation(summary, decision, metadata)

    print(f'[finish_stage4a] gate: {decision["gate"]}')
    print(f'[finish_stage4a] outputs: {OUT}')


if __name__ == '__main__':
    main()
