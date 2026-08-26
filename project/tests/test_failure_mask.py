"""失效mask正确性回归测试。

确保:
  1. 失效阵元权值严格为零
  2. 活跃阵元权值保持不变
  3. 失效数量与mask一致
  4. 整数索引 vs 布尔mask不混淆
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_failure_mask_boolean():
    """~运算符只能用于布尔mask，不能用于整数索引。"""
    fmask = np.zeros(10, dtype=bool)
    fmask[[2, 5, 8]] = True

    active_mask = ~fmask  # 布尔，正确
    assert active_mask.dtype == bool
    assert np.sum(active_mask) == 7

    # 整数索引不能用~
    active_idx = np.where(~fmask)[0]
    assert active_idx.dtype == np.intp
    assert len(active_idx) == 7

    # ~active_idx 是按位取反，不是补集
    neg = ~active_idx
    assert np.all(neg < 0), "~active_idx should be negative (bitwise NOT)"

    # 正确的补集
    failed_idx = np.where(fmask)[0]
    assert len(failed_idx) == 3
    assert set(failed_idx.tolist()) == {2, 5, 8}

    print("PASS: test_failure_mask_boolean")


def test_weight_zeroing():
    """失效阵元权值为零，活跃阵元保持。"""
    Nx = Ny = 32
    n_fail = int(Nx * Ny * 0.10)
    rng = np.random.RandomState(42)

    fmask = np.zeros(Nx * Ny, dtype=bool)
    fmask[rng.choice(Nx * Ny, n_fail, replace=False)] = True

    w = np.ones(Nx * Ny, dtype=complex)

    # 正确方法1: 用布尔mask
    w1 = w.copy()
    w1[fmask] = 0
    assert np.sum(w1[fmask]) == 0, "failed elements should be zero"
    assert np.sum(w1[~fmask]) == np.sum(~fmask), "active elements should be preserved"
    assert np.sum(w1 != 0) == (Nx * Ny - n_fail)

    # 正确方法2: 用整数索引
    w2 = w.copy()
    failed_idx = np.where(fmask)[0]
    w2[failed_idx] = 0
    assert np.array_equal(w1, w2), "both methods should give same result"

    # 错误方法: ~active_idx (按位取反)
    w3 = w.copy()
    active_idx = np.where(~fmask)[0]
    w3[~active_idx] = 0  # BUG
    assert not np.array_equal(w3, w1), "~active_idx is wrong and should differ"
    assert np.sum(w3 != 0) != (Nx * Ny - n_fail), "wrong count"

    print("PASS: test_weight_zeroing")


def test_socp_weight_reconstruction():
    """SOCP权值重建: w_full[active_idx] = w_opt 只设活跃阵元。"""
    Nx = Ny = 32
    rng = np.random.RandomState(42)
    n_fail = 51
    fmask = np.zeros(Nx * Ny, dtype=bool)
    fmask[rng.choice(Nx * Ny, n_fail, replace=False)] = True
    active_idx = np.where(~fmask)[0]

    n_active = len(active_idx)
    assert n_active == Nx * Ny - n_fail

    w_opt = np.ones(n_active, dtype=complex) * 0.5

    # 正确重建
    w_full = np.zeros(Nx * Ny, dtype=complex)
    w_full[active_idx] = w_opt

    assert np.sum(w_full[fmask]) == 0, "failed elements must be zero"
    assert np.allclose(w_full[active_idx], 0.5), "active elements must be set"
    assert np.sum(w_full != 0) == n_active

    print("PASS: test_socp_weight_reconstruction")


def test_failure_count_consistency():
    """失效数量在所有步骤中保持一致。"""
    for rate in [0.05, 0.10, 0.20]:
        Nx = Ny = 32
        n_total = Nx * Ny
        n_expected = int(n_total * rate)

        rng = np.random.RandomState(42)
        fmask = np.zeros(n_total, dtype=bool)
        fmask[rng.choice(n_total, n_expected, replace=False)] = True

        assert np.sum(fmask) == n_expected
        assert np.sum(~fmask) == n_total - n_expected

        active_idx = np.where(~fmask)[0]
        assert len(active_idx) == n_total - n_expected
        assert len(np.where(fmask)[0]) == n_expected

    print("PASS: test_failure_count_consistency")


if __name__ == '__main__':
    tests = [
        test_failure_mask_boolean,
        test_weight_zeroing,
        test_socp_weight_reconstruction,
        test_failure_count_consistency,
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
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("=== ALL TESTS PASSED ===")
