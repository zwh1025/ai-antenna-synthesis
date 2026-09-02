"""Deterministic truth-table tests for the Stage 1 canonical evaluator."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mylib.official_evaluator import (
    array_factor_complex,
    difference_sll_from_pattern_db,
    evaluate_nulls,
    field_to_dbc,
    pointing_error_deg,
    pointing_rmse_deg,
    sum_sll_from_pattern_db,
    summarize_latency,
)


def _synthetic_uv_grid(n=201):
    axis = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(axis, axis, indexing="ij")


def test_power_normalization_is_scale_invariant():
    u, v = _synthetic_uv_grid(21)
    field = 2.5 * np.ones_like(u, dtype=np.complex128)
    assert np.allclose(field_to_dbc(field), 0.0)
    assert np.allclose(field_to_dbc(7.0 * field), 0.0)

    amp = np.ones((2, 2))
    phase = np.zeros((2, 2))
    pos = np.array([-0.25, 0.25])
    first = array_factor_complex(amp, phase, pos, pos, u, v)
    second = array_factor_complex(3.0 * amp, phase, pos, pos, u, v)
    assert np.allclose(field_to_dbc(first), field_to_dbc(second), atol=1e-12)


def test_sum_sll_excludes_known_first_null_main_lobe():
    u, v = _synthetic_uv_grid()
    pattern = np.full_like(u, -70.0, dtype=float)

    # A square main lobe with sampled first-null boundaries at |u|, |v|=.12.
    central = (np.abs(u) <= 0.08) & (np.abs(v) <= 0.08)
    shoulder = ((np.abs(u) <= 0.12) & (np.abs(v) <= 0.12)) & ~central
    pattern[central] = -1.5 * (u[central] ** 2 + v[central] ** 2) / (0.08 ** 2)
    pattern[shoulder] = -10.0

    # The official SLL must be this known sidelobe, not the main-lobe shoulder.
    pattern[(np.abs(u - 0.50) < 0.011) & (np.abs(v - 0.20) < 0.011)] = -17.0
    pattern[(np.abs(u + 0.50) < 0.011) & (np.abs(v - 0.20) < 0.011)] = -25.0

    sll, details = sum_sll_from_pattern_db(pattern, u, v)
    assert np.isclose(sll, -17.0)
    assert details["definition"] == "axis-aligned first-null envelope in uv"


def test_difference_sll_ignores_center_null_and_excludes_two_lobes():
    u, v = _synthetic_uv_grid()
    pattern = np.full_like(u, -80.0, dtype=float)

    left_lobe = (u >= -0.25) & (u <= -0.10) & (np.abs(v) <= 0.06)
    right_lobe = (u >= 0.10) & (u <= 0.25) & (np.abs(v) <= 0.06)
    pattern[left_lobe | right_lobe] = -1.0
    pattern[(np.abs(u - 0.55) < 0.011) & (np.abs(v) < 0.011)] = -18.0
    pattern[(np.abs(u + 0.55) < 0.011) & (np.abs(v) < 0.011)] = -24.0
    # Explicit center null: it must not become the reported SLL or a lobe.
    pattern[np.abs(u) < 0.011] = -300.0

    sll, details = difference_sll_from_pattern_db(
        pattern, u, v, theta0=0.0, phi0=0.0, difference_axis_phi_deg=0.0)
    assert np.isclose(sll, -18.0)
    assert details["definition"] == "two difference lobes bounded by first-null envelopes"


def test_null_center_is_continuous_and_window_reports_worst_maximum():
    pos = np.array([-0.5, 0.5])
    theta0, phi0 = 37.0, 23.0
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    a0 = np.exp(1j * 2.0 * np.pi * pos * u0)

    # Choose w so conj(w) @ a0 is exactly zero for the target direction.
    weights = np.array([1.0 + 0.0j, -np.conj(a0[0] / a0[1])])
    amp = np.abs(weights).reshape(2, 1) * np.ones((1, 2))
    phase = np.angle(weights).reshape(2, 1) * np.ones((1, 2))
    # Use an x-only 2-element array replicated in y with zero y spacing.
    posy = np.zeros(2)
    peak_field = np.max(np.abs(array_factor_complex(
        amp, phase, pos, posy, np.array([0.0]), np.array([0.0]))))
    result = evaluate_nulls(
        amp, phase, pos, posy, [(theta0, phi0)], peak_field,
        window_radius_deg=3.0, window_step_deg=0.25)

    assert result["center_db"][0] <= -250.0
    assert result["window_worst_db"][0] > result["center_db"][0]
    assert result["strict_field"] == "center_db"


def test_pointing_rmse_and_phi_seam_are_spherical():
    assert np.isclose(pointing_rmse_deg([0.0, 0.1, 0.2]), np.sqrt(0.05 / 3.0))
    assert pointing_error_deg(45.0, 179.0, 45.0, -179.0) < 2.0


def test_linear_latency_statistics_are_stable():
    stats = summarize_latency([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert np.isclose(stats["mean_ms"], 2.5)
    assert np.isclose(stats["std_ms"], np.std([1.0, 2.0, 3.0, 4.0]))
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 4.0
