"""Stage 2B Difference-axis and legality tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mylib.antenna_calc import (
    beam_steering_phase_2d,
    combine_2d_excitation,
    taylor_2d_separable,
    uniform_linear_array_pos,
)
from mylib.official_evaluator import evaluate_difference_beam, evaluate_official_case
from mylib.sum_diff import (
    bayliss_2d_separable,
    capon_nulling_difference_2d,
    difference_axis_to_phi_deg,
    difference_null_is_legal,
)


def _reference(theta0, phi0, difference_axis, nx=32, ny=32):
    posx = uniform_linear_array_pos(nx)
    posy = uniform_linear_array_pos(ny)
    amp_x, amp_y, _ = bayliss_2d_separable(
        nx, ny, 35, difference_axis=difference_axis)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp, phase = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
    return posx, posy, amp, phase


def test_azimuth_difference_legacy_regression():
    """Default azimuth construction remains numerically unchanged."""
    old_amp_x, old_amp_y, old_split = bayliss_2d_separable(32, 32, 35)
    new_amp_x, new_amp_y, new_split = bayliss_2d_separable(
        32, 32, 35, difference_axis="azimuth")
    assert np.array_equal(old_amp_x, new_amp_x)
    assert np.array_equal(old_amp_y, new_amp_y)
    assert old_split == new_split

    posx, posy, amp, phase = _reference(30.0, 40.0, "azimuth", 8, 8)
    nulls = [(35.0, 90.0), (40.0, 180.0), (45.0, 270.0), (50.0, 45.0)]
    old_out = capon_nulling_difference_2d(
        posx, posy, amp, phase, 30.0, 40.0, nulls)
    new_out = capon_nulling_difference_2d(
        posx, posy, amp, phase, 30.0, 40.0, nulls,
        difference_axis="azimuth")
    assert np.array_equal(old_out[0], new_out[0])
    assert np.array_equal(old_out[1], new_out[1])


def test_elevation_difference_broadside_symmetry():
    """At broadside, y-difference is the transpose of x-difference."""
    _, _, az_amp, az_phase = _reference(0.0, 0.0, "azimuth")
    _, _, el_amp, el_phase = _reference(0.0, 0.0, "elevation")
    assert np.allclose(el_amp, az_amp.T, atol=1e-12)
    assert np.allclose(el_phase, az_phase.T, atol=1e-12)

    posx = uniform_linear_array_pos(32)
    posy = uniform_linear_array_pos(32)
    az = evaluate_difference_beam(
        az_amp, az_phase, posx, posy, 0.0, 0.0,
        difference_axis_phi_deg=0.0, n_uv=101)
    el = evaluate_difference_beam(
        el_amp, el_phase, posx, posy, 0.0, 0.0,
        difference_axis_phi_deg=90.0, n_uv=101)
    assert abs(az["sll_db"] - el["sll_db"]) < 1e-10
    assert az["pointing_error_deg"] < 1e-8
    assert el["pointing_error_deg"] < 1e-8


def test_elevation_intrinsic_and_adaptive_nulls():
    theta0, phi0 = 30.0, 40.0
    posx, posy, diff_amp, diff_phase = _reference(
        theta0, phi0, "elevation")
    sum_x, sum_y = taylor_2d_separable(32, 32, 35)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
    sum_amp, sum_phase = combine_2d_excitation(
        sum_x, sum_y, phase_x, phase_y)
    nulls = [(35.0, 90.0), (40.0, 180.0), (45.0, 270.0), (50.0, 45.0)]
    adaptive_amp, adaptive_phase = capon_nulling_difference_2d(
        posx, posy, diff_amp, diff_phase, theta0, phi0, nulls,
        difference_axis="elevation")
    result = evaluate_official_case(
        sum_amp, sum_phase, posx, posy, theta0, phi0,
        amp_difference=adaptive_amp, phase_difference=adaptive_phase,
        difference_null_dirs=[(theta0, phi0)] + nulls,
        difference_axis_phi_deg=90.0, n_uv=101)
    centers = result["adaptive_null"]["difference"]["center_db"]
    assert result["difference"]["sll_db"] <= -20.0
    assert centers[0] <= -30.0
    assert all(value <= -50.0 for value in centers[1:])
    assert result["difference"]["pointing_error_deg"] <= (
        result["sum"]["beamwidth_3db_deg"] / 30.0)


def test_difference_axis_argument_validation():
    with pytest.raises(ValueError):
        difference_axis_to_phi_deg("diagonal")
    with pytest.raises(ValueError):
        bayliss_2d_separable(8, 8, 25, difference_axis="diagonal")
    posx, posy, amp, phase = _reference(20.0, 15.0, "azimuth", 8, 8)
    with pytest.raises(ValueError):
        capon_nulling_difference_2d(
            posx, posy, amp, phase, 20.0, 15.0, [],
            difference_axis="diagonal")


def test_difference_null_legality_uses_official_main_lobe():
    posx = uniform_linear_array_pos(32)
    posy = uniform_linear_array_pos(32)
    _, _, amp, phase = _reference(0.0, 0.0, "elevation")
    difference = evaluate_difference_beam(
        amp, phase, posx, posy, 0.0, 0.0,
        difference_axis_phi_deg=90.0, n_uv=101)
    main_lobe = difference["main_lobe"]
    assert not difference_null_is_legal(main_lobe, 0.0, 0.0)
    assert not difference_null_is_legal(main_lobe, 1.0, 90.0)
    assert difference_null_is_legal(main_lobe, 30.0, 90.0)
