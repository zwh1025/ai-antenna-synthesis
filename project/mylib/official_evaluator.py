"""Canonical metric evaluator for the antenna-synthesis project.

This module freezes the Stage 1 measurement rules without changing any
synthesis algorithm.  The older helpers in :mod:`mylib.evaluation` and
:mod:`mylib.antenna_calc` remain available as legacy implementations; new
formal result producers should call :func:`evaluate_official_case`.

Coordinate convention
---------------------
``theta`` is the angle from +z (the broadside normal of the x-y array plane)
and ``phi`` is the azimuth from +x towards +y.  Angles in the public API are
degrees.  The implementation converts them to radians internally.  The
official domain is the upper hemisphere, ``theta in [0, 90]`` or equivalently
``u**2 + v**2 <= 1`` with ``u=sin(theta)cos(phi)`` and
``v=sin(theta)sin(phi)``.

The array factor follows the project's existing physical convention::

    F(u, v) = sum(conj(w_n) * exp(1j*k*(x_n*u + y_n*v)))

where ``w = amp * exp(1j*phase)``.  dBc is field-amplitude normalization,
``20*log10(abs(F)/max(abs(F)))``, which is exactly power normalization in dB.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label


OFFICIAL_EVALUATOR_VERSION = "1.0.0"
DEFAULT_GRID_SIZE = 201
DEFAULT_PROFILE_SAMPLES = 2001
DEFAULT_CUT_STEP_DEG = 0.05
DEFAULT_NULL_WINDOW_RADIUS_DEG = 3.0
DEFAULT_NULL_WINDOW_STEP_DEG = 0.25

SUM_SLL_THRESHOLD_DB = -35.0
DIFFERENCE_SLL_THRESHOLD_DB = -20.0
ADAPTIVE_NULL_THRESHOLD_DB = -30.0
STRICT_SUM_NULL_THRESHOLD_DB = -65.0
STRICT_DIFFERENCE_NULL_THRESHOLD_DB = -50.0


def direction_to_uv(theta_deg, phi_deg):
    """Convert project angles in degrees to direction cosines ``(u, v)``."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)


def uv_to_direction(u, v):
    """Convert visible ``(u, v)`` direction cosines to degrees."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64),
                               np.asarray(v, dtype=np.float64))
    rho = np.clip(np.sqrt(u * u + v * v), 0.0, 1.0)
    theta = np.rad2deg(np.arcsin(rho))
    phi = np.rad2deg(np.arctan2(v, u)) % 360.0
    return theta, phi


def angular_error_deg(theta_a, phi_a, theta_b, phi_b):
    """Return the true spherical angular separation in degrees."""
    ta, pa = np.deg2rad(theta_a), np.deg2rad(phi_a)
    tb, pb = np.deg2rad(theta_b), np.deg2rad(phi_b)
    cosine = (np.cos(ta) * np.cos(tb) +
              np.sin(ta) * np.sin(tb) * np.cos(pa - pb))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pointing_error_deg(theta_target, phi_target,
                       theta_estimate, phi_estimate):
    """Official per-case pointing error: spherical angular separation."""
    return angular_error_deg(theta_target, phi_target,
                             theta_estimate, phi_estimate)


def pointing_rmse_deg(errors_deg):
    """Official dataset pointing metric: RMS of per-case angular errors."""
    errors = np.asarray(errors_deg, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors ** 2)))


def pointing_threshold_deg(beamwidth_3db_deg):
    """Competition pointing threshold derived from the measured sum BW."""
    return float(beamwidth_3db_deg) / 30.0


def _flatten_positions(amp, posx, posy):
    amp = np.asarray(amp, dtype=np.float64)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    if amp.ndim != 2:
        raise ValueError("amp must be a 2D array")
    if posx.ndim == 1 and posy.ndim == 1:
        if amp.shape != (len(posx), len(posy)):
            raise ValueError("amp shape must equal (len(posx), len(posy))")
        px, py = np.meshgrid(posx, posy, indexing="ij")
    elif posx.shape == amp.shape and posy.shape == amp.shape:
        px, py = posx, posy
    else:
        raise ValueError("positions must be two 1D axes or two arrays matching amp")
    return amp, px.ravel(), py.ravel()


def array_factor_complex(amp, phase, posx, posy, u, v, lamb=1.0,
                         chunk_size=8192):
    """Evaluate the continuous complex array factor at arbitrary ``u, v``.

    The calculation is chunked so a 201x201 official grid does not create a
    multi-gigabyte ``(grid, element)`` temporary.  This is an evaluator
    utility only; it is not used by synthesis or training.
    """
    amp, px, py = _flatten_positions(amp, posx, posy)
    phase = np.asarray(phase, dtype=np.float64)
    if phase.shape != amp.shape:
        raise ValueError("phase shape must equal amp shape")
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64),
                               np.asarray(v, dtype=np.float64))
    flat_u, flat_v = u.ravel(), v.ravel()
    weights_conj = np.conj((amp * np.exp(1j * phase)).ravel())
    k = 2.0 * np.pi / float(lamb)
    result = np.empty(flat_u.size, dtype=np.complex128)
    for start in range(0, flat_u.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat_u.size)
        exponent = 1j * k * (
            flat_u[start:stop, None] * px[None, :] +
            flat_v[start:stop, None] * py[None, :]
        )
        result[start:stop] = np.exp(exponent) @ weights_conj
    return result.reshape(u.shape)


def field_to_dbc(field, peak_abs=None):
    """Normalize a complex field to dBc with a stable -300 dBc floor."""
    field = np.asarray(field)
    if peak_abs is None:
        peak_abs = float(np.max(np.abs(field)))
    peak_abs = float(peak_abs)
    if not np.isfinite(peak_abs) or peak_abs <= 0.0:
        raise ValueError("field peak must be finite and positive")
    ratio = np.abs(field) / peak_abs
    return 20.0 * np.log10(np.maximum(ratio, 10.0 ** (-300.0 / 20.0)))


def _pattern_grid(amp, phase, posx, posy, n_uv=DEFAULT_GRID_SIZE, lamb=1.0):
    if int(n_uv) < 5:
        raise ValueError("n_uv must be at least 5")
    axis = np.linspace(-1.0, 1.0, int(n_uv))
    u_grid, v_grid = np.meshgrid(axis, axis, indexing="ij")
    visible = (u_grid * u_grid + v_grid * v_grid) <= 1.0 + 1e-12
    field = array_factor_complex(amp, phase, posx, posy,
                                 u_grid, v_grid, lamb=lamb)
    peak_abs = float(np.max(np.abs(field[visible])))
    pattern_db = field_to_dbc(field, peak_abs)
    pattern_db = np.where(visible, pattern_db, -300.0)
    return pattern_db, u_grid, v_grid, visible, peak_abs


def _bilinear_sample(pattern_db, u, v):
    """Bilinearly sample a uniform ``[-1, 1]`` uv pattern grid."""
    pattern_db = np.asarray(pattern_db, dtype=np.float64)
    if pattern_db.ndim != 2 or pattern_db.shape[0] != pattern_db.shape[1]:
        raise ValueError("pattern_db must be a square uniform uv grid")
    n = pattern_db.shape[0]
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64),
                               np.asarray(v, dtype=np.float64))
    step = 2.0 / (n - 1)
    fi = (u + 1.0) / step
    fj = (v + 1.0) / step
    valid = ((fi >= 0.0) & (fi <= n - 1) &
             (fj >= 0.0) & (fj <= n - 1) &
             (u * u + v * v <= 1.0 + 1e-12))
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 1)
    j0 = np.clip(np.floor(fj).astype(int), 0, n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    j1 = np.minimum(j0 + 1, n - 1)
    di = fi - i0
    dj = fj - j0
    value = ((1.0 - di) * (1.0 - dj) * pattern_db[i0, j0] +
             di * (1.0 - dj) * pattern_db[i1, j0] +
             (1.0 - di) * dj * pattern_db[i0, j1] +
             di * dj * pattern_db[i1, j1])
    return np.where(valid, value, -300.0)


def _first_null_boundary(coords, values, center_idx, direction):
    """Find a deterministic first-null boundary on a sampled profile.

    A first local minimum at least 3 dB below the profile peak is preferred.
    If no local minimum is sampled, the first -3 dB crossing is used as an
    explicit deterministic fallback and its method is returned in metadata.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    center_idx = int(center_idx)
    peak = float(values[center_idx])

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    indices = range(center_idx + direction,
                    len(values) - 1 if direction > 0 else 0,
                    direction)
    for idx in indices:
        left, right = idx - 1, idx + 1
        if not (0 <= left < len(values) and 0 <= right < len(values)):
            continue
        if min(values[left], values[idx], values[right]) <= -299.0:
            continue
        toward_null = values[idx] < values[idx - direction]
        away_from_null = values[idx] <= values[idx + direction]
        if toward_null and away_from_null:
            if values[idx] <= peak - 3.0:
                # Do not mistake a flat shoulder for a null when the profile
                # continues descending immediately after the plateau.
                plateau_end = idx
                while (0 <= plateau_end + direction < len(values) and
                       values[plateau_end + direction] == values[idx]):
                    plateau_end += direction
                next_idx = plateau_end + direction
                if (0 <= next_idx < len(values) and
                        values[next_idx] < values[idx]):
                    continue
                if next_idx < 0 or next_idx >= len(values):
                    return float(coords[idx]), "first_local_minimum"
                return float(coords[plateau_end]), "first_local_minimum"

    threshold = peak - 3.0
    for idx in indices:
        previous = idx - direction
        if not (0 <= previous < len(values)):
            continue
        if values[idx] > -299.0 and values[idx] <= threshold:
            return float(coords[idx]), "first_3db_crossing_fallback"

    valid = np.flatnonzero(values > -299.0)
    if valid.size == 0:
        return float(coords[center_idx]), "invalid_profile"
    boundary_idx = valid[-1] if direction > 0 else valid[0]
    return float(coords[boundary_idx]), "visible_boundary_fallback"


def _profile(pattern_db, center_u, center_v, axis_u, axis_v,
             n_samples=DEFAULT_PROFILE_SAMPLES):
    coords = np.linspace(-1.0, 1.0, int(n_samples))
    u = center_u + coords * axis_u
    v = center_v + coords * axis_v
    return coords, _bilinear_sample(pattern_db, u, v)


def _sum_main_lobe_mask(pattern_db, u_grid, v_grid, visible,
                        n_profile=DEFAULT_PROFILE_SAMPLES):
    peak_idx = np.unravel_index(
        np.argmax(np.where(visible, pattern_db, -300.0)),
        pattern_db.shape,
    )
    peak_u = float(u_grid[peak_idx])
    peak_v = float(v_grid[peak_idx])

    u_coords, u_values = _profile(pattern_db, peak_u, peak_v, 1.0, 0.0,
                                   n_profile)
    v_coords, v_values = _profile(pattern_db, peak_u, peak_v, 0.0, 1.0,
                                   n_profile)
    u_center = int(np.argmax(u_values))
    v_center = int(np.argmax(v_values))
    u_left, u_left_method = _first_null_boundary(
        u_coords, u_values, u_center, -1)
    u_right, u_right_method = _first_null_boundary(
        u_coords, u_values, u_center, +1)
    v_left, v_left_method = _first_null_boundary(
        v_coords, v_values, v_center, -1)
    v_right, v_right_method = _first_null_boundary(
        v_coords, v_values, v_center, +1)

    mask = (visible &
            (u_grid >= peak_u + u_left) & (u_grid <= peak_u + u_right) &
            (v_grid >= peak_v + v_left) & (v_grid <= peak_v + v_right))
    details = {
        "definition": "axis-aligned first-null envelope in uv",
        "peak_uv": [peak_u, peak_v],
        "u_bounds_relative": [u_left, u_right],
        "v_bounds_relative": [v_left, v_right],
        "boundary_methods": {
            "u_left": u_left_method, "u_right": u_right_method,
            "v_left": v_left_method, "v_right": v_right_method,
        },
    }
    return mask, details, peak_idx


def sum_main_lobe_mask(pattern_db, u_grid, v_grid, visible=None):
    """Public helper returning the official sum-beam first-null mask."""
    if visible is None:
        visible = (u_grid * u_grid + v_grid * v_grid) <= 1.0 + 1e-12
    return _sum_main_lobe_mask(pattern_db, u_grid, v_grid, visible)[:2]


def sum_sll_from_pattern_db(pattern_db, u_grid, v_grid, visible=None):
    """Compute official sum SLL from a normalized uniform uv pattern."""
    if visible is None:
        visible = (u_grid * u_grid + v_grid * v_grid) <= 1.0 + 1e-12
    main_mask, details, _ = _sum_main_lobe_mask(
        pattern_db, u_grid, v_grid, visible)
    sidelobe = visible & ~main_mask
    value = float(np.max(pattern_db[sidelobe])) if np.any(sidelobe) else float("nan")
    return value, details


def _difference_main_lobe_mask(pattern_db, u_grid, v_grid, theta0, phi0,
                               difference_axis_phi_deg=0.0, visible=None,
                               n_profile=DEFAULT_PROFILE_SAMPLES):
    if visible is None:
        visible = (u_grid * u_grid + v_grid * v_grid) <= 1.0 + 1e-12
    u0, v0 = direction_to_uv(theta0, phi0)
    u0, v0 = float(u0), float(v0)
    axis = np.deg2rad(float(difference_axis_phi_deg))
    du, dv = float(np.cos(axis)), float(np.sin(axis))
    tu, tv = -dv, du

    q, q_values = _profile(pattern_db, u0, v0, du, dv, n_profile)
    valid_positive = (q > 0.0) & (q_values > -299.0)
    valid_negative = (q < 0.0) & (q_values > -299.0)
    if not np.any(valid_positive) or not np.any(valid_negative):
        raise ValueError("difference profile does not contain two visible lobes")
    positive_indices = np.flatnonzero(valid_positive)
    negative_indices = np.flatnonzero(valid_negative)
    q_pos_idx = positive_indices[np.argmax(q_values[positive_indices])]
    q_neg_idx = negative_indices[np.argmax(q_values[negative_indices])]
    q_pos_outer, q_pos_method = _first_null_boundary(
        q, q_values, q_pos_idx, +1)
    q_neg_outer, q_neg_method = _first_null_boundary(
        q, q_values, q_neg_idx, -1)

    def transverse_bounds(q_peak):
        r, r_values = _profile(
            pattern_db, u0 + q_peak * du, v0 + q_peak * dv,
            tu, tv, n_profile)
        near_center = np.flatnonzero((np.abs(r) <= 0.25) & (r_values > -299.0))
        if near_center.size == 0:
            center_idx = int(np.argmax(r_values))
        else:
            center_idx = int(near_center[np.argmax(r_values[near_center])])
        left, left_method = _first_null_boundary(r, r_values, center_idx, -1)
        right, right_method = _first_null_boundary(r, r_values, center_idx, +1)
        return left, right, {
            "left": left_method, "right": right_method,
        }

    r_neg_left, r_neg_right, r_neg_methods = transverse_bounds(float(q[q_neg_idx]))
    r_pos_left, r_pos_right, r_pos_methods = transverse_bounds(float(q[q_pos_idx]))

    negative_mask = (
        (q_grid := (u_grid - u0) * du + (v_grid - v0) * dv) <= 0.0
    ) & (q_grid >= q_neg_outer)
    positive_mask = (q_grid >= 0.0) & (q_grid <= q_pos_outer)
    r_grid = (u_grid - u0) * tu + (v_grid - v0) * tv
    negative_mask &= (r_grid >= r_neg_left) & (r_grid <= r_neg_right)
    positive_mask &= (r_grid >= r_pos_left) & (r_grid <= r_pos_right)
    mask = visible & (negative_mask | positive_mask)
    details = {
        "definition": "two difference lobes bounded by first-null envelopes",
        "target_uv": [u0, v0],
        "difference_axis_phi_deg": float(difference_axis_phi_deg) % 360.0,
        "negative_lobe_q_bounds": [q_neg_outer, 0.0],
        "positive_lobe_q_bounds": [0.0, q_pos_outer],
        "negative_lobe_r_bounds": [r_neg_left, r_neg_right],
        "positive_lobe_r_bounds": [r_pos_left, r_pos_right],
        "boundary_methods": {
            "negative_q": q_neg_method,
            "positive_q": q_pos_method,
            "negative_r": r_neg_methods,
            "positive_r": r_pos_methods,
        },
    }
    return mask, details


def difference_main_lobe_mask(pattern_db, u_grid, v_grid, theta0, phi0,
                              difference_axis_phi_deg=0.0, visible=None):
    """Public helper returning the official two-lobe difference mask."""
    return _difference_main_lobe_mask(
        pattern_db, u_grid, v_grid, theta0, phi0,
        difference_axis_phi_deg, visible,
    )


def difference_sll_from_pattern_db(pattern_db, u_grid, v_grid, theta0, phi0,
                                   difference_axis_phi_deg=0.0, visible=None):
    """Compute official difference SLL without counting the center null."""
    if visible is None:
        visible = (u_grid * u_grid + v_grid * v_grid) <= 1.0 + 1e-12
    main_mask, details = difference_main_lobe_mask(
        pattern_db, u_grid, v_grid, theta0, phi0,
        difference_axis_phi_deg, visible,
    )
    sidelobe = visible & ~main_mask
    value = float(np.max(pattern_db[sidelobe])) if np.any(sidelobe) else float("nan")
    return value, details


def _legacy_connected_sll(pattern_db, visible):
    main = pattern_db > -30.0
    labels, n_labels = label(main, structure=np.ones((3, 3), dtype=int))
    if n_labels == 0:
        return float("nan")
    peak_idx = np.unravel_index(np.argmax(np.where(visible, pattern_db, -300.0)),
                                pattern_db.shape)
    peak_label = labels[peak_idx]
    if peak_label == 0:
        return float("nan")
    main_component = (labels == peak_label) & visible
    outside = visible & ~main_component
    return float(np.max(pattern_db[outside])) if np.any(outside) else float("nan")


def _legacy_3bw_sll(pattern_db, u_grid, v_grid, visible,
                    theta0, phi0, nx):
    u0, v0 = direction_to_uv(theta0, phi0)
    bw_approx = 0.886 * 2.0 / float(nx) * 180.0 / np.pi
    exclusion_deg = 3.0 * bw_approx / max(np.cos(np.deg2rad(theta0)), 0.1)
    dist_uv = np.sqrt((u_grid - float(u0)) ** 2 + (v_grid - float(v0)) ** 2)
    outside = visible & (dist_uv >= np.sin(np.deg2rad(exclusion_deg)))
    return float(np.max(pattern_db[outside])) if np.any(outside) else float("nan")


def _linear_crossing(x0, y0, x1, y1, threshold):
    if y1 == y0:
        return float((x0 + x1) / 2.0)
    fraction = (float(threshold) - float(y0)) / (float(y1) - float(y0))
    return float(x0 + fraction * (x1 - x0))


def _signed_great_circle(alpha_deg, phi0_deg):
    """Map signed elevation in the phi0 great-circle plane to theta/phi."""
    alpha = np.asarray(alpha_deg, dtype=np.float64)
    theta = np.abs(alpha)
    phi = np.where(alpha >= 0.0, float(phi0_deg), float(phi0_deg) + 180.0)
    return theta, phi % 360.0


def _sum_beamwidth(amp, phase, posx, posy, theta0, phi0,
                   peak_abs, lamb=1.0, step_deg=DEFAULT_CUT_STEP_DEG):
    alpha = np.arange(-90.0, 90.0 + 0.5 * step_deg, step_deg)
    theta, phi = _signed_great_circle(alpha, phi0)
    u, v = direction_to_uv(theta, phi)
    field = array_factor_complex(amp, phase, posx, posy, u, v, lamb=lamb)
    values = field_to_dbc(field, peak_abs)
    near_target = np.abs(alpha - float(theta0)) <= 15.0
    if not np.any(near_target):
        return float("nan"), {}
    candidates = np.flatnonzero(near_target)
    peak_idx = int(candidates[np.argmax(values[candidates])])
    threshold = float(values[peak_idx] - 3.0)

    left_cross = None
    for idx in range(peak_idx - 1, -1, -1):
        if values[idx] <= threshold < values[idx + 1]:
            left_cross = _linear_crossing(
                alpha[idx], values[idx], alpha[idx + 1], values[idx + 1], threshold)
            break
    right_cross = None
    for idx in range(peak_idx, len(alpha) - 1):
        if values[idx] >= threshold > values[idx + 1]:
            right_cross = _linear_crossing(
                alpha[idx], values[idx], alpha[idx + 1], values[idx + 1], threshold)
            break
    if left_cross is None or right_cross is None:
        return float("nan"), {
            "cut": "signed great-circle at fixed phi0",
            "step_deg": float(step_deg),
            "peak_alpha_deg": float(alpha[peak_idx]),
            "crossing_status": "missing crossing",
        }
    return float(right_cross - left_cross), {
        "cut": "signed great-circle at fixed phi0",
        "step_deg": float(step_deg),
        "peak_alpha_deg": float(alpha[peak_idx]),
        "left_crossing_alpha_deg": float(left_cross),
        "right_crossing_alpha_deg": float(right_cross),
        "interpolation": "linear in dB",
    }


def _null_window_directions(theta0, phi0, radius_deg, step_deg):
    offsets = np.arange(-float(radius_deg), float(radius_deg) +
                        0.5 * float(step_deg), float(step_deg))
    theta = float(theta0) + offsets[:, None]
    phi = float(phi0) + offsets[None, :]
    theta, phi = np.broadcast_arrays(theta, phi)
    valid = (theta >= 0.0) & (theta <= 90.0)
    # A spherical cap, not a rectangular angular box, defines the window.
    distance = np.empty_like(theta)
    target_u, target_v = direction_to_uv(theta0, phi0)
    u, v = direction_to_uv(theta, phi)
    target_w = np.cos(np.deg2rad(theta0))
    w = np.sqrt(np.clip(1.0 - u * u - v * v, 0.0, 1.0))
    dot = float(target_u) * u + float(target_v) * v + target_w * w
    distance[:] = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    valid &= distance <= float(radius_deg) + 1e-9
    return theta[valid], phi[valid] % 360.0


def evaluate_nulls(amp, phase, posx, posy, null_dirs, peak_abs,
                   lamb=1.0, window_radius_deg=DEFAULT_NULL_WINDOW_RADIUS_DEG,
                   window_step_deg=DEFAULT_NULL_WINDOW_STEP_DEG):
    """Evaluate exact center nulls and independent spherical-cap worst cases."""
    center_db = []
    window_worst_db = []
    targets = []
    details = []
    for theta_null, phi_null in list(null_dirs or []):
        un, vn = direction_to_uv(theta_null, phi_null)
        center_field = array_factor_complex(
            amp, phase, posx, posy, un, vn, lamb=lamb)
        center_value = float(field_to_dbc(center_field, peak_abs))
        window_theta, window_phi = _null_window_directions(
            theta_null, phi_null, window_radius_deg, window_step_deg)
        wu, wv = direction_to_uv(window_theta, window_phi)
        window_field = array_factor_complex(
            amp, phase, posx, posy, wu, wv, lamb=lamb)
        window_values = field_to_dbc(window_field, peak_abs)
        worst_value = float(np.max(window_values)) if window_values.size else float("nan")
        target = {"theta_deg": float(theta_null), "phi_deg": float(phi_null) % 360.0}
        targets.append(target)
        center_db.append(center_value)
        window_worst_db.append(worst_value)
        details.append({
            **target,
            "center_db": center_value,
            "window_worst_db": worst_value,
            "window_radius_deg": float(window_radius_deg),
            "window_step_deg": float(window_step_deg),
        })
    return {
        "targets": targets,
        "center_db": center_db,
        "window_worst_db": window_worst_db,
        "center_threshold_db": ADAPTIVE_NULL_THRESHOLD_DB,
        "strict_sum_center_threshold_db": STRICT_SUM_NULL_THRESHOLD_DB,
        "strict_difference_center_threshold_db": STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
        "strict_field": "center_db",
        "window_role": "robustness diagnostic; never substitutes for center_db",
        "details": details,
    }


def evaluate_sum_beam(amp, phase, posx, posy, theta0, phi0,
                      n_uv=DEFAULT_GRID_SIZE, lamb=1.0):
    """Evaluate one sum beam with the frozen official definitions."""
    pattern, u_grid, v_grid, visible, peak_abs = _pattern_grid(
        amp, phase, posx, posy, n_uv=n_uv, lamb=lamb)
    main_mask, lobe_details, peak_idx = _sum_main_lobe_mask(
        pattern, u_grid, v_grid, visible)
    sidelobe = visible & ~main_mask
    sll = (float(np.max(pattern[sidelobe]))
           if np.any(sidelobe) else float("nan"))
    peak_u = float(u_grid[peak_idx])
    peak_v = float(v_grid[peak_idx])
    peak_theta, peak_phi = uv_to_direction(peak_u, peak_v)
    bw, bw_details = _sum_beamwidth(
        amp, phase, posx, posy, theta0, phi0, peak_abs, lamb=lamb)
    return {
        "sll_db": float(sll),
        "sll_threshold_db": SUM_SLL_THRESHOLD_DB,
        "sll_definition": "visible-domain maximum outside sum first-null uv envelope",
        "beamwidth_3db_deg": float(bw),
        "beamwidth_definition": bw_details,
        "peak_direction": {
            "theta_deg": float(np.asarray(peak_theta)),
            "phi_deg": float(np.asarray(peak_phi)) % 360.0,
        },
        "pointing_error_deg": pointing_error_deg(
            theta0, phi0, float(np.asarray(peak_theta)), float(np.asarray(peak_phi))),
        "pointing_threshold_deg": (pointing_threshold_deg(bw)
                                    if np.isfinite(bw) else float("nan")),
        "main_lobe": lobe_details,
        "diagnostic": {
            "sll_connected_db": _legacy_connected_sll(pattern, visible),
            "sll_3bw_db": _legacy_3bw_sll(
                pattern, u_grid, v_grid, visible, theta0, phi0,
                np.asarray(amp).shape[0]),
            "grid_size": int(n_uv),
            "normalization": "visible-domain max field amplitude",
        },
        "_peak_abs": peak_abs,
    }


def _difference_zero_crossing(amp, phase, posx, posy, theta0, phi0,
                              difference_axis_phi_deg, peak_abs, lamb=1.0):
    u0, v0 = direction_to_uv(theta0, phi0)
    axis = np.deg2rad(float(difference_axis_phi_deg))
    du, dv = np.cos(axis), np.sin(axis)
    q = np.linspace(-0.25, 0.25, 1001)
    u = float(u0) + q * du
    v = float(v0) + q * dv
    visible = u * u + v * v <= 1.0 + 1e-12
    field = array_factor_complex(amp, phase, posx, posy, u, v, lamb=lamb)
    magnitude_sq = np.abs(field) ** 2
    magnitude_sq[~visible] = np.inf
    idx = int(np.argmin(magnitude_sq))
    q_est = float(q[idx])
    if 0 < idx < len(q) - 1 and np.isfinite(magnitude_sq[idx - 1:idx + 2]).all():
        y0, y1, y2 = magnitude_sq[idx - 1:idx + 2]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-20:
            q_est += 0.5 * (y0 - y2) / denom * (q[1] - q[0])
    u_est, v_est = float(u0) + q_est * du, float(v0) + q_est * dv
    theta_est, phi_est = uv_to_direction(u_est, v_est)
    return {
        "theta_deg": float(np.asarray(theta_est)),
        "phi_deg": float(np.asarray(phi_est)) % 360.0,
        "error_deg": pointing_error_deg(
            theta0, phi0, float(np.asarray(theta_est)), float(np.asarray(phi_est))),
        "search": "continuous uv line through target; 1001 samples + parabolic magnitude interpolation",
    }


def evaluate_difference_beam(amp, phase, posx, posy, theta0, phi0,
                             difference_axis_phi_deg=0.0,
                             n_uv=DEFAULT_GRID_SIZE, lamb=1.0):
    """Evaluate one two-lobe difference beam with the frozen definitions."""
    pattern, u_grid, v_grid, visible, peak_abs = _pattern_grid(
        amp, phase, posx, posy, n_uv=n_uv, lamb=lamb)
    sll, lobe_details = difference_sll_from_pattern_db(
        pattern, u_grid, v_grid, theta0, phi0,
        difference_axis_phi_deg, visible)
    zero_crossing = _difference_zero_crossing(
        amp, phase, posx, posy, theta0, phi0,
        difference_axis_phi_deg, peak_abs, lamb=lamb)
    return {
        "sll_db": float(sll),
        "sll_threshold_db": DIFFERENCE_SLL_THRESHOLD_DB,
        "sll_definition": "visible-domain maximum outside both difference-lobe first-null envelopes",
        "difference_axis_phi_deg": float(difference_axis_phi_deg) % 360.0,
        "zero_crossing_direction": {
            "theta_deg": zero_crossing["theta_deg"],
            "phi_deg": zero_crossing["phi_deg"],
        },
        "pointing_error_deg": zero_crossing["error_deg"],
        "main_lobe": lobe_details,
        "diagnostic": {
            "grid_size": int(n_uv),
            "normalization": "difference-beam visible-domain max field amplitude",
        },
        "_peak_abs": peak_abs,
    }


def evaluate_official_case(amp_sum, phase_sum, posx, posy, theta0, phi0,
                           null_dirs=None, amp_difference=None,
                           phase_difference=None,
                           difference_null_dirs=None,
                           difference_axis_phi_deg=0.0,
                           n_uv=DEFAULT_GRID_SIZE, lamb=1.0):
    """Evaluate a complete case and return the versioned official schema."""
    sum_result = evaluate_sum_beam(
        amp_sum, phase_sum, posx, posy, theta0, phi0,
        n_uv=n_uv, lamb=lamb)
    sum_peak_abs = sum_result.pop("_peak_abs")
    difference_result = None
    difference_peak_abs = None
    if amp_difference is not None or phase_difference is not None:
        if amp_difference is None or phase_difference is None:
            raise ValueError("amp_difference and phase_difference must be provided together")
        difference_result = evaluate_difference_beam(
            amp_difference, phase_difference, posx, posy, theta0, phi0,
            difference_axis_phi_deg=difference_axis_phi_deg,
            n_uv=n_uv, lamb=lamb)
        difference_peak_abs = difference_result.pop("_peak_abs")
        difference_result["pointing_threshold_deg"] = sum_result[
            "pointing_threshold_deg"
        ]

    adaptive_null = {
        "sum": evaluate_nulls(
            amp_sum, phase_sum, posx, posy, null_dirs, sum_peak_abs,
            lamb=lamb),
        "difference": None,
    }
    if difference_result is not None and difference_null_dirs is not None:
        adaptive_null["difference"] = evaluate_nulls(
            amp_difference, phase_difference, posx, posy,
            difference_null_dirs, difference_peak_abs,
            lamb=lamb,
        )
    return {
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "task": {
            "theta0_deg": float(theta0),
            "phi0_deg": float(phi0) % 360.0,
            "upper_hemisphere": True,
            "wavelength": float(lamb),
        },
        "sum": sum_result,
        "difference": difference_result,
        "adaptive_null": adaptive_null,
        "latency": {
            "inference_only": None,
            "end_to_end_synthesis": None,
            "optimizer_runtime": None,
        },
    }


def summarize_latency(samples_ms):
    """Return reproducible latency statistics for an already timed sample set."""
    samples = np.asarray(samples_ms, dtype=np.float64).ravel()
    if samples.size == 0 or not np.all(np.isfinite(samples)) or np.any(samples < 0.0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "n": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "std_ms": float(np.std(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


__all__ = [
    "OFFICIAL_EVALUATOR_VERSION",
    "DEFAULT_GRID_SIZE",
    "direction_to_uv",
    "uv_to_direction",
    "angular_error_deg",
    "pointing_error_deg",
    "pointing_rmse_deg",
    "pointing_threshold_deg",
    "array_factor_complex",
    "field_to_dbc",
    "sum_main_lobe_mask",
    "sum_sll_from_pattern_db",
    "difference_main_lobe_mask",
    "difference_sll_from_pattern_db",
    "evaluate_nulls",
    "evaluate_sum_beam",
    "evaluate_difference_beam",
    "evaluate_official_case",
    "summarize_latency",
]
