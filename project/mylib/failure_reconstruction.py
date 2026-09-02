"""Failure-aware active-element weight reconstruction primitives.

The module is deliberately small and deterministic.  It implements the
Stage 4B baselines without touching the frozen evaluator or any Stage 2
algorithm:

* B0: mask the original complex weights, with no reconstruction;
* B1: mask then rescale active weights to the original l2 norm;
* B2: field-fitting active-element constrained least-squares correction,
  followed by the same
  original-norm power policy.

The steering constraints use arbitrary active-element coordinates, so a
failure mask produces an irregular active aperture rather than a shortened
uniform array.
"""

from __future__ import annotations

import numpy as np


FAILURE_RECONSTRUCTION_VERSION = "1.0.0"
FIELD_SAMPLE_THETA = 61
FIELD_SAMPLE_PHI = 121
REFERENCE_RIDGE_RATIO = 0.05


def complex_weights(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    amp = np.asarray(amp, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    if amp.shape != phase.shape:
        raise ValueError("amp and phase must have the same shape")
    return amp * np.exp(1j * phase)


def amp_phase(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.complex128)
    return np.abs(weights), np.mod(np.angle(weights), 2.0 * np.pi)


def _validate_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"failure mask shape {mask.shape} != weights shape {shape}")
    if not np.any(~mask):
        raise ValueError("at least one active element is required")
    return mask


def no_reconstruction(amp: np.ndarray, phase: np.ndarray,
                      failure_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """B0: exact masked original weights; no normalization or redesign."""
    w0 = complex_weights(amp, phase)
    mask = _validate_mask(failure_mask, w0.shape)
    w = w0.copy()
    w[mask] = 0.0
    return (*amp_phase(w), {
        "method": "B0_no_reconstruction",
        "solver_status": "MASKED_REFERENCE",
        "failed_count": int(mask.sum()),
        "active_count": int((~mask).sum()),
        "power_norm_reference": float(np.linalg.norm(w0)),
        "power_norm_before": float(np.linalg.norm(w)),
        "power_norm_after": float(np.linalg.norm(w)),
        "power_scale": 1.0,
    })


def active_renormalization(amp: np.ndarray, phase: np.ndarray,
                           failure_mask: np.ndarray):
    """B1: retain active relative weights and restore original l2 norm."""
    w0 = complex_weights(amp, phase)
    mask = _validate_mask(failure_mask, w0.shape)
    w = w0.copy()
    w[mask] = 0.0
    reference_norm = float(np.linalg.norm(w0))
    before_norm = float(np.linalg.norm(w))
    if before_norm <= 0.0:
        raise ValueError("masked reference has zero active power")
    scale = reference_norm / before_norm
    w *= scale
    return (*amp_phase(w), {
        "method": "B1_active_renormalization",
        "solver_status": "CLOSED_FORM",
        "failed_count": int(mask.sum()),
        "active_count": int((~mask).sum()),
        "power_norm_reference": reference_norm,
        "power_norm_before": before_norm,
        "power_norm_after": float(np.linalg.norm(w)),
        "power_scale": float(scale),
    })


def _direction_uv(theta_deg: float, phi_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(float(theta_deg))
    phi = np.deg2rad(float(phi_deg))
    return float(np.sin(theta) * np.cos(phi)), float(np.sin(theta) * np.sin(phi))


def _steering(posx: np.ndarray, posy: np.ndarray, theta_deg: float,
              phi_deg: float, lamb: float) -> np.ndarray:
    u, v = _direction_uv(theta_deg, phi_deg)
    return np.exp(1j * (2.0 * np.pi / float(lamb)) *
                  (posx * u + posy * v))


def _field_matrix(posx: np.ndarray, posy: np.ndarray,
                  theta0_deg: float, phi0_deg: float,
                  lamb: float) -> np.ndarray:
    """Build the fixed visible-domain complex-field matrix used by B2."""
    del theta0_deg, phi0_deg
    theta = np.linspace(0.0, 90.0, FIELD_SAMPLE_THETA)
    phi = np.linspace(0.0, 360.0, FIELD_SAMPLE_PHI, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_rad = np.deg2rad(theta_grid)
    phi_rad = np.deg2rad(phi_grid)
    u = np.sin(theta_rad) * np.cos(phi_rad)
    v = np.sin(theta_rad) * np.sin(phi_rad)
    visible = u * u + v * v <= 1.0 + 1e-12
    u, v = u[visible], v[visible]
    phase = (2.0 * np.pi / float(lamb)) * (
        u[:, None] * posx[None, :] + v[:, None] * posy[None, :]
    )
    # a^H w is the conjugate of the field convention's a^T conj(w).
    return np.exp(-1j * phase)


def minimum_norm_active_lcmv(
    amp: np.ndarray,
    phase: np.ndarray,
    failure_mask: np.ndarray,
    posx_2d: np.ndarray,
    posy_2d: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    null_dirs: list | tuple,
    lamb: float = 1.0,
    beam_kind: str = "sum",
    field_gram: np.ndarray | None = None,
    field_rhs: np.ndarray | None = None,
    shared_system: dict | None = None,
):
    """B2 active-aperture field-fitting constrained least-squares correction.

    For Sum, the active weights minimize ``||w_A-w0_A||_2`` subject to a
    the ideal reference complex response at the target.  Difference uses the
    intrinsic target null plus zero response at the four frozen nulls.
    For Difference, the target intrinsic null plus the four frozen nulls are
    constrained to zero.  Failed entries are never optimization variables.
    In both cases the correction minimizes a fixed visible-domain ideal-field fitting
    objective plus a ridge penalty to the active original weights.  The
    resulting complex vector is scaled to the original ideal l2 norm (Policy
    A), so reconstruction cannot increase total excitation power.

    ``field_gram`` and ``field_rhs`` are optional deterministic caches for the
    fixed geometry.  ``shared_system`` lets Sum and Difference reuse the
    Cholesky factor for one mask; neither cache changes the mathematical
    solution.
    """
    amp = np.asarray(amp, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    posx_2d = np.asarray(posx_2d, dtype=np.float64)
    posy_2d = np.asarray(posy_2d, dtype=np.float64)
    if amp.shape != phase.shape or amp.shape != posx_2d.shape or amp.shape != posy_2d.shape:
        raise ValueError("amp, phase, and 2D coordinates must have identical shapes")
    if beam_kind not in {"sum", "difference"}:
        raise ValueError("beam_kind must be 'sum' or 'difference'")
    mask = _validate_mask(failure_mask, amp.shape)
    w0 = complex_weights(amp, phase).ravel()
    px = posx_2d.ravel()
    py = posy_2d.ravel()
    active = ~mask.ravel()
    active_px, active_py = px[active], py[active]
    active_reference = w0[active]

    directions = [(float(theta0_deg), float(phi0_deg))]
    if beam_kind == "difference":
        directions.extend((float(theta), float(phi)) for theta, phi in null_dirs)
    columns = [
        _steering(active_px, active_py, theta, phi, lamb)
        for theta, phi in directions
    ]
    constraints = np.column_stack(columns)
    desired = np.zeros(len(directions), dtype=np.complex128)
    if beam_kind == "sum":
        ideal_target = _steering(px, py, theta0_deg, phi0_deg, lamb)
        desired[0] = np.vdot(ideal_target, w0)

    constraint_before = constraints.conj().T @ active_reference - desired
    if field_gram is None:
        field_active = _field_matrix(active_px, active_py, theta0_deg, phi0_deg, lamb)
        field_full = _field_matrix(px, py, theta0_deg, phi0_deg, lamb)
        gram = field_active.conj().T @ field_active
        active_field_rhs = field_active.conj().T @ (field_full @ w0)
        field_sample_count = int(field_active.shape[0])
    else:
        field_gram = np.asarray(field_gram, dtype=np.complex128)
        if field_gram.shape != (len(w0), len(w0)):
            raise ValueError("field_gram must be square with one row per element")
        if field_rhs is None:
            field_rhs = field_gram @ w0
        field_rhs = np.asarray(field_rhs, dtype=np.complex128).ravel()
        if field_rhs.shape != (len(w0),):
            raise ValueError("field_rhs must have one entry per element")
        gram = field_gram[np.ix_(active, active)]
        active_field_rhs = field_rhs[active]
        field_sample_count = int(field_gram.shape[0])
    ridge = REFERENCE_RIDGE_RATIO * float(np.trace(gram).real) / len(active_px)
    system = gram + ridge * np.eye(len(active_px), dtype=np.complex128)
    if shared_system is not None:
        if shared_system.get("active_indices") is None:
            shared_system["active_indices"] = np.flatnonzero(active).copy()
            shared_system["system"] = system
            shared_system["cholesky"] = np.linalg.cholesky(system)
        elif not np.array_equal(shared_system["active_indices"], np.flatnonzero(active)):
            raise ValueError("shared_system was supplied for a different active set")
        lower = shared_system["cholesky"]

        def solve_system(rhs):
            return np.linalg.solve(
                lower.conj().T, np.linalg.solve(lower, rhs)
            )
    else:
        solve_system = lambda rhs: np.linalg.solve(system, rhs)
    solved_rhs = solve_system(
        np.column_stack((active_field_rhs + ridge * active_reference, constraints))
    )
    unconstrained = solved_rhs[:, 0]
    solved_constraints = solved_rhs[:, 1:]
    constraint_gram = constraints.conj().T @ solved_constraints
    lagrange, _, rank, singular_values = np.linalg.lstsq(
        constraint_gram, desired - constraints.conj().T @ unconstrained,
        rcond=1e-12,
    )
    active_reconstructed = unconstrained + solved_constraints @ lagrange
    constraint_after = constraints.conj().T @ active_reconstructed - desired
    weights = np.zeros_like(w0)
    weights[active] = active_reconstructed

    reference_norm = float(np.linalg.norm(w0))
    before_norm = float(np.linalg.norm(weights))
    if before_norm <= 0.0:
        raise ValueError("active LCMV produced zero weight norm")
    scale = reference_norm / before_norm
    weights *= scale
    amp_out, phase_out = amp_phase(weights.reshape(amp.shape))
    finite = bool(np.all(np.isfinite(weights.real)) and np.all(np.isfinite(weights.imag)))
    if not finite:
        raise FloatingPointError("active LCMV produced non-finite weights")
    return amp_out, phase_out, {
        "method": "B2_field_fit_active_lcmv",
        "beam_kind": beam_kind,
        "solver_status": "LSTSQ_CONSTRAINED",
        "failed_count": int(mask.sum()),
        "active_count": int(active.sum()),
        "constraint_count": int(len(directions)),
        "field_sample_count": field_sample_count,
        "reference_ridge_ratio": REFERENCE_RIDGE_RATIO,
        "reference_ridge_value": float(ridge),
        "constraint_rank": int(rank),
        "constraint_residual_before_l2": float(np.linalg.norm(constraint_before)),
        "constraint_residual_after_l2": float(np.linalg.norm(constraint_after)),
        "constraint_singular_values": [float(value) for value in singular_values],
        "power_norm_reference": reference_norm,
        "power_norm_before": before_norm,
        "power_norm_after": float(np.linalg.norm(weights)),
        "power_scale": float(scale),
        "finite": finite,
        "lambda": float(lamb),
        "constraint_definition": (
            "Sum: ideal target complex response preserved; frozen nulls remain in field-fit objective and are measured"
            if beam_kind == "sum" else
            "Difference: intrinsic target null=0 and frozen null responses=0"
        ),
    }


__all__ = [
    "FAILURE_RECONSTRUCTION_VERSION",
    "complex_weights",
    "amp_phase",
    "no_reconstruction",
    "active_renormalization",
    "minimum_norm_active_lcmv",
]
