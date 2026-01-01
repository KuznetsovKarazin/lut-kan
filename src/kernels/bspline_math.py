# src/kernels/bspline_math.py
from __future__ import annotations

from typing import Literal

import numpy as np

BoundaryMode = Literal["half_open", "closed"]


def apply_base(x: np.ndarray, base_kind: str) -> np.ndarray:
    base_kind = (base_kind or "none").lower().strip()
    x = np.asarray(x, dtype=np.float32)
    if base_kind == "silu":
        return x / (1.0 + np.exp(-x))
    return x


def bspline_basis_all_numpy(x: np.ndarray, knots: np.ndarray, degree: int, boundary_mode: BoundaryMode) -> np.ndarray:
    """
    Cox–de Boor recursion for all basis functions N_{i,degree}(x) on a knot vector.

    knots: (M,)
    x: (N,)
    returns: (M-degree-1, N) == (coef_len, N)
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    t = np.asarray(knots, dtype=np.float32).reshape(-1)
    k = int(degree)

    M = int(t.shape[0])
    N = int(x.shape[0])
    if M < k + 2:
        raise ValueError(f"knots too short: M={M}, degree={k}")

    # degree 0 basis: (M-1, N)
    B = np.zeros((M - 1, N), dtype=np.float32)
    for i in range(M - 1):
        left = t[i]
        right = t[i + 1]
        if boundary_mode == "closed" and i == (M - 2):
            B[i, :] = ((x >= left) & (x <= right)).astype(np.float32)
        else:
            B[i, :] = ((x >= left) & (x < right)).astype(np.float32)

    # elevate degree to k
    for d in range(1, k + 1):
        Bn = np.zeros((M - 1 - d, N), dtype=np.float32)
        for i in range(M - 1 - d):
            denom1 = t[i + d] - t[i]
            denom2 = t[i + d + 1] - t[i + 1]

            if denom1 != 0.0:
                w1 = (x - t[i]) / denom1
                term1 = w1 * B[i, :]
            else:
                term1 = 0.0

            if denom2 != 0.0:
                w2 = (t[i + d + 1] - x) / denom2
                term2 = w2 * B[i + 1, :]
            else:
                term2 = 0.0

            Bn[i, :] = term1 + term2
        B = Bn

    # shape (M-k-1, N)
    return B


def coef2curve_numpy(
    x: np.ndarray,
    *,
    grid: np.ndarray,  # kept for API parity; not used
    coef: np.ndarray,
    degree: int,
    boundary_mode: BoundaryMode,
    knots_aug: np.ndarray,
) -> np.ndarray:
    """
    NumPy coef2curve matching PyKAN semantics on augmented knots:
      y(x) = sum_i coef[i] * N_{i,degree}(x; knots_aug)
    Domain is the full augmented knot vector (no clamping to interior).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    c = np.asarray(coef, dtype=np.float32).reshape(-1)
    t = np.asarray(knots_aug, dtype=np.float32).reshape(-1)
    k = int(degree)

    nbasis = int(t.shape[0] - k - 1)
    if c.shape[0] != nbasis:
        raise ValueError(f"coef_len={c.shape[0]} but nbasis={nbasis} from knots_len={t.shape[0]}, degree={k}")

    B = bspline_basis_all_numpy(x, t, k, boundary_mode)  # (nbasis, N)
    y = (c[None, :] @ B).reshape(-1).astype(np.float32)
    return y


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


if numba_available():
    import numba as nb

    @nb.njit(cache=True)
    def coef2curve_numba_1d(x: np.ndarray, coef: np.ndarray, knots: np.ndarray, degree: int, closed: bool) -> np.ndarray:
        """
        Numba version of Cox–de Boor "all-basis" recursion (correctness-first).
        """
        k = int(degree)
        M = knots.shape[0]
        N = x.shape[0]
        nbasis = M - k - 1
        if coef.shape[0] != nbasis:
            raise ValueError("coef_len mismatch nbasis")

        # degree-0 basis at one x: length M-1
        B0 = np.zeros(M - 1, dtype=np.float32)
        # work buffers for higher degrees
        # maximum length is M-1, decreasing each iteration
        Bprev = np.zeros(M - 1, dtype=np.float32)
        Bnext = np.zeros(M - 1, dtype=np.float32)

        out = np.zeros(N, dtype=np.float32)

        for n in range(N):
            xv = x[n]

            # init B0
            for i in range(M - 1):
                B0[i] = 0.0
                left = knots[i]
                right = knots[i + 1]
                if closed and i == (M - 2):
                    if xv >= left and xv <= right:
                        B0[i] = 1.0
                else:
                    if xv >= left and xv < right:
                        B0[i] = 1.0

            # copy into Bprev
            for i in range(M - 1):
                Bprev[i] = B0[i]

            # elevate degree
            length = M - 1
            for d in range(1, k + 1):
                new_len = length - 1
                for i in range(new_len):
                    denom1 = knots[i + d] - knots[i]
                    denom2 = knots[i + d + 1] - knots[i + 1]

                    term1 = 0.0
                    term2 = 0.0
                    if denom1 != 0.0:
                        term1 = (xv - knots[i]) / denom1 * Bprev[i]
                    if denom2 != 0.0:
                        term2 = (knots[i + d + 1] - xv) / denom2 * Bprev[i + 1]
                    Bnext[i] = term1 + term2

                # swap
                for i in range(new_len):
                    Bprev[i] = Bnext[i]
                length = new_len

            # dot with coef over nbasis = M-k-1 == length after k steps
            s = 0.0
            for i in range(nbasis):
                s += coef[i] * Bprev[i]
            out[n] = s

        return out
