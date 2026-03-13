
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

# -----------------------------------------------------------------------------
# Embedded legacy core (kept here so ltgp is the single source of truth)
# -----------------------------------------------------------------------------

import numpy as np
import cvxpy as cp
from numpy.linalg import norm
from scipy.linalg import eigh

# ==========================================
# Linear algebra / thermodynamics helpers
# ==========================================

def dagger(X): 
    return X.conj().T

def safe_eigvals(rho, tol=1e-12):
    # Hermitian eigenvalues (sorted descending); negatives clipped to 0
    w, _ = eigh((rho + dagger(rho)) / 2.0)
    w = np.real(w)
    w[w < 0] = 0.0
    return np.flip(np.sort(w))

def von_neumann_entropy(rho, tol=1e-12):
    w = safe_eigvals(rho, tol=tol)
    w = w[w > tol]
    if w.size == 0:
        return 0.0
    return -np.sum(w * np.log(w))

def gibbs_state(H, beta, tol=1e-12):
    w, U = eigh((H + dagger(H)) / 2.0)
    w = np.real(w)
    w = w - np.min(w)
    e = np.exp(-beta * w)
    Z = np.sum(e)
    p = e / Z
    return U @ np.diag(p) @ dagger(U)

def kron(A, B):
    return np.kron(A, B)

def partial_trace(rho, dims, keep=[0], tol=1e-12):
    # dims = (dA, dB), keep=[0] returns Tr_B; keep=[1] returns Tr_A
    dA, dB = dims
    rho = rho.reshape(dA, dB, dA, dB)
    if keep == [0]:
        out = np.zeros((dA, dA), dtype=complex)
        for i in range(dB):
            out += rho[:, i, :, i]
        return 0.5 * (out + dagger(out))
    if keep == [1]:
        out = np.zeros((dB, dB), dtype=complex)
        for i in range(dA):
            out += rho[i, :, i, :]
        return 0.5 * (out + dagger(out))
    raise ValueError("keep must be [0] or [1].")

def relative_entropy(rho, sigma, tol=1e-12):
    # D(rho||sigma) = Tr rho (log rho - log sigma)
    rho = 0.5 * (rho + dagger(rho))
    sigma = 0.5 * (sigma + dagger(sigma))
    wr, Ur = eigh(rho)
    ws, Us = eigh(sigma)
    wr = np.real(wr); ws = np.real(ws)
    wr[wr < tol] = tol
    ws[ws < tol] = tol
    log_rho = Ur @ np.diag(np.log(wr)) @ dagger(Ur)
    log_sig = Us @ np.diag(np.log(ws)) @ dagger(Us)
    return float(np.real(np.trace(rho @ (log_rho - log_sig))))

def mutual_information(rho, dims, tol=1e-12):
    rhoA = partial_trace(rho, dims, keep=[0], tol=tol)
    rhoB = partial_trace(rho, dims, keep=[1], tol=tol)
    return (
        von_neumann_entropy(rhoA, tol=tol)
        + von_neumann_entropy(rhoB, tol=tol)
        - von_neumann_entropy(rho, tol=tol)
    )

def relative_entropy_of_coherence(rho, H, tol=1e-12):
    # coherence wrt energy eigenbasis: S(Δ[rho]) - S(rho)
    w, U = eigh((H + dagger(H)) / 2.0)
    rho_e = dagger(U) @ rho @ U
    rho_deph = np.diag(np.diag(rho_e))
    rho_deph = U @ rho_deph @ dagger(U)
    return von_neumann_entropy(rho_deph, tol=tol) - von_neumann_entropy(rho, tol=tol)

def choi_identity(d):
    # Choi of identity channel in column-stacking convention: sum_{ij} |i><j| ⊗ |i><j|
    J = np.zeros((d*d, d*d), dtype=complex)
    for i in range(d):
        for j in range(d):
            eij = np.zeros((d, d), dtype=complex)
            eij[i, j] = 1.0
            J += np.kron(eij, eij)
    return J
# ==========================================
# Qubit Pauli helpers (used for LT families)
# ==========================================

def paulis():
    """Return (σx, σy, σz) as 2x2 complex arrays."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz

# ==========================================
# Core class: LTSDPSystem
# ==========================================

class LTSDPSystem:
    """
    Core SDP environment for locally thermal (LT) and Gibbs-preserving (GP) analysis.

    This class provides:
      - LT membership checks and thermodynamic monotones
      - Projection to LT (trace-distance)
      - Extremal LT samplers
      - Global GP channel feasibility (Choi SDP)
      - Local GP (two-step) heuristic feasibility and residual scores
      - Global GP channel extraction (returns a concrete Choi matrix)
    """

    def __init__(
        self,
        H_A,
        H_Ap,
        beta,
        solver="SCS",
        tol=1e-7,
        eps_eq_global=1e-8,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    ):
        self.H_A = np.array(H_A, dtype=complex)
        self.H_Ap = np.array(H_Ap, dtype=complex)
        self.beta = float(beta)

        self.dA = int(self.H_A.shape[0])
        self.dAp = int(self.H_Ap.shape[0])
        self.dims = (self.dA, self.dAp)

        self.gammaA = gibbs_state(self.H_A, self.beta)
        self.gammaAp = gibbs_state(self.H_Ap, self.beta)

        self.solver_default = solver
        self.tol_default = tol

        # Mapping tolerances (tau -> tau')
        self.eps_eq_global = eps_eq_global
        self.eps_eq_local = eps_eq_local

        # Gibbs-preserving tolerance (gamma -> gamma)
        self.eps_gibbs = eps_gibbs

    # --------- Basic LT stuff ---------

    def lt_membership(self, rho, tol=1e-8):
        dA, dAp = self.dims
        A = partial_trace(rho, self.dims, keep=[0], tol=tol)
        Ap = partial_trace(rho, self.dims, keep=[1], tol=tol)
        okA = np.allclose(A, self.gammaA, atol=tol)
        okAp = np.allclose(Ap, self.gammaAp, atol=tol)
        return okA and okAp, okA, okAp, A, Ap

    def monotones(self, rho, tol=1e-12):
        GAxGAp = kron(self.gammaA, self.gammaAp)
        D_rho = relative_entropy(rho, GAxGAp, tol=tol)
        I_rho = mutual_information(rho, self.dims, tol=tol)
        rho_A = partial_trace(rho, self.dims, keep=[0], tol=tol)
        rho_Ap = partial_trace(rho, self.dims, keep=[1], tol=tol)
        C_A = relative_entropy_of_coherence(rho_A, self.H_A, tol=tol)
        C_Ap = relative_entropy_of_coherence(rho_Ap, self.H_Ap, tol=tol)
        return D_rho, I_rho, C_A, C_Ap

    # --------- Internal helpers for GP SDPs ---------
    @staticmethod
    def trace_norm_hermitian(X, tol: float = 1e-12) -> float:
        """
        ||X||_1 for (approximately) Hermitian X via eigenvalues.
        """
        Xh = 0.5 * (X + X.conj().T)
        w, _ = eigh(Xh)
        w = np.real(w)
        w[np.abs(w) < tol] = 0.0
        return float(np.sum(np.abs(w)))

    def correlation_C(self, rho: np.ndarray) -> np.ndarray:
        """
        C := rho - gammaA⊗gammaAp (Hermitian symmetrized).
        On LT states, C has zero marginals.
        """
        GAxGAp = kron(self.gammaA, self.gammaAp)
        rho_h = 0.5 * (rho + rho.conj().T)
        C = rho_h - GAxGAp
        return 0.5 * (C + C.conj().T)

    def operator_schmidt_svals_C(self, rho: np.ndarray) -> np.ndarray:
        """
        Operator-Schmidt singular values of C across A|A':
          reshape C into (dA^2, dA'^2) and take svals.
        Useful as a vector signature of correlation structure.
        """
        dA, dAp = self.dims
        C = self.correlation_C(rho)
        M = C.reshape(dA * dA, dAp * dAp)
        svals = np.linalg.svd(M, compute_uv=False)
        return np.real_if_close(svals)

    def correlation_metrics(self, rho: np.ndarray, tol: float = 1e-12) -> dict:
        """
        Bundle of diagnostics for C.
        """
        C = self.correlation_C(rho)

        # zero-marginal checks (should be ~0 on LT states)
        CA = partial_trace(C, self.dims, keep=[0], tol=tol)
        CAp = partial_trace(C, self.dims, keep=[1], tol=tol)

        C_fro = float(norm(C, "fro"))
        C_tr  = 0.5 * self.trace_norm_hermitian(C, tol=tol)

        svals = self.operator_schmidt_svals_C(rho)
        s_top = svals[: min(6, len(svals))].copy()

        return {
            "C": C,
            "C_fro": C_fro,
            "C_trace_dist": C_tr,   # = 0.5||C||_1
            "C_marginalA_fro": float(norm(CA, "fro")),
            "C_marginalAp_fro": float(norm(CAp, "fro")),
            "C_svals": svals,
            "C_svals_top": s_top,
        }
    
    # ==========================================
    # LT family helpers (ray + qubit correlation tensor)
    # ==========================================
        # ==========================================
    # Verification + multistart for Local GP
    # ==========================================

    def _apply_kraus_on_A(self, rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
        """Apply Kraus {K} on subsystem A: ρ -> Σ (K⊗I) ρ (K†⊗I). No renormalization."""
        dA, dAp = self.dims
        IAp = np.eye(dAp, dtype=complex)
        rho_h = 0.5 * (rho + rho.conj().T)
        out = np.zeros_like(rho_h, dtype=complex)
        for K in kraus:
            Kext = np.kron(K, IAp)
            out += Kext @ rho_h @ Kext.conj().T
        return 0.5 * (out + out.conj().T)

    def _apply_kraus_on_Ap(self, rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
        """Apply Kraus {K} on subsystem A': ρ -> Σ (I⊗K) ρ (I⊗K†). No renormalization."""
        dA, dAp = self.dims
        IA = np.eye(dA, dtype=complex)
        rho_h = 0.5 * (rho + rho.conj().T)
        out = np.zeros_like(rho_h, dtype=complex)
        for K in kraus:
            Kext = np.kron(IA, K)
            out += Kext @ rho_h @ Kext.conj().T
        return 0.5 * (out + out.conj().T)

    # def apply_local_choi_A_no_norm(self, rho: np.ndarray, J_A: np.ndarray, tol_kraus: float = 1e-12) -> np.ndarray:
    #     dA, _ = self.dims
    #     kraus = self.kraus_from_choi(J_A, d_in=dA, d_out=dA, tol=tol_kraus)
    #     return self._apply_kraus_on_A(rho, kraus)

    # def apply_local_choi_Ap_no_norm(self, rho: np.ndarray, J_Ap: np.ndarray, tol_kraus: float = 1e-12) -> np.ndarray:
    #     _, dAp = self.dims
    #     kraus = self.kraus_from_choi(J_Ap, d_in=dAp, d_out=dAp, tol=tol_kraus)
    #     return self._apply_kraus_on_Ap(rho, kraus)

    # def apply_local_product_choi_no_norm(
    #     self, rho: np.ndarray, J_A: np.ndarray, J_Ap: np.ndarray, tol_kraus: float = 1e-12
    # ) -> np.ndarray:
    #     # maps commute; order doesn't matter
    #     out = self.apply_local_choi_A_no_norm(rho, J_A, tol_kraus=tol_kraus)
    #     out = self.apply_local_choi_Ap_no_norm(out, J_Ap, tol_kraus=tol_kraus)
    #     return out
    def apply_local_choi_A_no_norm(self, rho: np.ndarray, J_A: np.ndarray, tol_kraus: float = 1e-12) -> np.ndarray:
        return self.apply_local_choi_A_via_blocks(rho, J_A)

    def apply_local_choi_Ap_no_norm(self, rho: np.ndarray, J_Ap: np.ndarray, tol_kraus: float = 1e-12) -> np.ndarray:
        return self.apply_local_choi_Ap_via_blocks(rho, J_Ap)

    def apply_local_product_choi_no_norm(
        self, rho: np.ndarray, J_A: np.ndarray, J_Ap: np.ndarray, tol_kraus: float = 1e-12
    ) -> np.ndarray:
        out = self.apply_local_choi_A_via_blocks(rho, J_A)
        out = self.apply_local_choi_Ap_via_blocks(out, J_Ap)
        return 0.5 * (out + out.conj().T)

    def apply_local_choi_A_via_blocks(self, rho: np.ndarray, J_A: np.ndarray) -> np.ndarray:
        """Apply local channel on A using the direct block/Choi formula."""
        dA, dAp = self.dims
        rho_h = 0.5 * (rho + rho.conj().T)
        blocks = rho_h.reshape(dA, dAp, dA, dAp)
        out = np.zeros((dA * dAp, dA * dAp), dtype=complex)
        for i in range(dA):
            for j in range(dA):
                Eij = np.zeros((dA, dA), dtype=complex)
                Eij[i, j] = 1.0
                Phi_Eij = self.choi_apply_numpy(J_A, Eij, d_in=dA, d_out=dA)
                Tij = blocks[i, :, j, :]
                out += np.kron(Phi_Eij, Tij)
        return 0.5 * (out + out.conj().T)

    def apply_local_choi_Ap_via_blocks(self, rho: np.ndarray, J_Ap: np.ndarray) -> np.ndarray:
        """Apply local channel on A' using the direct block/Choi formula."""
        dA, dAp = self.dims
        rho_h = 0.5 * (rho + rho.conj().T)
        blocks = rho_h.reshape(dA, dAp, dA, dAp)
        out = np.zeros((dA * dAp, dA * dAp), dtype=complex)
        for a in range(dAp):
            for b in range(dAp):
                Eab = np.zeros((dAp, dAp), dtype=complex)
                Eab[a, b] = 1.0
                Phi_Eab = self.choi_apply_numpy(J_Ap, Eab, d_in=dAp, d_out=dAp)
                Xab = blocks[:, a, :, b]
                out += np.kron(Xab, Phi_Eab)
        return 0.5 * (out + out.conj().T)

    def apply_local_product_choi_via_blocks(self, rho: np.ndarray, J_A: np.ndarray, J_Ap: np.ndarray) -> np.ndarray:
        out = self.apply_local_choi_A_via_blocks(rho, J_A)
        out = self.apply_local_choi_Ap_via_blocks(out, J_Ap)
        return 0.5 * (out + out.conj().T)

    def _matrix_debug_report(self, X: np.ndarray, tol: float = 1e-12) -> dict:
        Xh = 0.5 * (X + X.conj().T)
        ev = np.linalg.eigvalsh(Xh)
        return {
            "trace": float(np.real(np.trace(Xh))),
            "trace_err": float(abs(np.trace(Xh) - 1.0)),
            "hermiticity_fro": float(norm(X - X.conj().T, "fro")),
            "min_eig": float(np.min(np.real(ev))),
            "max_eig": float(np.max(np.real(ev))),
            "fro_norm": float(norm(Xh, "fro")),
        }

    def state_debug_report(self, rho: np.ndarray, lt_tol: float = 1e-8, tol: float = 1e-12) -> dict:
        rho_h = 0.5 * (rho + rho.conj().T)
        lt_ok, okA, okAp, rhoA, rhoAp = self.lt_membership(rho_h, tol=lt_tol)
        D_rho, I_rho, C_A, C_Ap = self.monotones(rho_h, tol=tol)
        corr = self.correlation_metrics(rho_h, tol=tol)
        ppt = self.ppt_status(rho_h, tol=max(tol, 1e-10))
        rep = self._matrix_debug_report(rho_h, tol=tol)
        rep.update({
            "LT": bool(lt_ok),
            "LT_A": bool(okA),
            "LT_Ap": bool(okAp),
            "marginal_err_A": float(norm(rhoA - self.gammaA, "fro")),
            "marginal_err_Ap": float(norm(rhoAp - self.gammaAp, "fro")),
            "D": float(D_rho),
            "I": float(I_rho),
            "D_minus_I": float(D_rho - I_rho),
            "C_A": float(C_A),
            "C_Ap": float(C_Ap),
            "C_trace_dist": float(corr["C_trace_dist"]),
            "C_fro": float(corr["C_fro"]),
            "C_marginalA_fro": float(corr["C_marginalA_fro"]),
            "C_marginalAp_fro": float(corr["C_marginalAp_fro"]),
            "ppt": bool(ppt["is_ppt"]),
            "ppt_min_eig": float(ppt["min_pt_eig"]),
        })
        return rep

    def local_channel_application_debug(self, rho: np.ndarray, J_A: np.ndarray | None = None, J_Ap: np.ndarray | None = None) -> dict:
        out: dict[str, float | dict | None] = {}
        rho_h = 0.5 * (rho + rho.conj().T)
        if J_A is not None:
            JAh = 0.5 * (J_A + J_A.conj().T)
            via_kraus = self.apply_local_choi_A_no_norm(rho_h, JAh)
            via_blocks = self.apply_local_choi_A_via_blocks(rho_h, JAh)
            out["A_method_gap"] = float(norm(via_kraus - via_blocks, "fro"))
            out["A_out_report"] = self.state_debug_report(via_kraus)
            out["A_diag"] = self.choi_diagnostics(JAh, d_in=self.dA, d_out=self.dA, gamma_in=self.gammaA, gamma_out=self.gammaA)
            gamma_k = self.apply_local_choi_A_no_norm(np.kron(self.gammaA, self.gammaAp), JAh)
            out["A_gamma_marginal_err"] = float(norm(partial_trace(gamma_k, self.dims, keep=[0]) - self.gammaA, "fro"))
        if J_Ap is not None:
            JAp_h = 0.5 * (J_Ap + J_Ap.conj().T)
            via_kraus = self.apply_local_choi_Ap_no_norm(rho_h, JAp_h)
            via_blocks = self.apply_local_choi_Ap_via_blocks(rho_h, JAp_h)
            out["Ap_method_gap"] = float(norm(via_kraus - via_blocks, "fro"))
            out["Ap_out_report"] = self.state_debug_report(via_kraus)
            out["Ap_diag"] = self.choi_diagnostics(JAp_h, d_in=self.dAp, d_out=self.dAp, gamma_in=self.gammaAp, gamma_out=self.gammaAp)
            gamma_k = self.apply_local_choi_Ap_no_norm(np.kron(self.gammaA, self.gammaAp), JAp_h)
            out["Ap_gamma_marginal_err"] = float(norm(partial_trace(gamma_k, self.dims, keep=[1]) - self.gammaAp, "fro"))
        if J_A is not None and J_Ap is not None:
            JAh = 0.5 * (J_A + J_A.conj().T)
            JAp_h = 0.5 * (J_Ap + J_Ap.conj().T)
            via_kraus = self.apply_local_product_choi_no_norm(rho_h, JAh, JAp_h)
            via_blocks = self.apply_local_product_choi_via_blocks(rho_h, JAh, JAp_h)
            out["product_method_gap"] = float(norm(via_kraus - via_blocks, "fro"))
            out["product_out_report"] = self.state_debug_report(via_kraus)
        return out

    def verify_local_gp_details(
        self,
        tau: np.ndarray,
        tau_p: np.ndarray,
        details: dict,
        eps_map: float | None = None,
        eps_gibbs: float | None = None,
        tol_psd: float = 1e-7,
        tol_tp: float = 1e-6,
        tol_kraus: float = 1e-12,
    ) -> dict:
        """
        Verifies a *claimed* local solution by explicit channel application + diagnostics.

        Returns dict with:
          - ok (bool)
          - map_err, gp_err_A, gp_err_Ap, tp_err_A, tp_err_Ap, min_eig_JA, min_eig_JAp
          - omega_consistency_err (if omega provided)
        """
        dA, dAp = self.dims
        eps_map = self.eps_eq_local if eps_map is None else float(eps_map)
        eps_g = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)

        J_A = details.get("J_A", None)
        J_Ap = details.get("J_Ap", None)
        omega = details.get("omega", None)

        if J_A is None or J_Ap is None:
            return {"ok": False, "reason": "missing_J"}

        J_Ah = 0.5 * (J_A + J_A.conj().T)
        J_Aph = 0.5 * (J_Ap + J_Ap.conj().T)

        diagA = self.choi_diagnostics(J_Ah, d_in=dA, d_out=dA, gamma_in=self.gammaA, gamma_out=self.gammaA)
        diagAp = self.choi_diagnostics(J_Aph, d_in=dAp, d_out=dAp, gamma_in=self.gammaAp, gamma_out=self.gammaAp)

        mapped = self.apply_local_product_choi_no_norm(tau, J_Ah, J_Aph, tol_kraus=tol_kraus)
        map_err = float(norm(0.5 * (mapped + mapped.conj().T) - 0.5 * (tau_p + tau_p.conj().T), "fro"))

        omega_err = None
        if omega is not None:
            omega_pred = self.apply_local_choi_A_no_norm(tau, J_Ah, tol_kraus=tol_kraus)
            omega_err = float(norm(0.5 * (omega_pred + omega_pred.conj().T) - 0.5 * (omega + omega.conj().T), "fro"))

        ok = True
        # CP
        if diagA["min_eig_J"] < -tol_psd or diagAp["min_eig_J"] < -tol_psd:
            ok = False
        # TP
        if diagA["tp_fro_err"] > tol_tp or diagAp["tp_fro_err"] > tol_tp:
            ok = False
        # GP
        if (diagA["gp_fro_err"] is None) or (diagAp["gp_fro_err"] is None):
            ok = False
        else:
            if diagA["gp_fro_err"] > eps_g + 10 * tol_tp or diagAp["gp_fro_err"] > eps_g + 10 * tol_tp:
                ok = False
        # Mapping
        if map_err > eps_map + 10 * tol_tp:
            ok = False

        return {
            "ok": ok,
            "map_err": map_err,
            "omega_consistency_err": omega_err,
            "min_eig_JA": diagA["min_eig_J"],
            "tp_err_A": diagA["tp_fro_err"],
            "gp_err_A": diagA["gp_fro_err"],
            "min_eig_JAp": diagAp["min_eig_J"],
            "tp_err_Ap": diagAp["tp_fro_err"],
            "gp_err_Ap": diagAp["gp_fro_err"],
        }

    def check_local_gp_feasible_multistart(
        self,
        tau: np.ndarray,
        tau_p: np.ndarray,
        solver=None,
        tol=None,
        eps_map: float | None = None,
        eps_gibbs: float | None = None,
        n_random_starts: int = 6,
        seed: int = 0,
        verify: bool = True,
        verbose: bool = False,
        return_details: bool = False,
    ):
        """
        Multistart wrapper around the existing two-step heuristic.
        Goal: reduce false negatives that look like "fragmentation".

        Strategy:
          - try multiple omega_hint choices (deterministic + random GP-preimages).
          - take the best residual solution found.
          - optionally verify it with explicit channel application.
        """
        eps_map_val = self.eps_eq_local if eps_map is None else float(eps_map)
        eps_g_val = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)

        tau_h = 0.5 * (tau + tau.conj().T)
        tau_p_h = 0.5 * (tau_p + tau_p.conj().T)

        hints = [
            None,                         # default (close to tau)
            tau_p_h,                      # bias omega toward target
            0.5 * (tau_h + tau_p_h),      # midpoint
            kron(self.gammaA, self.gammaAp),  # product thermal
        ]

        # Random starts: pick random local GP channel on A, generate omega=(G_A⊗I)(tau)
        rng = np.random.default_rng(seed)
        for k in range(int(n_random_starts)):
            J, st, _diag = self.find_random_local_gp_channel(which="A", solver=solver, tol=tol, eps_gibbs=eps_g_val, seed=int(rng.integers(0, 10**9)))
            if J is None:
                continue
            omega_k = self.apply_local_choi_A_no_norm(tau_h, J)
            hints.append(omega_k)

        best = {"res": np.inf, "feasible": False, "status": "no_attempt", "details": None, "verify": None}

        for idx, h in enumerate(hints):
            feas, status, det = self.check_local_gp_feasible(
                tau_h,
                tau_p_h,
                solver=solver,
                tol=tol,
                eps_map=eps_map_val,
                eps_gibbs=eps_g_val,
                omega_hint=h,
                verbose=verbose,
                return_details=True,
            )
            res = float(det.get("residual", np.inf))
            if res < best["res"]:
                best.update({"res": res, "feasible": bool(feas), "status": status, "details": det})

            if feas and verify:
                v = self.verify_local_gp_details(tau_h, tau_p_h, det, eps_map=eps_map_val, eps_gibbs=eps_g_val)
                best["verify"] = v
                if v["ok"]:
                    if return_details:
                        return True, f"multistart_ok (hint#{idx}) | {status}", {"best": best, "hint_index": idx}
                    return True, f"multistart_ok (hint#{idx}) | {status}"

        # If nothing verified, still return best attempt (useful as a gap score).
        if best.get("details") is not None and best.get("verify") is None:
            try:
                if best["details"].get("J_A") is not None and best["details"].get("J_Ap") is not None:
                    best["verify"] = self.verify_local_gp_details(tau_h, tau_p_h, best["details"], eps_map=eps_map_val, eps_gibbs=eps_g_val)
            except Exception as exc:
                best["verify"] = {"ok": False, "reason": f"verify_error: {exc}"}
        if return_details:
            return (best["feasible"] and (not verify)), f"multistart_best_only | {best['status']}", {"best": best}
        return (best["feasible"] and (not verify)), f"multistart_best_only | {best['status']}"

    @staticmethod
    def _invsqrt_psd(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Return mat^{-1/2} for PSD Hermitian mat via eigendecomposition."""
        w, U = eigh(0.5 * (mat + mat.conj().T))
        w = np.real(w)
        w[w < tol] = tol
        return U @ np.diag(w ** (-0.5)) @ dagger(U)

    def whiten_C(self, C0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Compute C~ = (γ^{-1/2}⊗γ'^{-1/2}) C0 (γ^{-1/2}⊗γ'^{-1/2})."""
        GinvA = self._invsqrt_psd(self.gammaA, tol=tol)
        GinvB = self._invsqrt_psd(self.gammaAp, tol=tol)
        W = np.kron(GinvA, GinvB)
        C0h = 0.5 * (C0 + C0.conj().T)
        Ct = W @ C0h @ W
        return 0.5 * (Ct + Ct.conj().T)

    def lt_ray_p_bounds(self, C0: np.ndarray, tol: float = 1e-12) -> tuple[float, float]:
        """
        For the LT ray family ρ(p) = γ⊗γ + p C0, positivity is equivalent to
          I + p C~ ⪰ 0,
        where C~ = (γ^{-1/2}⊗γ'^{-1/2}) C0 (γ^{-1/2}⊗γ'^{-1/2}).

        Returns (p_min, p_max) such that ρ(p) ⪰ 0 for all p in [p_min, p_max].
        """
        Ct = self.whiten_C(C0, tol=tol)
        lam = np.linalg.eigvalsh(0.5 * (Ct + Ct.conj().T))
        lam = np.real(lam)

        p_min = -np.inf
        p_max = +np.inf
        for x in lam:
            if x > tol:
                p_min = max(p_min, -1.0 / x)
            elif x < -tol:
                p_max = min(p_max, -1.0 / x)  # -1/negative is positive

        # If Ct has only one sign, one side is unbounded; clamp for safety.
        if not np.isfinite(p_min):
            p_min = -1e6
        if not np.isfinite(p_max):
            p_max = +1e6
        return float(p_min), float(p_max)

    def lt_ray_state(self, C0: np.ndarray, p: float) -> np.ndarray:
        """Construct ρ(p)=γ⊗γ+pC0 (Hermitian symmetrized)."""
        GAxGAp = kron(self.gammaA, self.gammaAp)
        rho = GAxGAp + float(p) * 0.5 * (C0 + C0.conj().T)
        rho = 0.5 * (rho + rho.conj().T)
        # trace should be 1 if Tr(C0)=0; enforce numerically anyway
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return 0.5 * (rho + rho.conj().T)

    def qubit_C0_from_pauli_label(self, label: str) -> np.ndarray:
        """
        Build a canonical zero-marginal, traceless direction C0 for (2,2) from a label:
          'XX','YY','ZZ','XY','XZ','YZ' (case-insensitive).

        Convention: C0 = (1/4) σ_i ⊗ σ_j.
        Then the correlation tensor coordinate t_{ij} = Tr(C σ_i⊗σ_j) equals 1 at p=1.
        """
        if self.dims != (2, 2):
            raise ValueError("qubit_C0_from_pauli_label requires dims=(2,2)")
        sx, sy, sz = paulis()
        pauli = {"X": sx, "Y": sy, "Z": sz}
        lab = label.strip().upper()
        if len(lab) != 2 or lab[0] not in pauli or lab[1] not in pauli:
            raise ValueError(f"Unknown pauli label '{label}'. Use one of XX,YY,ZZ,XY,XZ,YZ.")
        return 0.25 * np.kron(pauli[lab[0]], pauli[lab[1]])

    def qubit_C_from_diag_T(self, tx: float, ty: float, tz: float) -> np.ndarray:
        """Return C = (1/4)(tx XX + ty YY + tz ZZ) for dims=(2,2)."""
        if self.dims != (2, 2):
            raise ValueError("qubit_C_from_diag_T requires dims=(2,2)")
        sx, sy, sz = paulis()
        XX = np.kron(sx, sx)
        YY = np.kron(sy, sy)
        ZZ = np.kron(sz, sz)
        C = 0.25 * (float(tx) * XX + float(ty) * YY + float(tz) * ZZ)
        return 0.5 * (C + C.conj().T)

    def qubit_correlation_tensor_T(self, rho: np.ndarray, use_C: bool = True) -> np.ndarray:
        """
        Correlation tensor T_{ij} for i,j∈{x,y,z} extracted via
          C = rho - γ⊗γ,
          T_{ij} = Tr(C σ_i⊗σ_j)   (so C = (1/4) Σ_{ij} T_{ij} σ_i⊗σ_j).
        """
        if self.dims != (2, 2):
            raise ValueError("qubit_correlation_tensor_T requires dims=(2,2)")
        sx, sy, sz = paulis()
        sig = [sx, sy, sz]
        if use_C:
            X = self.correlation_C(rho)
        else:
            X = 0.5 * (rho + rho.conj().T)
        T = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                O = np.kron(sig[i], sig[j])
                T[i, j] = float(np.real(np.trace(X @ O)))
        return T

    @staticmethod
    def majorization_holds(x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
        """Check x majorizes y for real vectors (assumes nonnegative entries)."""
        xs = np.sort(np.real(x))[::-1]
        ys = np.sort(np.real(y))[::-1]
        if xs.shape != ys.shape:
            return False
        if xs.sum() + tol < ys.sum():
            return False
        cxs = np.cumsum(xs)
        cys = np.cumsum(ys)
        return bool(np.all(cxs + tol >= cys))
    
    
    def _select_solver(self, solver, verbose=False):
        solver_actual = self.solver_default if solver is None else solver
        if str(solver_actual).upper() == "AUTO":
            for s in ["MOSEK", "COSMO", "SCS"]:
                if s in cp.installed_solvers():
                    solver_actual = s
                    break
        if verbose:
            print(f"Using solver: {solver_actual}")
        return solver_actual

    @staticmethod
    def _scs_kwargs(tol: float, verbose: bool):
        return {
            "eps": tol,
            "max_iters": 200000,
            "alpha": 1.5,
            "scale": 5.0,
            "normalize": True,
            "use_indirect": False,
            "verbose": verbose,
        }

    # ----------------------------
    # Choi helpers (cvx + numpy)
    # ----------------------------

    @staticmethod
    def _choi_tp_constraints(J_var, d_in: int, d_out: int):
        """
        Trace-preserving constraint in Choi form:
          Tr_out(J) = I_in
        """
        I_in = np.eye(d_in)
        rows = []
        for m in range(d_in):
            r = []
            for n in range(d_in):
                s = 0
                for mu in range(d_out):
                    s += J_var[mu * d_in + m, mu * d_in + n]
                r.append(s)
            rows.append(r)
        Tr_out = cp.vstack([cp.hstack(r) for r in rows])
        return [Tr_out == I_in]

    @staticmethod
    def _choi_apply_cvx(J_var, X_const, d_in: int, d_out: int):
        """
        Apply interleaved-order Choi matrix J_var to X_const.

        J indices are ordered as (mu, i), i.e. flattened index = mu * d_in + i.
        Then block J[i::d_in, j::d_in] is Φ(|i><j|), so
            Φ(X) = sum_{i,j} X[i,j] * J[i::d_in, j::d_in].
        """
        Y = 0
        for i in range(d_in):
            for j in range(d_in):
                block = J_var[i::d_in, j::d_in]   # d_out x d_out
                Y += X_const[i, j] * block
        return Y

    @staticmethod
    def choi_apply_numpy(J: np.ndarray, X: np.ndarray, d_in: int, d_out: int):
        """
        Numpy version of the same interleaved-order Choi application.
        """
        Y = np.zeros((d_out, d_out), dtype=complex)
        for i in range(d_in):
            for j in range(d_in):
                block = J[i::d_in, j::d_in]
                Y += X[i, j] * block
        return Y

    @staticmethod
    def kraus_from_choi(J: np.ndarray, d_in: int, d_out: int, tol: float = 1e-12):
        """
        Extract Kraus operators from an interleaved-order Choi matrix J.

        J uses flattened index (mu, i) -> mu * d_in + i.
        Therefore eigenvectors unvectorize to K with reshape(..., order='C').
        """
        Jh = 0.5 * (J + J.conj().T)
        w, V = eigh(Jh)

        kraus = []
        for lam, v in zip(w, V.T):
            lam = float(np.real(lam))
            if lam <= tol:
                continue
            K = np.sqrt(lam) * v.reshape((d_out, d_in), order="C")
            kraus.append(K)
        return kraus

    # ==========================================
    # Local-channel utilities (random GP + apply)
    # ==========================================

    def _choi_tr_out_numpy(self, J: np.ndarray, d_in: int, d_out: int) -> np.ndarray:
        """Compute Tr_out(J) as a (d_in x d_in) matrix (numpy)."""
        out = np.zeros((d_in, d_in), dtype=complex)
        for m in range(d_in):
            for n in range(d_in):
                s = 0.0 + 0.0j
                for mu in range(d_out):
                    s += J[mu * d_in + m, mu * d_in + n]
                out[m, n] = s
        return 0.5 * (out + out.conj().T)

    def choi_diagnostics(
        self,
        J: np.ndarray,
        d_in: int,
        d_out: int,
        gamma_in: np.ndarray | None = None,
        gamma_out: np.ndarray | None = None,
        tol: float = 1e-12,
    ) -> dict:
        """Basic CPTP + (optional) Gibbs-preservation diagnostics for a Choi matrix."""
        Jh = 0.5 * (J + J.conj().T)

        # CP: smallest eigenvalue
        w = np.linalg.eigvalsh(Jh)
        min_eig = float(np.min(np.real(w)))

        # TP: Tr_out(J)=I_in
        Tr_out = self._choi_tr_out_numpy(Jh, d_in=d_in, d_out=d_out)
        tp_err = float(norm(Tr_out - np.eye(d_in), "fro"))

        gp_err = None
        if (gamma_in is not None) and (gamma_out is not None):
            Phi_gamma = self.choi_apply_numpy(Jh, gamma_in, d_in=d_in, d_out=d_out)
            gp_err = float(norm(Phi_gamma - gamma_out, "fro"))

        return {
            "min_eig_J": min_eig,
            "tp_fro_err": tp_err,
            "gp_fro_err": gp_err,
        }

    def find_random_local_gp_channel(
        self,
        which: str = "A",
        solver=None,
        tol=None,
        eps_gibbs: float | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray | None, str, dict]:
        """
        Construct a *random* local Gibbs-preserving channel Φ on subsystem A (or A') by solving:

          maximise    Re Tr(K^† J)
          subject to  J ⪰ 0, Tr_out(J)=I,  ||Φ(γ) - γ||_F ≤ eps_gibbs

        Returns (J, status, diagnostics). If infeasible/failed returns (None, status, diagnostics).
        """
        which_u = str(which).strip().upper()
        if which_u not in ("A", "AP", "A'"):
            raise ValueError("which must be 'A' or 'Ap'")
        if which_u == "A":
            d = self.dA
            gamma = self.gammaA
        else:
            d = self.dAp
            gamma = self.gammaAp

        solver_actual = self._select_solver(solver, verbose=verbose)
        tol_val = self.tol_default if tol is None else float(tol)
        eps_g = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)

        rng = np.random.default_rng(seed)
        R = rng.normal(size=(d * d, d * d)) + 1j * rng.normal(size=(d * d, d * d))
        K = 0.5 * (R + R.conj().T)  # Hermitian objective

        J = cp.Variable((d * d, d * d), complex=True, name=f"J_rand_{which_u}")
        cons = [J >> 0]
        cons += self._choi_tp_constraints(J, d_in=d, d_out=d)
        cons += [cp.norm(self._choi_apply_cvx(J, gamma, d_in=d, d_out=d) - gamma, "fro") <= eps_g]

        obj = cp.Maximize(cp.real(cp.trace(K.conj().T @ J)))
        prob = cp.Problem(obj, cons)

        scs_kwargs = self._scs_kwargs(tol=tol_val, verbose=verbose) if solver_actual == "SCS" else {"verbose": verbose}

        try:
            prob.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            return None, f"random-LGP solver error: {e}", {"status": "error"}

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None, f"random-LGP status: {prob.status}", {"status": prob.status}

        if J.value is None:
            return None, f"random-LGP status: {prob.status} (no J)", {"status": prob.status}

        Jv = 0.5 * (J.value + J.value.conj().T)
        diag = self.choi_diagnostics(Jv, d_in=d, d_out=d, gamma_in=gamma, gamma_out=gamma, tol=1e-12)
        diag["status"] = prob.status
        return Jv, f"{prob.status}", diag

    def apply_local_channel_A(self, rho: np.ndarray, J_A: np.ndarray) -> np.ndarray:
        """Apply local channel (Choi J_A) on subsystem A to a bipartite operator rho (numpy)."""
        dA, dAp = self.dims
        rho_h = 0.5 * (rho + rho.conj().T)
        blocks = rho_h.reshape(dA, dAp, dA, dAp)
        out = np.zeros((dA * dAp, dA * dAp), dtype=complex)
        for i in range(dA):
            for j in range(dA):
                Eij = np.zeros((dA, dA), dtype=complex)
                Eij[i, j] = 1.0
                Phi_Eij = self.choi_apply_numpy(J_A, Eij, d_in=dA, d_out=dA)
                Tij = blocks[i, :, j, :]
                out += np.kron(Phi_Eij, Tij)
        out = 0.5 * (out + out.conj().T)
        tr = np.trace(out)
        if abs(tr) > 1e-15:
            out = out / tr
        return 0.5 * (out + out.conj().T)

    def apply_local_channel_Ap(self, rho: np.ndarray, J_Ap: np.ndarray) -> np.ndarray:
        """Apply local channel (Choi J_Ap) on subsystem A' to a bipartite operator rho (numpy)."""
        dA, dAp = self.dims
        rho_h = 0.5 * (rho + rho.conj().T)
        blocks = rho_h.reshape(dA, dAp, dA, dAp)
        out = np.zeros((dA * dAp, dA * dAp), dtype=complex)
        for a in range(dAp):
            for b in range(dAp):
                Eab = np.zeros((dAp, dAp), dtype=complex)
                Eab[a, b] = 1.0
                Phi_Eab = self.choi_apply_numpy(J_Ap, Eab, d_in=dAp, d_out=dAp)
                Xab = blocks[:, a, :, b]
                out += np.kron(Xab, Phi_Eab)
        out = 0.5 * (out + out.conj().T)
        tr = np.trace(out)
        if abs(tr) > 1e-15:
            out = out / tr
        return 0.5 * (out + out.conj().T)

    # ==========================================
    # Global GP (channel extraction + feasibility)
    # ==========================================

    def find_global_gp_channel(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        eps_gibbs=None,
        verbose=False,
    ):
        """
        Solve for a *concrete* global Gibbs-preserving CPTP map Φ (via Choi J)
        that best approximates tau -> tau_p:

            minimise  || Φ(tau) - tau_p ||_F
            subject to  J ⪰ 0, Tr_out(J)=I,  ||Φ(γ⊗γ) - (γ⊗γ)||_F ≤ eps_gibbs

        Returns a dict with:
          - status
          - J (Choi matrix) if available
          - map_residual (optimal value)
          - gibbs_residual (computed a posteriori if J found)
        """
        dA, dAp = self.dims
        d_in = d_out = dA * dAp

        tau_clean = 0.5 * (tau + tau.conj().T)
        tau_p_clean = 0.5 * (tau_p + tau_p.conj().T)

        eps_g = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)
        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else float(tol)

        if norm(tau_clean - tau_p_clean, "fro") <= 1e-12:
            J_id = choi_identity(d_in)
            return {
                "status": "identity_case",
                "solver": solver_actual,
                "J": J_id,
                "map_residual": 0.0,
                "gibbs_residual": 0.0,
            }

        scs_kwargs = self._scs_kwargs(tol, verbose) if str(solver_actual).upper() == "SCS" else {}

        J = cp.Variable((d_out * d_in, d_out * d_in), complex=True, name="J")

        GAxGAp = kron(self.gammaA, self.gammaAp)
        GAxGAp_mat = 0.5 * (GAxGAp + GAxGAp.conj().T)

        constraints = [J >> 0]
        constraints += self._choi_tp_constraints(J, d_in=d_in, d_out=d_out)

        Y_gp = self._choi_apply_cvx(J, GAxGAp_mat, d_in=d_in, d_out=d_out)
        constraints += [cp.norm(Y_gp - GAxGAp_mat, "fro") <= eps_g]

        Y_conv = self._choi_apply_cvx(J, tau_clean, d_in=d_in, d_out=d_out)
        objective = cp.Minimize(cp.norm(Y_conv - tau_p_clean, "fro"))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"Global GP solver error: {e}")
            return {
                "status": f"error: {str(e)}",
                "solver": solver_actual,
                "J": None,
                "map_residual": np.inf,
                "gibbs_residual": np.inf,
            }

        J_val = None if J.value is None else 0.5 * (J.value + J.value.conj().T)

        gibbs_res = np.inf
        if J_val is not None:
            Yg = self.choi_apply_numpy(J_val, GAxGAp_mat, d_in=d_in, d_out=d_out)
            gibbs_res = float(norm(Yg - GAxGAp_mat, "fro"))

        return {
            "status": prob.status,
            "solver": solver_actual,
            "J": J_val,
            "map_residual": float(prob.value) if prob.value is not None else np.inf,
            "gibbs_residual": gibbs_res,
        }

    def check_global_gp_feasible(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        eps_eq=None,      # legacy name
        eps_map=None,     # preferred name
        eps_gibbs=None,
        verbose=False,
        return_details=False,
    ):
        """
        Feasibility check for existence of global GP CPTP Φ such that:
          ||Φ(γ⊗γ)-(γ⊗γ)||_F ≤ eps_gibbs
          ||Φ(tau) - tau_p||_F ≤ eps_map

        Returns:
          - (feasible: bool, status: str) by default
          - (feasible: bool, status: str, details: dict) if return_details=True
        """
        eps_map_val = self.eps_eq_global if (eps_eq is None and eps_map is None) else float(eps_map if eps_map is not None else eps_eq)
        eps_g_val = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)

        details = self.find_global_gp_channel(
            tau, tau_p,
            solver=solver,
            tol=tol,
            eps_gibbs=eps_g_val,
            verbose=verbose,
        )

        status = details.get("status", "unknown")
        map_res = float(details.get("map_residual", np.inf))
        gibbs_res = float(details.get("gibbs_residual", np.inf))

        # If solver reported infeasible, treat as infeasible regardless of residual.
        if str(status).lower().startswith("infeasible"):
            feasible = False
        else:
            feasible = (map_res <= eps_map_val + 1e-12) and (gibbs_res <= eps_g_val + 1e-8)

        status_str = f"{status} (map_res={map_res:.3e}, gibbs_res={gibbs_res:.3e})"

        if return_details:
            return feasible, status_str, details
        return feasible, status_str

    # ==========================================
    # Local GP (two-step heuristic + residual)
    # ==========================================


    def check_local_gp_feasible(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        eps_eq=None,      # legacy name
        eps_map=None,     # preferred name
        eps_gibbs=None,
        verbose=False,
        omega_hint=None,
        return_details=False,
    ):
        """
        Two-step local GP test:

        Step 1: Find a GP channel on A giving intermediate omega = (G_A ⊗ id)(tau).
        Step 2: Find a GP channel on A' that best maps omega -> tau_p.

        The returned residual from Step 2 is a useful quantitative "gap" score.
        When return_details=True, this method also returns channel diagnostics,
        omega diagnostics, and explicit re-application checks to help debug false
        positives / negatives.
        """
        dA, dAp = self.dims

        tau_clean = 0.5 * (tau + tau.conj().T)
        tau_p_clean = 0.5 * (tau_p + tau_p.conj().T)

        if norm(tau_clean - tau_p_clean, "fro") <= 1e-12:
            ident_details = {
                "residual": 0.0,
                "threshold": 0.0,
                "J_A": None,
                "J_Ap": None,
                "omega": tau_clean,
                "status_step1": "identity_case",
                "status_step2": "identity_case",
                "tau_report": self.state_debug_report(tau_clean),
                "tau_p_report": self.state_debug_report(tau_p_clean),
                "omega_report": self.state_debug_report(tau_clean),
            }
            if return_details:
                return True, "identity_case", ident_details
            return True, "identity_case"

        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else float(tol)

        eps_map_val = self.eps_eq_local if (eps_eq is None and eps_map is None) else float(eps_map if eps_map is not None else eps_eq)
        eps_g_val = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)

        scs_kwargs = self._scs_kwargs(tol, verbose) if str(solver_actual).upper() == "SCS" else {}

        def choi_apply_local(J_var, X_const, d):
            XT = X_const.T
            Y = 0
            for i in range(d):
                acc = 0
                for j in range(d):
                    block = J_var[i::d, j::d]
                    acc += XT[i, j] * block
                Y += acc
            return Y

        # -------- STEP 1: channel on A --------
        JA = cp.Variable((dA * dA, dA * dA), complex=True, name="J_A")
        omega = cp.Variable((dA * dAp, dA * dAp), complex=True, name="omega")
        I_A = np.eye(dA)

        try:
            JA.value = choi_identity(dA)
        except Exception:
            pass

        cons1 = [JA >> 0]
        rows = []
        for m in range(dA):
            r = []
            for n in range(dA):
                s = 0
                for mu in range(dA):
                    s += JA[mu * dA + m, mu * dA + n]
                r.append(s)
            rows.append(r)
        cons1 += [cp.vstack([cp.hstack(r) for r in rows]) == I_A]
        cons1 += [cp.norm(choi_apply_local(JA, self.gammaA, dA) - self.gammaA, "fro") <= eps_g_val]

        tau_blocks = tau_clean.reshape(dA, dAp, dA, dAp)
        omega_expr = 0
        for i in range(dA):
            for j in range(dA):
                Eij = np.zeros((dA, dA), dtype=complex)
                Eij[i, j] = 1.0
                GA_Eij = choi_apply_local(JA, Eij, dA)
                Tij = tau_blocks[i, :, j, :]
                omega_expr += cp.kron(GA_Eij, Tij)

        cons1 += [omega >> 0, cp.trace(omega) == 1, omega == omega_expr]

        if omega_hint is not None:
            omega_target = 0.5 * (omega_hint + omega_hint.conj().T)
            obj1 = cp.Minimize(cp.norm(omega - omega_target, "fro"))
        else:
            obj1 = cp.Minimize(cp.norm(omega - tau_clean, "fro"))

        prob1 = cp.Problem(obj1, cons1)
        try:
            prob1.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"LGP step-1 solver error: {e}")
            err_details = {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": None, "status_step1": f"error: {e}", "status_step2": "not_run"}
            if return_details:
                return False, f"LGP step-1 error: {str(e)}", err_details
            return False, f"LGP step-1 error: {str(e)}"

        if prob1.status not in ["optimal", "optimal_inaccurate"]:
            if verbose:
                print(f"LGP step-1 status: {prob1.status}")
            fail_details = {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": None, "status_step1": prob1.status, "status_step2": "not_run"}
            if return_details:
                return False, f"LGP step-1 {prob1.status}", fail_details
            return False, f"LGP step-1 {prob1.status}"

        omega_val = 0.5 * (omega.value + omega.value.conj().T)

        # -------- STEP 2: channel on A' --------
        JAp = cp.Variable((dAp * dAp, dAp * dAp), complex=True, name="J_Ap")
        I_Ap = np.eye(dAp)

        try:
            JAp.value = choi_identity(dAp)
        except Exception:
            pass

        cons2 = [JAp >> 0]
        rows = []
        for m in range(dAp):
            r = []
            for n in range(dAp):
                s = 0
                for mu in range(dAp):
                    s += JAp[mu * dAp + m, mu * dAp + n]
                r.append(s)
            rows.append(r)
        cons2 += [cp.vstack([cp.hstack(r) for r in rows]) == I_Ap]
        cons2 += [cp.norm(choi_apply_local(JAp, self.gammaAp, dAp) - self.gammaAp, "fro") <= eps_g_val]

        omega_blocks = omega_val.reshape(dA, dAp, dA, dAp)
        tau_p_expr = 0
        for a in range(dAp):
            for b in range(dAp):
                Eab = np.zeros((dAp, dAp), dtype=complex)
                Eab[a, b] = 1.0
                GAp_Eab = choi_apply_local(JAp, Eab, dAp)
                Xab = omega_blocks[:, a, :, b]
                tau_p_expr += cp.kron(Xab, GAp_Eab)

        obj2 = cp.Minimize(cp.norm(tau_p_expr - tau_p_clean, "fro"))
        prob2 = cp.Problem(obj2, cons2)

        try:
            prob2.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"LGP step-2 solver error: {e}")
            err_details = {"residual": np.inf, "J_A": None if JA.value is None else 0.5 * (JA.value + JA.value.conj().T), "J_Ap": None, "omega": omega_val, "status_step1": prob1.status, "status_step2": f"error: {e}"}
            if return_details:
                return False, f"LGP step-2 error: {str(e)}", err_details
            return False, f"LGP step-2 error: {str(e)}"

        if prob2.status not in ["optimal", "optimal_inaccurate"]:
            if verbose:
                print(f"LGP step-2 status: {prob2.status}")
            fail_details = {"residual": np.inf, "J_A": None if JA.value is None else 0.5 * (JA.value + JA.value.conj().T), "J_Ap": None, "omega": omega_val, "status_step1": prob1.status, "status_step2": prob2.status}
            if return_details:
                return False, f"LGP step-2 {prob2.status}", fail_details
            return False, f"LGP step-2 {prob2.status}"

        res = float(prob2.value) if prob2.value is not None else np.inf
        feasible = res <= eps_map_val + 1e-12

        status = f"{prob2.status} (residual={res:.3e}, threshold={eps_map_val:.3e})"
        details = {
            "residual": res,
            "threshold": eps_map_val,
            "J_A": None if JA.value is None else 0.5 * (JA.value + JA.value.conj().T),
            "J_Ap": None if JAp.value is None else 0.5 * (JAp.value + JAp.value.conj().T),
            "omega": omega_val,
            "status_step1": prob1.status,
            "status_step2": prob2.status,
            "step1_obj": float(prob1.value) if prob1.value is not None else np.inf,
            "step2_obj": res,
            "tau_report": self.state_debug_report(tau_clean),
            "tau_p_report": self.state_debug_report(tau_p_clean),
            "omega_report": self.state_debug_report(omega_val),
        }

        if details["J_A"] is not None:
            details["J_A_diag"] = self.choi_diagnostics(details["J_A"], d_in=dA, d_out=dA, gamma_in=self.gammaA, gamma_out=self.gammaA)
            details["A_application_debug"] = self.local_channel_application_debug(tau_clean, J_A=details["J_A"])
        if details["J_Ap"] is not None:
            details["J_Ap_diag"] = self.choi_diagnostics(details["J_Ap"], d_in=dAp, d_out=dAp, gamma_in=self.gammaAp, gamma_out=self.gammaAp)
            details["Ap_application_debug"] = self.local_channel_application_debug(omega_val, J_Ap=details["J_Ap"])
        if details["J_A"] is not None and details["J_Ap"] is not None:
            details["verification"] = self.verify_local_gp_details(tau_clean, tau_p_clean, details, eps_map=eps_map_val, eps_gibbs=eps_g_val)
            details["product_application_debug"] = self.local_channel_application_debug(tau_clean, J_A=details["J_A"], J_Ap=details["J_Ap"])

        if return_details:
            return feasible, status, details
        return feasible, status

    # --------- Support function: extremal LT state ---------

    def extremal_lt_state(
        self,
        K,
        classical=False,
        solver=None,
        tol=None,
        verbose=False
    ):
        """
        Maximise Tr(K rho) over:
          - all LT states (if classical=False),
          - classical LT (diagonal) states (if classical=True).
        Returns rho*, optimum value, problem status.
        """
        dA, dAp = self.dims
        d = dA * dAp
        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol

        if solver_actual.upper() == "SCS":
            scs_kwargs = {
                "eps": tol,
                "max_iters": 200000,
                "alpha": 1.5,
                "scale": 5.0,
                "normalize": True,
                "use_indirect": False,
                "verbose": verbose,
            }
        else:
            scs_kwargs = {}

        rho = cp.Variable((d, d), complex=True, name="rho")
        constraints = [rho >> 0, cp.trace(rho) == 1]

        # partial trace constraints
        # Use explicit summation for small dims:
        # Tr_B rho = gammaA, Tr_A rho = gammaAp
        rhoA_blocks = []
        for i in range(dA):
            row = []
            for j in range(dA):
                s = 0
                for k in range(dAp):
                    idx_row = i*dAp + k
                    idx_col = j*dAp + k
                    s += rho[idx_row, idx_col]
                row.append(s)
            rhoA_blocks.append(row)
        rhoA = cp.vstack([cp.hstack(r) for r in rhoA_blocks])

        rhoAp_blocks = []
        for i in range(dAp):
            row = []
            for j in range(dAp):
                s = 0
                for k in range(dA):
                    idx_row = k*dAp + i
                    idx_col = k*dAp + j
                    s += rho[idx_row, idx_col]
                row.append(s)
            rhoAp_blocks.append(row)
        rhoAp = cp.vstack([cp.hstack(r) for r in rhoAp_blocks])

        constraints += [rhoA == self.gammaA, rhoAp == self.gammaAp]

        # classical restriction: diagonal in energy basis
        if classical:
            for i in range(d):
                for j in range(d):
                    if i != j:
                        constraints += [rho[i, j] == 0]

        objective = cp.Maximize(cp.real(cp.trace(K @ rho)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver_actual, **scs_kwargs)

        return (
            rho.value if rho.value is not None else None,
            prob.value,
            prob.status
        )

    # --------- Projection to LT set (trace distance) ---------

    def closest_lt_state(
        self,
        rho0,
        classical=False,
        solver=None,
        tol=None,
        verbose=False
    ):
        """
        Compute closest (classical) LT state in trace distance:
          min_{sigma in LT} 0.5 ||rho0 - sigma||_1.

        If classical=True: sigma restricted to diagonal in energy basis.
        """
        dA, dAp = self.dims
        d = dA * dAp

        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol

        if solver_actual.upper() == "SCS":
            scs_kwargs = {
                "eps": tol,
                "max_iters": 200000,
                "alpha": 1.5,
                "scale": 5.0,
                "normalize": True,
                "use_indirect": False,
                "verbose": verbose,
            }
        else:
            scs_kwargs = {}

        sigma = cp.Variable((d, d), complex=True, name="sigma")
        P = cp.Variable((d, d), complex=True, name="P")
        N = cp.Variable((d, d), complex=True, name="N")

        constraints = [
            sigma >> 0,
            P >> 0,
            N >> 0,
            cp.trace(sigma) == 1,
        ]

        # partial trace constraints as above
        sigmaA_blocks = []
        for i in range(dA):
            row = []
            for j in range(dA):
                s = 0
                for k in range(dAp):
                    idx_row = i*dAp + k
                    idx_col = j*dAp + k
                    s += sigma[idx_row, idx_col]
                row.append(s)
            sigmaA_blocks.append(row)
        sigmaA = cp.vstack([cp.hstack(r) for r in sigmaA_blocks])

        sigmaAp_blocks = []
        for i in range(dAp):
            row = []
            for j in range(dAp):
                s = 0
                for k in range(dA):
                    idx_row = k*dAp + i
                    idx_col = k*dAp + j
                    s += sigma[idx_row, idx_col]
                row.append(s)
            sigmaAp_blocks.append(row)
        sigmaAp = cp.vstack([cp.hstack(r) for r in sigmaAp_blocks])

        constraints += [sigmaA == self.gammaA, sigmaAp == self.gammaAp]

        if classical:
            for i in range(d):
                for j in range(d):
                    if i != j:
                        constraints += [sigma[i, j] == 0]

        constraints += [rho0 - sigma == P - N]
        objective = cp.Minimize(0.5 * cp.real(cp.trace(P + N)))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver_actual, **scs_kwargs)

        return (
            sigma.value if sigma.value is not None else None,
            prob.value,
            prob.status
        )

    # --------- High-level analysis wrapper ---------

    def analyze_convertibility(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        verbose=False,
        eps_eq_global=None,
        eps_eq_local=None,
        omega_hint=None
    ):
        """
        High-level wrapper:
          - check LT membership of tau and tau_p,
          - compute monotones,
          - check Global GP feasibility,
          - check Local GP feasibility.
        """
        tol = self.tol_default if tol is None else tol
        if eps_eq_global is not None:
            self.eps_eq_global = eps_eq_global
        if eps_eq_local is not None:
            self.eps_eq_local = eps_eq_local

        LT_tau,  ltA, ltAp, tauA, tauAp = self.lt_membership(tau,   tol=1e-8)
        LT_taup, ltA2, ltAp2, tA2, tAp2 = self.lt_membership(tau_p, tol=1e-8)

        D_tau,  I_tau,  C_A,  C_Ap  = self.monotones(tau)
        D_taup, I_taup, C2_A, C2_Ap = self.monotones(tau_p)

        gp_feas,  gp_status  = self.check_global_gp_feasible(
            tau, tau_p, solver=solver, tol=tol, verbose=verbose
        )
        lgp_feas, lgp_status = self.check_local_gp_feasible(
            tau, tau_p, solver=solver, tol=tol, verbose=verbose,
            omega_hint=omega_hint
        )

        report = {
            "dims": self.dims,
            "beta": self.beta,
            "gammaA": self.gammaA, 
            "gammaAp": self.gammaAp,
            "LT_tau": LT_tau,
            "LT_tau_breakdown": {"A": ltA, "Ap": ltAp},
            "LT_taup": LT_taup,
            "LT_taup_breakdown": {"A": ltA2, "Ap": ltAp2},
            "monotones": {
                "D_tau_vs_gamma": D_tau,
                "D_taup_vs_gamma": D_taup,
                "I_tau": I_tau,
                "I_taup": I_taup,
                "C_rel_entropy_A_tau": C_A,
                "C_rel_entropy_Ap_tau": C_Ap,
                "C_rel_entropy_A_taup": C2_A,
                "C_rel_entropy_Ap_taup": C2_Ap,
            },
            "feasibility": {
                "Global_GP": {"feasible": gp_feas, "status": gp_status},
                "Local_GP":  {"feasible": lgp_feas, "status": lgp_status},
            }
        }
        return report



# -----------------------------------------------------------------------------
# LTGP extensions + architecture cleanup
# -----------------------------------------------------------------------------

def pauli_basis_orthonormal() -> list[np.ndarray]:
    """Orthonormal traceless basis for d=2: {σx/√2, σy/√2, σz/√2}."""
    sx, sy, sz = paulis()
    return [sx / np.sqrt(2.0), sy / np.sqrt(2.0), sz / np.sqrt(2.0)]


def gell_mann_basis_orthonormal() -> list[np.ndarray]:
    """Orthonormal traceless Gell-Mann basis for d=3 with Tr(Bi Bj)=δij."""
    lam = []
    lam.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))
    lam.append((1.0 / np.sqrt(3.0)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex))
    return [L / np.sqrt(2.0) for L in lam]


def _invsqrt_psd(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    H = 0.5 * (mat + mat.conj().T)
    w, U = np.linalg.eigh(H)
    w = np.real(w)
    w[w < tol] = tol
    return U @ np.diag(w ** (-0.5)) @ dagger(U)


@dataclass
class PPTLocalRelaxResult:
    feasible: bool
    status: str
    map_residual: float
    gibbs_residual: float


@dataclass
class PairReport:
    label: str
    global_feasible: bool
    global_status: str
    local_feasible: bool
    local_status: str
    monotones: Dict[str, float]
    diagnostics: Dict[str, Any]


class LTGPSystem(LTSDPSystem):
    """Research-grade LT/GP system.

    Improvements over the legacy core:
    - energy-basis diagonal constraints are implemented correctly;
    - verified local GP is exposed as the default report-facing path;
    - renormalizing local channel application is removed from diagnostics;
    - qutrit commuting LT sampling and PPT outer relaxation are included;
    - qubit/qutrit correlation-basis parameterisations are available.
    """

    # ---------- basis / exact restrictions ----------

    def energy_eigenbasis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local energy eigenvector matrices and their product basis unitary."""
        _eA, UA = eigh(0.5 * (self.H_A + self.H_A.conj().T))
        _eB, UB = eigh(0.5 * (self.H_Ap + self.H_Ap.conj().T))
        U = np.kron(UA, UB)
        return UA, UB, U

    def to_energy_basis(self, rho: np.ndarray) -> np.ndarray:
        _, _, U = self.energy_eigenbasis()
        return dagger(U) @ rho @ U

    def from_energy_basis(self, rho_e: np.ndarray) -> np.ndarray:
        _, _, U = self.energy_eigenbasis()
        return U @ rho_e @ dagger(U)

    def dephase_global_in_energy_basis(self, rho: np.ndarray) -> np.ndarray:
        rho_e = self.to_energy_basis(rho)
        rho_de = np.diag(np.diag(rho_e))
        out = self.from_energy_basis(rho_de)
        return 0.5 * (out + out.conj().T)

    def is_energy_diagonal(self, rho: np.ndarray, tol: float = 1e-10) -> bool:
        rho_e = self.to_energy_basis(rho)
        off = rho_e - np.diag(np.diag(rho_e))
        return bool(norm(off, 'fro') <= tol)

    def _cvx_energy_diag_constraints(self, X: cp.Expression) -> list:
        d = self.dA * self.dAp
        _, _, U = self.energy_eigenbasis()
        X_e = dagger(U) @ X @ U
        cons = []
        for i in range(d):
            for j in range(d):
                if i != j:
                    cons.append(X_e[i, j] == 0)
        return cons

    def apply_local_channel_A(self, rho: np.ndarray, J_A: np.ndarray) -> np.ndarray:
        """Apply local channel on A without renormalizing the output."""
        return self.apply_local_choi_A_no_norm(rho, J_A)

    def apply_local_channel_Ap(self, rho: np.ndarray, J_Ap: np.ndarray) -> np.ndarray:
        """Apply local channel on A' without renormalizing the output."""
        return self.apply_local_choi_Ap_no_norm(rho, J_Ap)

    def apply_local_product_channel(self, rho: np.ndarray, J_A: np.ndarray, J_Ap: np.ndarray) -> np.ndarray:
        return self.apply_local_product_choi_no_norm(rho, J_A, J_Ap)

    # ---------- corrected LT SDPs ----------

    def extremal_lt_state(self, K, classical: bool = False, solver=None, tol=None, verbose: bool = False):
        dA, dAp = self.dims
        d = dA * dAp
        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol
        scs_kwargs = self._scs_kwargs(tol, verbose) if str(solver_actual).upper() == 'SCS' else {}

        rho = cp.Variable((d, d), complex=True, name='rho')
        constraints = [rho >> 0, cp.trace(rho) == 1]

        rhoA_blocks = []
        for i in range(dA):
            row = []
            for j in range(dA):
                s = 0
                for k in range(dAp):
                    idx_row = i * dAp + k
                    idx_col = j * dAp + k
                    s += rho[idx_row, idx_col]
                row.append(s)
            rhoA_blocks.append(row)
        rhoA = cp.vstack([cp.hstack(r) for r in rhoA_blocks])

        rhoAp_blocks = []
        for i in range(dAp):
            row = []
            for j in range(dAp):
                s = 0
                for k in range(dA):
                    idx_row = k * dAp + i
                    idx_col = k * dAp + j
                    s += rho[idx_row, idx_col]
                row.append(s)
            rhoAp_blocks.append(row)
        rhoAp = cp.vstack([cp.hstack(r) for r in rhoAp_blocks])
        constraints += [rhoA == self.gammaA, rhoAp == self.gammaAp]
        if classical:
            constraints += self._cvx_energy_diag_constraints(rho)

        objective = cp.Maximize(cp.real(cp.trace(K @ rho)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver_actual, **scs_kwargs)
        return (rho.value if rho.value is not None else None, prob.value, prob.status)

    def closest_lt_state(self, rho0, classical: bool = False, solver=None, tol=None, verbose: bool = False):
        dA, dAp = self.dims
        d = dA * dAp
        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol
        scs_kwargs = self._scs_kwargs(tol, verbose) if str(solver_actual).upper() == 'SCS' else {}

        sigma = cp.Variable((d, d), complex=True, name='sigma')
        P = cp.Variable((d, d), complex=True, name='P')
        N = cp.Variable((d, d), complex=True, name='N')

        constraints = [sigma >> 0, P >> 0, N >> 0, cp.trace(sigma) == 1]

        sigmaA_blocks = []
        for i in range(dA):
            row = []
            for j in range(dA):
                s = 0
                for k in range(dAp):
                    idx_row = i * dAp + k
                    idx_col = j * dAp + k
                    s += sigma[idx_row, idx_col]
                row.append(s)
            sigmaA_blocks.append(row)
        sigmaA = cp.vstack([cp.hstack(r) for r in sigmaA_blocks])

        sigmaAp_blocks = []
        for i in range(dAp):
            row = []
            for j in range(dAp):
                s = 0
                for k in range(dA):
                    idx_row = k * dAp + i
                    idx_col = k * dAp + j
                    s += sigma[idx_row, idx_col]
                row.append(s)
            sigmaAp_blocks.append(row)
        sigmaAp = cp.vstack([cp.hstack(r) for r in sigmaAp_blocks])

        constraints += [sigmaA == self.gammaA, sigmaAp == self.gammaAp]
        if classical:
            constraints += self._cvx_energy_diag_constraints(sigma)

        constraints += [0.5 * ((rho0 + rho0.conj().T)) - sigma == P - N]
        prob = cp.Problem(cp.Minimize(0.5 * cp.real(cp.trace(P + N))), constraints)
        prob.solve(solver=solver_actual, **scs_kwargs)
        return (sigma.value if sigma.value is not None else None, prob.value, prob.status)

    # ---------- extra parameterisations ----------

    def lt_from_correlation_matrix(self, C: np.ndarray) -> np.ndarray:
        """ρ = γ⊗γ + Σ Cij (Bi⊗Bj), with traceless Bi preserving LT marginals."""
        dA, dB = self.dims
        G = np.kron(self.gammaA, self.gammaAp)
        if (dA, dB) == (2, 2):
            B = pauli_basis_orthonormal(); expected = (3, 3)
        elif (dA, dB) == (3, 3):
            B = gell_mann_basis_orthonormal(); expected = (8, 8)
        else:
            raise NotImplementedError('Implemented for (2,2) and (3,3).')
        if tuple(C.shape) != expected:
            raise ValueError(f'Expected correlation matrix shape {expected}, got {C.shape}')
        rho = G.copy()
        for i in range(len(B)):
            for j in range(len(B)):
                rho = rho + float(C[i, j]) * np.kron(B[i], B[j])
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return 0.5 * (rho + rho.conj().T)

    def correlation_matrix_from_lt(self, rho: np.ndarray) -> np.ndarray:
        dA, dB = self.dims
        G = np.kron(self.gammaA, self.gammaAp)
        C_op = 0.5 * ((rho - G) + (rho - G).conj().T)
        if (dA, dB) == (2, 2):
            B = pauli_basis_orthonormal(); n = 3
        elif (dA, dB) == (3, 3):
            B = gell_mann_basis_orthonormal(); n = 8
        else:
            raise NotImplementedError('Implemented for (2,2) and (3,3).')
        C = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                C[i, j] = float(np.real(np.trace(C_op @ np.kron(B[i], B[j]))))
        return C

    def lt_ray_bounds(self, C0: np.ndarray, tol: float = 1e-12) -> Tuple[float, float]:
        return self.lt_ray_p_bounds(C0, tol=tol)

    # ---------- qutrit commuting LT ----------

    def sample_commuting_lt_d3(self, n: int, seed: int = 0, iters: int = 300) -> list[np.ndarray]:
        if self.dims != (3, 3):
            raise ValueError('sample_commuting_lt_d3 requires dims=(3,3)')
        rng = np.random.default_rng(int(seed))
        _wA, UA = np.linalg.eigh(0.5 * (self.H_A + self.H_A.conj().T))
        _wB, UB = np.linalg.eigh(0.5 * (self.H_Ap + self.H_Ap.conj().T))
        GA_e = np.diag(dagger(UA) @ self.gammaA @ UA).real
        GB_e = np.diag(dagger(UB) @ self.gammaAp @ UB).real
        gA = np.clip(GA_e, 0.0, 1.0); gA = gA / gA.sum()
        gB = np.clip(GB_e, 0.0, 1.0); gB = gB / gB.sum()
        U_tot = np.kron(UA, UB)
        states = []
        for _ in range(int(n)):
            P = rng.random((3, 3)) + 1e-12
            for _k in range(int(iters)):
                P = P * (gA / (P.sum(axis=1) + 1e-18))[:, None]
                P = P * (gB / (P.sum(axis=0) + 1e-18))[None, :]
            P = np.clip(P, 0.0, None)
            P = P / P.sum()
            rho_e = np.diag(P.reshape(9))
            rho = U_tot @ rho_e @ dagger(U_tot)
            rho = 0.5 * (rho + rho.conj().T)
            states.append(rho)
        return states

    # ---------- PPT local outer relaxation ----------

    @staticmethod
    def _choi_ppt_partial_transpose_cvx(J: cp.Expression, dA: int, dB: int) -> cp.Expression:
        d_out = d_in = dA * dB
        D = d_out * d_in
        def decomp(o: int) -> tuple[int, int]:
            return o // dB, o % dB
        def idx(o: int, i: int) -> int:
            return o * d_in + i
        rows = []
        for r in range(D):
            out_r = r // d_in
            in_r = r % d_in
            aout, bout = decomp(out_r)
            ain, bin_ = decomp(in_r)
            row = []
            for c in range(D):
                out_c = c // d_in
                in_c = c % d_in
                aout_p, bout_p = decomp(out_c)
                ain_p, bin_p = decomp(in_c)
                out_r_src = aout * dB + bout_p
                in_r_src = ain * dB + bin_p
                out_c_src = aout_p * dB + bout
                in_c_src = ain_p * dB + bin_
                row.append(J[idx(out_r_src, in_r_src), idx(out_c_src, in_c_src)])
            rows.append(cp.hstack(row))
        return cp.vstack(rows)

    def check_local_gp_ppt_relaxation(
        self,
        tau: np.ndarray,
        tau_p: np.ndarray,
        solver: Optional[str] = None,
        tol: Optional[float] = None,
        eps_map: Optional[float] = None,
        eps_gibbs: Optional[float] = None,
        verbose: bool = False,
    ) -> PPTLocalRelaxResult:
        dA, dB = self.dims
        d = dA * dB
        solver_actual = self._select_solver(solver, verbose)
        tol_val = self.tol_default if tol is None else float(tol)
        eps_m = self.eps_eq_local if eps_map is None else float(eps_map)
        eps_g = self.eps_gibbs if eps_gibbs is None else float(eps_gibbs)
        tau_h = 0.5 * (tau + tau.conj().T)
        taup_h = 0.5 * (tau_p + tau_p.conj().T)
        G = np.kron(self.gammaA, self.gammaAp)
        J = cp.Variable((d * d, d * d), complex=True)
        cons = [J >> 0]
        cons += self._choi_tp_constraints(J, d_in=d, d_out=d)
        cons += [cp.norm(self._choi_apply_cvx(J, G, d_in=d, d_out=d) - G, 'fro') <= eps_g]
        cons += [cp.norm(self._choi_apply_cvx(J, tau_h, d_in=d, d_out=d) - taup_h, 'fro') <= eps_m]
        cons += [self._choi_ppt_partial_transpose_cvx(J, dA=dA, dB=dB) >> 0]
        prob = cp.Problem(cp.Minimize(0), cons)
        scs_kwargs = self._scs_kwargs(tol=tol_val, verbose=verbose) if solver_actual == 'SCS' else {'verbose': verbose}
        try:
            prob.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            return PPTLocalRelaxResult(False, f'error: {e}', np.inf, np.inf)
        if prob.status not in ['optimal', 'optimal_inaccurate'] or J.value is None:
            return PPTLocalRelaxResult(False, str(prob.status), np.inf, np.inf)
        Jv = 0.5 * (J.value + J.value.conj().T)
        map_res = float(norm(self.choi_apply_numpy(Jv, tau_h, d_in=d, d_out=d) - taup_h, 'fro'))
        gibbs_res = float(norm(self.choi_apply_numpy(Jv, G, d_in=d, d_out=d) - G, 'fro'))
        feasible = (map_res <= eps_m * 1.1) and (gibbs_res <= eps_g * 1.1)
        return PPTLocalRelaxResult(feasible, str(prob.status), map_res, gibbs_res)

    # ---------- verified local GP defaults ----------

    def check_local_gp_verified(
        self,
        tau: np.ndarray,
        tau_p: np.ndarray,
        solver=None,
        tol=None,
        eps_map: float | None = None,
        eps_gibbs: float | None = None,
        n_random_starts: int = 6,
        seed: int = 0,
        verbose: bool = False,
        return_details: bool = False,
    ):
        return self.check_local_gp_feasible_multistart(
            tau,
            tau_p,
            solver=solver,
            tol=tol,
            eps_map=eps_map,
            eps_gibbs=eps_gibbs,
            n_random_starts=n_random_starts,
            seed=seed,
            verify=True,
            verbose=verbose,
            return_details=return_details,
        )

    def analyze_convertibility(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        verbose: bool = False,
        eps_eq_global=None,
        eps_eq_local=None,
        omega_hint=None,
        local_mode: str = 'verified',
        n_random_starts: int = 6,
        seed: int = 0,
        include_ppt: bool = True,
    ):
        tol = self.tol_default if tol is None else tol
        if eps_eq_global is not None:
            self.eps_eq_global = eps_eq_global
        if eps_eq_local is not None:
            self.eps_eq_local = eps_eq_local
        LT_tau, ltA, ltAp, _tauA, _tauAp = self.lt_membership(tau, tol=1e-8)
        LT_taup, ltA2, ltAp2, _tA2, _tAp2 = self.lt_membership(tau_p, tol=1e-8)
        D_tau, I_tau, C_A, C_Ap = self.monotones(tau)
        D_taup, I_taup, C2_A, C2_Ap = self.monotones(tau_p)
        gp_feas, gp_status, gp_details = self.check_global_gp_feasible(
            tau, tau_p, solver=solver, tol=tol, verbose=verbose, return_details=True
        )
        local_mode = str(local_mode).strip().lower()
        if local_mode in ('verified', 'verify'):
            lgp_feas, lgp_status, lgp_details = self.check_local_gp_verified(
                tau,
                tau_p,
                solver=solver,
                tol=tol,
                eps_map=self.eps_eq_local,
                eps_gibbs=self.eps_gibbs,
                n_random_starts=n_random_starts,
                seed=seed,
                verbose=verbose,
                return_details=True,
            )
        elif local_mode in ('multistart', 'multi'):
            lgp_feas, lgp_status, lgp_details = self.check_local_gp_feasible_multistart(
                tau, tau_p, solver=solver, tol=tol, eps_map=self.eps_eq_local, eps_gibbs=self.eps_gibbs,
                n_random_starts=n_random_starts, seed=seed, verify=False, verbose=verbose, return_details=True,
            )
        else:
            lgp_feas, lgp_status, lgp_details = self.check_local_gp_feasible(
                tau, tau_p, solver=solver, tol=tol, verbose=verbose, omega_hint=omega_hint, return_details=True
            )
            lgp_status = 'HEURISTIC | ' + str(lgp_status)
        ppt = None
        if include_ppt:
            try:
                ppt = self.check_local_gp_ppt_relaxation(tau, tau_p, solver=solver, tol=tol, eps_map=self.eps_eq_local, eps_gibbs=self.eps_gibbs, verbose=False)
            except Exception:
                ppt = None
        return {
            'dims': self.dims,
            'beta': self.beta,
            'LT_tau': LT_tau,
            'LT_tau_breakdown': {'A': ltA, 'Ap': ltAp},
            'LT_taup': LT_taup,
            'LT_taup_breakdown': {'A': ltA2, 'Ap': ltAp2},
            'monotones': {
                'D_tau_vs_gamma': D_tau, 'D_taup_vs_gamma': D_taup,
                'I_tau': I_tau, 'I_taup': I_taup,
                'C_rel_entropy_A_tau': C_A, 'C_rel_entropy_Ap_tau': C_Ap,
                'C_rel_entropy_A_taup': C2_A, 'C_rel_entropy_Ap_taup': C2_Ap,
            },
            'feasibility': {
                'Global_GP': {'feasible': gp_feas, 'status': gp_status, 'details': gp_details},
                'Local_GP': {'feasible': lgp_feas, 'status': lgp_status, 'details': lgp_details, 'mode': local_mode},
                'Local_PPT_outer': None if ppt is None else {
                    'feasible': bool(ppt.feasible), 'status': ppt.status,
                    'map_residual': ppt.map_residual, 'gibbs_residual': ppt.gibbs_residual,
                },
            },
            'diagnostics': {
                'C_tau': self.correlation_metrics(tau),
                'C_taup': self.correlation_metrics(tau_p),
            },
        }

    # ---------- separability / PPT diagnostics ----------

    @staticmethod
    def partial_transpose_b(rho: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
        dA, dB = dims
        X = rho.reshape(dA, dB, dA, dB)
        Xpt = np.transpose(X, axes=(0, 3, 2, 1))
        return Xpt.reshape(dA * dB, dA * dB)

    def ppt_spectrum(self, rho: np.ndarray) -> np.ndarray:
        pt = self.partial_transpose_b(0.5 * (rho + rho.conj().T), self.dims)
        return np.real(np.linalg.eigvalsh(0.5 * (pt + pt.conj().T)))

    def pt_negativity(self, rho: np.ndarray, tol: float = 1e-10) -> float:
        r"""
        Standard PPT-negativity:
            N(
ho) = \sum_{\lambda_k < 0} |\lambda_k(
ho^{T_B})|
                    = (\|
ho^{T_B}\|_1 - 1) / 2.
        For 2x2 and 2x3 this is a direct entanglement witness: N>0 iff the state is entangled.
        """
        ev = self.ppt_spectrum(rho)
        neg = ev[ev < -tol]
        if neg.size == 0:
            return 0.0
        return float(np.sum(-neg))

    def log_negativity(self, rho: np.ndarray, tol: float = 1e-10) -> float:
        neg = self.pt_negativity(rho, tol=tol)
        return float(np.log2(1.0 + 2.0 * neg))

    def ppt_status(self, rho: np.ndarray, tol: float = 1e-10) -> dict:
        ev = self.ppt_spectrum(rho)
        min_ev = float(np.min(ev))
        neg = ev[ev < -tol]
        negativity = float(np.sum(-neg)) if neg.size else 0.0
        log_negativity = float(np.log2(1.0 + 2.0 * negativity))
        is_ppt = bool(min_ev >= -tol)
        sep_cert = None
        entanglement_class = 'unknown'
        if self.dims in ((2, 2), (2, 3), (3, 2)):
            sep_cert = is_ppt
            entanglement_class = 'separable' if is_ppt else 'npt_entangled'
        else:
            entanglement_class = 'ppt_or_bound_entanglement_candidate' if is_ppt else 'npt_entangled'
        return {
            'is_ppt': is_ppt,
            'min_pt_eig': min_ev,
            'pt_negativity': negativity,
            'log_negativity': log_negativity,
            'num_negative_pt_eigs': int(neg.size),
            'negative_pt_eig_sum': negativity,
            'separable_if_low_dim': sep_cert,
            'entanglement_class': entanglement_class,
        }


def default_hamiltonian(d: int, scale: float = 1.0) -> np.ndarray:
    return np.diag(scale * np.arange(int(d), dtype=float))


def parse_variables_string(var_str: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not var_str:
        return out
    for chunk in var_str.split(','):
        chunk = chunk.strip()
        if not chunk or '=' not in chunk:
            continue
        k, v = chunk.split('=', 1)
        k = k.strip(); v = v.strip()
        try:
            if v.lower() in ('true', 'false'):
                out[k] = (v.lower() == 'true')
            elif '.' in v or 'e' in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except Exception:
            out[k] = v
    return out


def random_state(d: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = X @ dagger(X)
    rho = rho / np.trace(rho)
    return 0.5 * (rho + dagger(rho))


class LTStateFactory:
    def __init__(self, system: LTGPSystem):
        self.system = system

    def random_state_dim(self) -> np.ndarray:
        d = self.system.dA * self.system.dAp
        return random_state(d)

    def random_state(self) -> np.ndarray:
        return self.random_state_dim()

    def tfd_state(self) -> np.ndarray:
        if self.system.dA != self.system.dAp:
            raise ValueError('TFD requires equal local dimensions.')
        eA, UA = eigh(0.5 * (self.system.H_A + self.system.H_A.conj().T))
        eB, UB = eigh(0.5 * (self.system.H_Ap + self.system.H_Ap.conj().T))
        if not np.allclose(np.sort(np.real(eA)), np.sort(np.real(eB)), atol=1e-10):
            raise ValueError('TFD requires matching local spectra.')
        p = np.real(np.diag(dagger(UA) @ self.system.gammaA @ UA))
        psi = np.zeros(self.system.dA * self.system.dAp, dtype=complex)
        for i in range(self.system.dA):
            ei = UA[:, i]
            fi = UB[:, i]
            psi += np.sqrt(max(p[i], 0.0)) * np.kron(ei, fi)
        psi = psi / np.linalg.norm(psi)
        rho = np.outer(psi, np.conj(psi))
        return 0.5 * (rho + rho.conj().T)

    def classical_LT_point_qubit(self, a: float) -> np.ndarray:
        if self.system.dims != (2, 2):
            raise ValueError('Only for qubit-qubit.')
        g = np.real(np.diag(self.system.to_energy_basis(self.system.gammaA)))
        g0, g1 = float(g[0]), float(g[1])
        p00 = float(a)
        p01 = g0 - p00
        p10 = g0 - p00
        p11 = 1.0 - p00 - p01 - p10
        P = np.array([[p00, p01], [p10, p11]], dtype=float)
        if np.min(P) < -1e-12:
            raise ValueError('Chosen a produces negative classical probabilities.')
        rho_e = np.diag(P.reshape(4))
        return self.system.from_energy_basis(rho_e)

    def random_classical_LT_qubit(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        g = np.real(np.diag(self.system.to_energy_basis(self.system.gammaA)))
        g0 = float(g[0])
        a_min = max(0.0, 2 * g0 - 1.0)
        a_max = g0
        a = rng.uniform(a_min, a_max)
        return self.classical_LT_point_qubit(a)

    def lt_ray_state_pauli(self, label: str, p: float) -> np.ndarray:
        C0 = self.system.qubit_C0_from_pauli_label(label)
        return self.system.lt_ray_state(C0, p)

    def lt_diagT_state(self, tx: float, ty: float, tz: float, p: float = 1.0) -> np.ndarray:
        C0 = self.system.qubit_C_from_diag_T(tx=tx, ty=ty, tz=tz)
        return self.system.lt_ray_state(C0, p)


class LTAnalyzer:
    def __init__(self, system: LTGPSystem):
        self.system = system
        self.factory = LTStateFactory(system)

    def analyze_single_state(self, rho, label: str = 'state', solver=None, tol=None, verbose: bool = False, classical: bool = True):
        tol = self.system.tol_default if tol is None else tol
        lt_ok, okA, okAp, rhoA, rhoAp = self.system.lt_membership(rho, tol=1e-8)
        D_rho, I_rho, C_A, C_Ap = self.system.monotones(rho, tol=1e-12)
        sigma_lt, dist_lt, status_lt = self.system.closest_lt_state(rho, classical=False, solver=solver, tol=tol, verbose=verbose)
        sigma_cl, dist_cl, status_cl = (None, None, None)
        if classical:
            sigma_cl, dist_cl, status_cl = self.system.closest_lt_state(rho, classical=True, solver=solver, tol=tol, verbose=verbose)
        return {
            'label': label,
            'LT': lt_ok,
            'LT_breakdown': {'A': okA, 'Ap': okAp},
            'marginals': {'A': rhoA, 'Ap': rhoAp},
            'monotones': {
                'D_rho_vs_gamma': D_rho, 'I_rho': I_rho, 'C_A': C_A, 'C_Ap': C_Ap,
            },
            'distance_to_LT': {'distance': dist_lt, 'status': status_lt, 'sigma_closest': sigma_lt},
            'distance_to_classical_LT': {'distance': dist_cl, 'status': status_cl, 'sigma_closest': sigma_cl},
            'correlation': self.system.correlation_metrics(rho),
            'ppt': self.system.ppt_status(rho),
        }

    def analyze_pair(self, tau, tau_p, label: str = 'pair', solver=None, tol=None, verbose: bool = False, local_mode: str = 'verified', n_random_starts: int = 6, seed: int = 0):
        core = self.system.analyze_convertibility(
            tau, tau_p, solver=solver, tol=tol, verbose=verbose,
            local_mode=local_mode, n_random_starts=n_random_starts, seed=seed,
        )
        sigma_tau_LT, dist_tau_LT, status_tau_LT = self.system.closest_lt_state(tau, classical=False, solver=solver, tol=tol, verbose=verbose)
        sigma_taup_LT, dist_taup_LT, status_taup_LT = self.system.closest_lt_state(tau_p, classical=False, solver=solver, tol=tol, verbose=verbose)
        core['analysis_label'] = label
        core['extra_distances'] = {
            'tau': {'distance': dist_tau_LT, 'status': status_tau_LT, 'sigma_closest': sigma_tau_LT},
            'tau_p': {'distance': dist_taup_LT, 'status': status_taup_LT, 'sigma_closest': sigma_taup_LT},
        }
        return core

    def analyze_tfd_vs_dephased(self, solver=None, tol=None, verbose: bool = False):
        tfd = self.factory.tfd_state()
        tfd_deph = self.system.dephase_global_in_energy_basis(tfd)
        return {
            'tfd': self.analyze_single_state(tfd, label='TFD', solver=solver, tol=tol, verbose=verbose),
            'tfd_dephased': self.analyze_single_state(tfd_deph, label='TFD_dephased', solver=solver, tol=tol, verbose=verbose),
            'pair': self.analyze_pair(tfd, tfd_deph, label='TFD_to_dephased', solver=solver, tol=tol, verbose=verbose),
        }

    def sample_extremal_lt_states(self, num_samples: int = 20, classical: bool = False, solver=None, tol=None, verbose: bool = False, seed: int = 0):
        d = self.system.dA * self.system.dAp
        rng = np.random.default_rng(seed)
        extremals = []
        for _ in range(num_samples):
            X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            K = 0.5 * (X + X.conj().T)
            rho_ext, opt_val, status = self.system.extremal_lt_state(K, classical=classical, solver=solver, tol=tol, verbose=verbose)
            if rho_ext is None:
                continue
            extremals.append({
                'rho': rho_ext,
                'K': K,
                'opt_val': opt_val,
                'status': status,
                'monotones': self.analyze_single_state(rho_ext, label='extremal', solver=solver, tol=tol, verbose=False, classical=classical)['monotones'],
            })
        return extremals

    def scan_mixture_with_gamma(self, rho: np.ndarray, lam_list, solver=None, tol=None, verbose: bool = False):
        G = np.kron(self.system.gammaA, self.system.gammaAp)
        out = []
        for lam in lam_list:
            mix = (1 - lam) * rho + lam * G
            mix = 0.5 * (mix + mix.conj().T)
            rep = self.analyze_single_state(mix, label=f'mix_lambda={lam:.4f}', solver=solver, tol=tol, verbose=verbose)
            rep['lambda'] = float(lam)
            out.append(rep)
        return out

    def scan_lt_ray_family_pauli(self, label: str = 'XX', num_points: int = 21, include_negative: bool = False, p_shrink: float = 0.98, tol_psd: float = 1e-12):
        if self.system.dims != (2, 2):
            raise ValueError('Requires dims=(2,2)')
        C0 = self.system.qubit_C0_from_pauli_label(label)
        p_lo, p_hi = self.system.lt_ray_p_bounds(C0, tol=tol_psd)
        p_min_eff = p_lo if include_negative else 0.0
        p_list = np.linspace(p_min_eff * p_shrink, p_hi * p_shrink, int(num_points))
        states = [self.system.lt_ray_state(C0, float(p)) for p in p_list]
        return {'family': 'ray_pauli', 'label': label, 'C0': C0, 'p_bounds': (p_lo, p_hi), 'p_list': p_list, 'states': states}

    def scan_lt_diagT_family(self, t0: tuple[float, float, float] = (1.0, 0.0, 0.0), num_points: int = 21, include_negative: bool = False, p_shrink: float = 0.98, tol_psd: float = 1e-12):
        if self.system.dims != (2, 2):
            raise ValueError('Requires dims=(2,2)')
        t0x, t0y, t0z = [float(x) for x in t0]
        C0 = self.system.qubit_C_from_diag_T(tx=t0x, ty=t0y, tz=t0z)
        p_lo, p_hi = self.system.lt_ray_p_bounds(C0, tol=tol_psd)
        p_min_eff = p_lo if include_negative else 0.0
        p_list = np.linspace(p_min_eff * p_shrink, p_hi * p_shrink, int(num_points))
        states = [self.system.lt_ray_state(C0, float(p)) for p in p_list]
        return {'family': 'diagT_ray', 't0': (t0x, t0y, t0z), 'C0': C0, 'p_bounds': (p_lo, p_hi), 'p_list': p_list, 'states': states}

    def compute_family_observables(self, states: list[np.ndarray]) -> dict:
        I = []
        D = []
        C1 = []
        CF = []
        T_svals = []
        ppt_min = []
        pt_neg = []
        log_neg = []
        ent_class = []
        for rho in states:
            Dk, Ik, _, _ = self.system.monotones(rho, tol=1e-12)
            cm = self.system.correlation_metrics(rho, tol=1e-12)
            ppt = self.system.ppt_status(rho)
            D.append(float(Dk)); I.append(float(Ik)); C1.append(float(cm['C_trace_dist'])); CF.append(float(cm['C_fro']))
            if self.system.dims == (2, 2):
                sv = np.linalg.svd(self.system.qubit_correlation_tensor_T(rho), compute_uv=False)
                T_svals.append(np.sort(np.real(sv))[::-1])
            else:
                T_svals.append(None)
            ppt_min.append(float(ppt['min_pt_eig']))
            pt_neg.append(float(ppt['pt_negativity']))
            log_neg.append(float(ppt['log_negativity']))
            ent_class.append(str(ppt['entanglement_class']))
        return {
            'D': D,
            'I': I,
            'C_trace_dist': C1,
            'C_fro': CF,
            'T_svals': T_svals,
            'ppt_min_eig': ppt_min,
            'pt_negativity': pt_neg,
            'log_negativity': log_neg,
            'entanglement_class': ent_class,
        }


def build_system_and_analyzer(
    dA: int = 2,
    dAp: int = 2,
    beta: float = 1.0,
    solver: str = 'SCS',
    tol: float = 1e-7,
    symmetric: bool = True,
    eps_eq_global: float = 1e-6,
    eps_eq_local: float = 1e-6,
    eps_gibbs: float = 1e-8,
):
    dA = int(dA); dAp = int(dAp)
    if symmetric and dA == dAp:
        H_A = default_hamiltonian(dA, scale=1.0)
        H_Ap = H_A.copy()
    else:
        H_A = default_hamiltonian(dA, scale=1.0)
        scale_ap = 1.0 if dA != dAp else 1.3
        H_Ap = default_hamiltonian(dAp, scale=scale_ap)
    system = LTGPSystem(H_A, H_Ap, beta, solver=solver, tol=tol, eps_eq_global=eps_eq_global, eps_eq_local=eps_eq_local, eps_gibbs=eps_gibbs)
    analyzer = LTAnalyzer(system)
    return system, analyzer


def embed_state_3d(system: LTGPSystem, rho: np.ndarray, rng=None) -> np.ndarray:
    dA, dAp = system.dims
    d = dA * dAp
    if (dA, dAp) == (2, 2):
        B = system.correlation_C(rho)
        sx, sy, sz = paulis()
        return np.array([
            float(np.real(np.trace(B @ np.kron(sx, sx)))),
            float(np.real(np.trace(B @ np.kron(sy, sy)))),
            float(np.real(np.trace(B @ np.kron(sz, sz)))),
        ])
    if rng is None:
        rng = np.random.default_rng(0)
    Os = []
    for _ in range(3):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = 0.5 * (A + dagger(A))
        Os.append(H / (np.linalg.norm(H, 'fro') + 1e-12))
    return np.array([float(np.real(np.trace(rho @ O))) for O in Os])
