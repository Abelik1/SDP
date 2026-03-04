from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cvxpy as cp

from sdp_system import LTSDPSystem, dagger


def pauli_basis_orthonormal() -> list[np.ndarray]:
    """Orthonormal traceless basis for d=2: {σx/√2, σy/√2, σz/√2}. Tr(Bi Bj)=δij."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return [sx / np.sqrt(2.0), sy / np.sqrt(2.0), sz / np.sqrt(2.0)]


def gell_mann_basis_orthonormal() -> list[np.ndarray]:
    """Orthonormal traceless basis for d=3: {λi/√2}_{i=1..8}. Tr(λi λj)=2δij."""
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


class LTGPSystem(LTSDPSystem):
    """Extends your LTSDPSystem with:
    - d=3 Gell-Mann parameterization
    - d=3 commuting LT sampling (transport polytope via Sinkhorn)
    - convex outer relaxation for local GP via PPT constraint on joint Choi
    """

    # ---------- LT parameterizations (2,2 and 3,3) ----------

    def lt_from_correlation_matrix(self, C: np.ndarray) -> np.ndarray:
        """ρ = γ⊗γ + Σ Cij (Bi⊗Bj), with Bi traceless ⇒ LT marginals preserved."""
        dA, dB = self.dims
        G = np.kron(self.gammaA, self.gammaAp)

        if (dA, dB) == (2, 2):
            B = pauli_basis_orthonormal()
            if C.shape != (3, 3):
                raise ValueError("For d=2, expected C shape (3,3).")
        elif (dA, dB) == (3, 3):
            B = gell_mann_basis_orthonormal()
            if C.shape != (8, 8):
                raise ValueError("For d=3, expected C shape (8,8).")
        else:
            raise NotImplementedError("Implemented for (2,2) and (3,3).")

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
            raise NotImplementedError("Implemented for (2,2) and (3,3).")

        C = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                C[i, j] = float(np.real(np.trace(C_op @ np.kron(B[i], B[j]))))
        return C

    # ---------- Ray feasibility bounds ----------

    def lt_ray_bounds(self, C0: np.ndarray, tol: float = 1e-12) -> Tuple[float, float]:
        """Analytic PSD interval for ρ(p)=γ⊗γ + pC0 via whitening eigenvalues."""
        C0h = 0.5 * (C0 + C0.conj().T)
        GinvA = _invsqrt_psd(self.gammaA, tol=tol)
        GinvB = _invsqrt_psd(self.gammaAp, tol=tol)
        W = np.kron(GinvA, GinvB)
        Ct = 0.5 * (W @ C0h @ W + (W @ C0h @ W).conj().T)
        lam = np.real(np.linalg.eigvalsh(Ct))

        p_min, p_max = -np.inf, +np.inf
        for x in lam:
            if x > tol:
                p_min = max(p_min, -1.0 / x)
            elif x < -tol:
                p_max = min(p_max, -1.0 / x)

        if not np.isfinite(p_min): p_min = -1e6
        if not np.isfinite(p_max): p_max = +1e6
        return float(p_min), float(p_max)

    # ---------- d=3 commuting LT subclass ----------

    def sample_commuting_lt_d3(self, n: int, seed: int = 0, iters: int = 300) -> list[np.ndarray]:
        """Sample commuting (energy-diagonal) LT states for dims=(3,3) via Sinkhorn scaling."""
        if self.dims != (3, 3):
            raise ValueError("sample_commuting_lt_d3 requires dims=(3,3).")

        rng = np.random.default_rng(int(seed))

        wA, UA = np.linalg.eigh(0.5 * (self.H_A + self.H_A.conj().T))
        wB, UB = np.linalg.eigh(0.5 * (self.H_Ap + self.H_Ap.conj().T))
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

    # ---------- PPT outer relaxation for local GP ----------

    @staticmethod
    def _choi_ppt_partial_transpose_cvx(J: cp.Expression, dA: int, dB: int) -> cp.Expression:
        """Partial transpose on Bob for Choi J of channel on AB, across (Aout,Ain)|(Bout,Bin)."""
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

                r_src = idx(out_r_src, in_r_src)
                c_src = idx(out_c_src, in_c_src)
                row.append(J[r_src, c_src])
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
        """Outer relaxation: CPTP + GP + mapping + PPT(Choi) across (AoutAin)|(BoutBin)."""
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

        PhiG = self._choi_apply_cvx(J, G, d_in=d, d_out=d)
        Phitau = self._choi_apply_cvx(J, tau_h, d_in=d, d_out=d)
        cons += [cp.norm(PhiG - G, "fro") <= eps_g]
        cons += [cp.norm(Phitau - taup_h, "fro") <= eps_m]

        J_pt = self._choi_ppt_partial_transpose_cvx(J, dA=dA, dB=dB)
        cons += [J_pt >> 0]

        prob = cp.Problem(cp.Minimize(0), cons)
        scs_kwargs = self._scs_kwargs(tol=tol_val, verbose=verbose) if solver_actual == "SCS" else {"verbose": verbose}

        try:
            prob.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            return PPTLocalRelaxResult(False, f"error: {e}", np.inf, np.inf)

        if prob.status not in ["optimal", "optimal_inaccurate"] or J.value is None:
            return PPTLocalRelaxResult(False, str(prob.status), np.inf, np.inf)

        Jv = J.value
        PhiG_np = self.choi_apply_numpy(Jv, G, d_in=d, d_out=d)
        Phitau_np = self.choi_apply_numpy(Jv, tau_h, d_in=d, d_out=d)
        map_res = float(np.linalg.norm(Phitau_np - taup_h, ord="fro"))
        gibbs_res = float(np.linalg.norm(PhiG_np - G, ord="fro"))

        feasible = (map_res <= eps_m * 1.1) and (gibbs_res <= eps_g * 1.1)
        return PPTLocalRelaxResult(feasible, str(prob.status), map_res, gibbs_res)