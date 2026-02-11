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
        Apply Choi matrix J_var to a constant operator X_const.

        For Choi J of shape (d_out*d_in, d_out*d_in), the action is:
          Φ(X) = Tr_in[J (I_out ⊗ X^T)]
        Implemented via stride-block extraction (works for small dims).
        """
        XT = X_const.T
        blocks = []
        for i in range(d_in):
            acc = 0
            for j in range(d_in):
                block = J_var[i::d_in, j::d_in]  # each block is d_out x d_out
                acc += XT[i, j] * block
            blocks.append(acc)
        Y = 0
        for b in blocks:
            Y += b
        return Y

    @staticmethod
    def choi_apply_numpy(J: np.ndarray, X: np.ndarray, d_in: int, d_out: int):
        """
        Numpy version of Choi application (same as _choi_apply_cvx).
        """
        XT = X.T
        Y = np.zeros((d_out, d_out), dtype=complex)
        for i in range(d_in):
            acc = np.zeros((d_out, d_out), dtype=complex)
            for j in range(d_in):
                block = J[i::d_in, j::d_in]
                acc += XT[i, j] * block
            Y += acc
        return Y

    @staticmethod
    def kraus_from_choi(J: np.ndarray, d_in: int, d_out: int, tol: float = 1e-12):
        """
        Extract Kraus operators {K_k} from a Choi matrix J via eigendecomposition:
          J = Σ_k λ_k |v_k⟩⟨v_k|,  K_k = sqrt(λ_k) reshape(v_k, (d_out,d_in), order='F')
        """
        Jh = 0.5 * (J + J.conj().T)
        w, V = eigh(Jh)
        kraus = []
        for lam, v in zip(w, V.T):
            lam = float(np.real(lam))
            if lam <= tol:
                continue
            K = np.sqrt(lam) * v.reshape((d_out, d_in), order="F")
            kraus.append(K)
        return kraus

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
        """
        dA, dAp = self.dims

        tau_clean = 0.5 * (tau + tau.conj().T)
        tau_p_clean = 0.5 * (tau_p + tau_p.conj().T)

        if norm(tau_clean - tau_p_clean, "fro") <= 1e-12:
            if return_details:
                return True, "identity_case", {"residual": 0.0, "J_A": None, "J_Ap": None, "omega": tau_clean}
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
            if return_details:
                return False, f"LGP step-1 error: {str(e)}", {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": None}
            return False, f"LGP step-1 error: {str(e)}"

        if prob1.status not in ["optimal", "optimal_inaccurate"]:
            if verbose:
                print(f"LGP step-1 status: {prob1.status}")
            if return_details:
                return False, f"LGP step-1 {prob1.status}", {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": None}
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
            if return_details:
                return False, f"LGP step-2 error: {str(e)}", {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": omega_val}
            return False, f"LGP step-2 error: {str(e)}"

        if prob2.status not in ["optimal", "optimal_inaccurate"]:
            if verbose:
                print(f"LGP step-2 status: {prob2.status}")
            if return_details:
                return False, f"LGP step-2 {prob2.status}", {"residual": np.inf, "J_A": None, "J_Ap": None, "omega": omega_val}
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
        }

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
