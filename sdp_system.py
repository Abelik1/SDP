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
    return float(-np.sum(w * np.log(w)))

def matrix_log_psd(rho, tol=1e-12):
    # log of PSD matrix (on support)
    w, U = eigh((rho + dagger(rho))/2.0)
    w = np.real(w)
    w[w < 0] = 0.0
    with np.errstate(divide='ignore'):
        lw = np.where(w > tol, np.log(w), -np.inf)
    lw_finite = np.where(np.isfinite(lw), lw, 0.0)
    return U @ np.diag(lw_finite) @ dagger(U)

def relative_entropy(rho, sigma, tol=1e-12):
    # D(rho || sigma) = Tr[rho (log rho - log sigma)] on support of rho
    w_sigma, U_sigma = eigh((sigma + dagger(sigma))/2.0)
    w_sigma = np.real(w_sigma)
    w_sigma[w_sigma < 0] = 0.0
    support_sigma = w_sigma > tol

    P_sigma = (
        U_sigma[:, support_sigma] @ dagger(U_sigma[:, support_sigma]) 
        if np.any(support_sigma) else np.zeros_like(sigma)
    )
    if norm((np.eye(rho.shape[0]) - P_sigma) @ rho @ (np.eye(rho.shape[0]) - P_sigma), 2) > tol:
        return np.inf

    log_rho = matrix_log_psd(rho, tol=tol)
    log_sigma = matrix_log_psd(sigma, tol=tol)
    return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

def kron(*args):
    out = np.array([[1.0]])
    for a in args:
        out = np.kron(out, a)
    return out

def partial_trace(rho, dims, keep, tol=1e-12):
    """
    Partial trace over subsystems not in 'keep'.
    dims: list/tuple of subsystem dims [d1, d2, ... , dn]
    keep: tuple/list of subsystem indices to keep (0-based)
    """
    keep = tuple(keep)
    n = len(dims)
    perm = list(keep) + [i for i in range(n) if i not in keep]
    d_keep = int(np.prod([dims[i] for i in keep]))
    d_drop = int(np.prod([dims[i] for i in range(n) if i not in keep]))
    rho_resh = rho.reshape([*dims, *dims])
    order = perm + [i+n for i in perm]
    rho_perm = np.transpose(rho_resh, axes=order)
    rho_perm = rho_perm.reshape(d_keep, d_drop, d_keep, d_drop)
    return np.einsum('ijik->jk', rho_perm).reshape(d_keep, d_keep)

def dephase_in_energy_basis(rho, H, tol=1e-12):
    e, U = eigh((H + dagger(H))/2.0)
    rho_e = dagger(U) @ rho @ U
    rho_e = np.diag(np.diag(rho_e))
    return U @ rho_e @ dagger(U)

def gibbs_state(H, beta):
    # gamma = e^{-beta H}/Z
    e, U = eigh((H + dagger(H))/2.0)
    e = np.real(e)
    g = np.exp(-beta * e)
    Z = np.sum(g)
    G = U @ np.diag(g/Z) @ dagger(U)
    return (G + dagger(G))/2.0

def choi_identity(d):
    """Choi representation of identity channel on d-dim space."""
    v = np.eye(d).reshape(-1, 1, order='F')
    return v @ v.conj().T

# ---------- Mutual information and coherence ----------

def mutual_information(rho, dims, tol=1e-12):
    S_AB = von_neumann_entropy(rho, tol=tol)
    rho_A = partial_trace(rho, dims, keep=[0], tol=tol)
    rho_B = partial_trace(rho, dims, keep=[1], tol=tol)
    return (
        von_neumann_entropy(rho_A, tol=tol)
        + von_neumann_entropy(rho_B, tol=tol)
        - S_AB
    )

def relative_entropy_of_coherence(rho, H, tol=1e-12):
    rho_deph = dephase_in_energy_basis(rho, H, tol=tol)
    return von_neumann_entropy(rho_deph, tol=tol) - von_neumann_entropy(rho, tol=tol)

# ==========================================
# Core class: LTSDPSystem
# ==========================================

class LTSDPSystem:
    """
    Core SDP environment for locally thermal (LT) and Gibbs-preserving (GP) analysis.

    Handles:
      - system definition (H_A, H_Ap, beta),
      - Gibbs states gammaA, gammaAp,
      - LT membership checks,
      - monotones (D, I, coherence),
      - global and local GP feasibility SDPs,
      - support-function SDP for extremal LT states,
      - projection to (classical) LT set via trace-distance SDP.
    """

    def __init__(
        self,
        H_A,
        H_Ap,
        beta,
        solver='SCS',
        tol=1e-7,
        eps_eq_global=1e-8,
        eps_eq_local=1e-6
    ):
        self.H_A = np.array(H_A, dtype=complex)
        self.H_Ap = np.array(H_Ap, dtype=complex)
        self.beta = float(beta)

        self.dA = self.H_A.shape[0]
        self.dAp = self.H_Ap.shape[0]
        self.dims = (self.dA, self.dAp)

        self.gammaA = gibbs_state(self.H_A, self.beta)
        self.gammaAp = gibbs_state(self.H_Ap, self.beta)

        self.solver_default = solver
        self.tol_default = tol
        self.eps_eq_global = eps_eq_global
        self.eps_eq_local = eps_eq_local

    # --------- Basic LT stuff ---------

    def lt_membership(self, rho, tol=1e-8):
        dA, dAp = self.dims
        A  = partial_trace(rho, self.dims, keep=[0], tol=tol)
        Ap = partial_trace(rho, self.dims, keep=[1], tol=tol)
        okA  = np.allclose(A,  self.gammaA,  atol=tol)
        okAp = np.allclose(Ap, self.gammaAp, atol=tol)
        return okA and okAp, okA, okAp, A, Ap

    def monotones(self, rho, tol=1e-12):
        GAxGAp = kron(self.gammaA, self.gammaAp)
        D_rho  = relative_entropy(rho, GAxGAp, tol=tol)
        I_rho  = mutual_information(rho, self.dims, tol=tol)
        rho_A  = partial_trace(rho, self.dims, keep=[0], tol=tol)
        rho_Ap = partial_trace(rho, self.dims, keep=[1], tol=tol)
        C_A    = relative_entropy_of_coherence(rho_A,  self.H_A,  tol=tol)
        C_Ap   = relative_entropy_of_coherence(rho_Ap, self.H_Ap, tol=tol)
        return D_rho, I_rho, C_A, C_Ap

    # --------- Internal helpers for GP SDPs ---------

    def _select_solver(self, solver, verbose=False):
        solver_actual = self.solver_default if solver is None else solver
        if solver_actual.upper() == 'AUTO':
            for s in ['MOSEK', 'COSMO', 'SCS']:
                if s in cp.installed_solvers():
                    solver_actual = s
                    break
        if verbose:
            print(f"Using solver: {solver_actual}")
        return solver_actual

    # ---- Global GP: Choi-based feasibility ----

    def check_global_gp_feasible(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        eps_eq=None,
        verbose=False
    ):
        """
        Check existence of global Gibbs-preserving channel G with:
          G(γ⊗γ) = γ⊗γ (approx),
          G(tau)  = tau_p (approx).
        """
        dA, dAp = self.dims
        d_in = d_out = dA * dAp

        if norm(tau - tau_p, 'fro') <= 1e-10:
            return True, "identity_case"

        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol
        eps_eq = self.eps_eq_global if eps_eq is None else eps_eq

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

        J = cp.Variable((d_out * d_in, d_out * d_in), complex=True, name='J')  # Choi

        I_in = np.eye(d_in)
        GAxGAp = kron(self.gammaA, self.gammaAp)

        def choi_TP_constraints(J_var):
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

        def choi_apply(J_var, X):
            XT = X.T
            blocks = []
            for i in range(d_in):
                acc = 0
                for j in range(d_in):
                    block = J_var[i::d_in, j::d_in]
                    acc += XT[i, j] * block
                blocks.append(acc)
            Y = 0
            for b in blocks:
                Y += b
            return Y

        constraints = [J >> 0]
        constraints += choi_TP_constraints(J)

        GAxGAp_mat = 0.5 * (GAxGAp + GAxGAp.conj().T)
        Y_gp = choi_apply(J, GAxGAp_mat)
        constraints += [cp.norm(Y_gp - GAxGAp_mat, 'fro') <= eps_eq]

        tau_clean   = 0.5 * (tau   + tau.conj().T)
        tau_p_clean = 0.5 * (tau_p + tau_p.conj().T)
        Y_conv = choi_apply(J, tau_clean)
        constraints += [cp.norm(Y_conv - tau_p_clean, 'fro') <= eps_eq]

        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"Global GP solver error: {e}")
            return False, f"error: {str(e)}"

        feasible = (
            prob.status in ['optimal', 'optimal_inaccurate', 'infeasible_inaccurate'] 
            and prob.status != 'infeasible'
        )
        if verbose:
            print(f"Global GP status: {prob.status}, feasible: {feasible}")
        return feasible, prob.status

    # ---- Local GP: two-step feasibility (A then A') ----

    def check_local_gp_feasible(
        self,
        tau,
        tau_p,
        solver=None,
        tol=None,
        eps_eq=None,
        verbose=False,
        omega_hint=None
    ):
        """
        Two-step local GP test:
          Step 1: G_A on A,
          Step 2: G_A' on A'.
        """
        dA, dAp = self.dims

        if norm(tau - tau_p, 'fro') <= 1e-10:
            return True, "identity_case"

        solver_actual = self._select_solver(solver, verbose)
        tol = self.tol_default if tol is None else tol
        eps_eq = self.eps_eq_local if eps_eq is None else eps_eq

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
        JA   = cp.Variable((dA*dA, dA*dA), complex=True, name='J_A')
        omega = cp.Variable((dA*dAp, dA*dAp), complex=True, name='omega')
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
                    s += JA[mu*dA + m, mu*dA + n]
                r.append(s)
            rows.append(r)
        cons1 += [cp.vstack([cp.hstack(r) for r in rows]) == I_A]
        cons1 += [
            cp.norm(
                choi_apply_local(JA, self.gammaA, dA) - self.gammaA,
                'fro'
            ) <= eps_eq
        ]

        tau_blocks = tau.reshape(dA, dAp, dA, dAp)
        omega_expr = 0
        for i in range(dA):
            for j in range(dA):
                Eij = np.zeros((dA, dA), dtype=complex); Eij[i, j] = 1.0
                GA_Eij = choi_apply_local(JA, Eij, dA)
                Tij    = tau_blocks[i, :, j, :]
                omega_expr += cp.kron(GA_Eij, Tij)

        cons1 += [omega >> 0, cp.trace(omega) == 1, omega == omega_expr]

        if omega_hint is not None:
            omega_target = 0.5 * (omega_hint + omega_hint.conj().T)
            obj1 = cp.Minimize(cp.norm(omega - omega_target, 'fro'))
        else:
            tau_target = 0.5 * (tau + tau.conj().T)
            obj1 = cp.Minimize(cp.norm(omega - tau_target, 'fro'))

        prob1 = cp.Problem(obj1, cons1)
        try:
            prob1.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"LGP step-1 solver error: {e}")
            return False, f"LGP step-1 error: {str(e)}"

        if prob1.status not in ['optimal', 'optimal_inaccurate']:
            if verbose:
                print(f"LGP step-1 status: {prob1.status}")
            return False, f"LGP step-1 {prob1.status}"

        omega_val = 0.5 * (omega.value + omega.value.conj().T)

        # -------- STEP 2: channel on A' --------
        JAp  = cp.Variable((dAp*dAp, dAp*dAp), complex=True, name='J_Ap')
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
                    s += JAp[mu*dAp + m, mu*dAp + n]
                r.append(s)
            rows.append(r)
        cons2 += [cp.vstack([cp.hstack(r) for r in rows]) == I_Ap]
        cons2 += [
            cp.norm(
                choi_apply_local(JAp, self.gammaAp, dAp) - self.gammaAp,
                'fro'
            ) <= eps_eq
        ]

        omega_blocks = omega_val.reshape(dA, dAp, dA, dAp)
        tau_p_expr = 0
        for a in range(dAp):
            for b in range(dAp):
                Eab    = np.zeros((dAp, dAp), dtype=complex); Eab[a, b] = 1.0
                GAp_Eab = choi_apply_local(JAp, Eab, dAp)
                Xab     = omega_blocks[:, a, :, b]
                tau_p_expr += cp.kron(Xab, GAp_Eab)

        tau_p_target = 0.5 * (tau_p + tau_p.conj().T)
        obj2 = cp.Minimize(cp.norm(tau_p_expr - tau_p_target, 'fro'))
        prob2 = cp.Problem(obj2, cons2)

        try:
            prob2.solve(solver=solver_actual, **scs_kwargs)
        except Exception as e:
            if verbose:
                print(f"LGP step-2 solver error: {e}")
            return False, f"LGP step-2 error: {str(e)}"

        if prob2.status not in ['optimal', 'optimal_inaccurate']:
            if verbose:
                print(f"LGP step-2 status: {prob2.status}")
            return False, f"LGP step-2 {prob2.status}"

        res = prob2.value if prob2.value is not None else np.inf
        feasible = (res <= eps_eq)
        if verbose:
            print(f"LGP residual: {res:.3e}, threshold: {eps_eq}")
        return feasible, (f"residual={res:.3e}" if res is not None else "optimal")

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

# ==========================================
# Example driver / tests (similar to yours)
# ==========================================

if __name__ == "__main__":
    SEED = 42
    SOLVER = 'MOSEK'    # 'MOSEK', 'COSMO', 'SCS', or 'AUTO'
    TOL = 1e-8
    BETA = 1.0
    TEST_CASE = 'all'
    VERBOSE = False

    np.random.seed(SEED)

    dA = dAp = 2
    H_A  = np.diag([0.0, 1.0])
    H_Ap = np.diag([0.0, 1.5])

    system = LTSDPSystem(H_A, H_Ap, beta=BETA, solver=SOLVER, tol=TOL)

    def rand_state(d):
        X = np.random.randn(d, d) + 1j*np.random.randn(d, d)
        rho = X @ dagger(X)
        return rho / np.trace(rho)

    gammaA  = system.gammaA
    gammaAp = system.gammaAp

    def local_gp_A(rho, lam):
        """Simple local GP replacer on A."""
        rho_Ap = partial_trace(rho, (dA, dAp), keep=[1])
        out = (1 - lam) * rho + lam * np.kron(gammaA, rho_Ap)
        return 0.5 * (out + out.conj().T)

    def local_gp_Ap(rho, lam):
        """Simple local GP replacer on A'."""
        rho_A = partial_trace(rho, (dA, dAp), keep=[0])
        out = (1 - lam) * rho + lam * np.kron(rho_A, gammaAp)
        return 0.5 * (out + out.conj().T)

    def run_test_t0():
        print("\n=== Test T0: Identity Case ===")
        tau = rand_state(dA*dAp)
        tau_p = tau.copy()
        report = system.analyze_convertibility(
            tau, tau_p, solver=SOLVER, tol=TOL, verbose=VERBOSE
        )
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> "
              f"{report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> "
              f"{report['monotones']['I_taup']:.4f}")
        return report

    def run_test_t1():
        print("\n=== Test T1: Global Thermal Mix ===")
        tau = rand_state(dA*dAp)
        lam = 0.3
        GAxGAp = kron(gammaA, gammaAp)
        tau_p = (1 - lam) * tau + lam * GAxGAp
        tau_p = 0.5 * (tau_p + tau_p.conj().T)
        report = system.analyze_convertibility(
            tau, tau_p, solver=SOLVER, tol=TOL, verbose=VERBOSE
        )
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> "
              f"{report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> "
              f"{report['monotones']['I_taup']:.4f}")
        return report

    def run_test_t2():
        print("\n=== Test T2: Guaranteed LGP (with omega_hint) ===")
        tau = rand_state(dA*dAp)
        lamA, lamAp = 0.4, 0.5
        tau1  = local_gp_A(tau, lamA)
        tau_p = local_gp_Ap(tau1, lamAp)
        report = system.analyze_convertibility(
            tau, tau_p, solver=SOLVER, tol=TOL,
            verbose=VERBOSE, omega_hint=tau1
        )
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> "
              f"{report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> "
              f"{report['monotones']['I_taup']:.4f}")
        return report

    if TEST_CASE in ['T0', 'all']:
        run_test_t0()
    if TEST_CASE in ['T1', 'all']:
        run_test_t1()
    if TEST_CASE in ['T2', 'all']:
        run_test_t2()

    print("\nTests complete!")
