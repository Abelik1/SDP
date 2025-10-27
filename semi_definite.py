import numpy as np
import cvxpy as cp
from numpy.linalg import eig, norm
from scipy.linalg import eigh

# ---------- Linear algebra helpers ----------
def dagger(X): return X.conj().T

def safe_eigvals(rho, tol=1e-12):
    # Hermitian eigenvalues (sorted descending); negatives clipped to 0
    w, _ = eigh((rho + dagger(rho)) / 2.0)
    w = np.real(w)
    w[w < 0] = 0.0
    # sort descending just for consistency
    return np.flip(np.sort(w))

def von_neumann_entropy(rho, tol=1e-12):
    w = safe_eigvals(rho, tol=tol)
    w = w[w > tol]
    if w.size == 0: return 0.0
    return float(-np.sum(w * np.log(w)))

def matrix_log_psd(rho, tol=1e-12):
    # log of PSD matrix (on support)
    w, U = eigh((rho + dagger(rho))/2.0)
    w = np.real(w)
    w[w < 0] = 0.0
    # Avoid log(0): we will treat zero-eigs as -inf on nullspace by zeroing their projector contribution
    with np.errstate(divide='ignore'):
        lw = np.where(w > tol, np.log(w), -np.inf)
    # Build log matrix: on nullspace, contribution is 0*Projector so it vanishes
    # We do this by replacing -inf with 0 in the diagonal, consistent with projector times scalar 0.
    lw_finite = np.where(np.isfinite(lw), lw, 0.0)
    return U @ np.diag(lw_finite) @ dagger(U)

def relative_entropy(rho, sigma, tol=1e-12):
    # D(rho || sigma) = Tr[rho (log rho - log sigma)] on support of rho
    # If support(rho) not subset of support(sigma), return +inf
    # We check via eigenvectors with small tol.
    w_sigma, U_sigma = eigh((sigma + dagger(sigma))/2.0)
    w_sigma = np.real(w_sigma)
    w_sigma[w_sigma < 0] = 0.0
    support_sigma = w_sigma > tol

    # Projector onto support(sigma)
    P_sigma = U_sigma[:, support_sigma] @ dagger(U_sigma[:, support_sigma]) if np.any(support_sigma) else np.zeros_like(sigma)

    # If rho has weight outside support(sigma), divergence is inf.
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
    dims: list of subsystem dims [d1, d2, ... , dn]
    keep: tuple/list of subsystem indices to keep (0-based)
    """
    keep = tuple(keep)
    n = len(dims)
    # Permute systems so 'keep' come first
    perm = list(keep) + [i for i in range(n) if i not in keep]
    d_keep = int(np.prod([dims[i] for i in keep]))
    d_drop = int(np.prod([dims[i] for i in range(n) if i not in keep]))
    # reshape to (keep, drop; keep, drop)
    rho_resh = rho.reshape([*dims, *dims])
    # transpose to (keep, drop, keep', drop')
    order = perm + [i+n for i in perm]
    rho_perm = np.transpose(rho_resh, axes=order)
    rho_perm = rho_perm.reshape(d_keep, d_drop, d_keep, d_drop)
    # trace over drop
    return np.einsum('ijik->jk', rho_perm).reshape(d_keep, d_keep)

def dephase_in_energy_basis(rho, H, tol=1e-12):
    # Dephase in the eigenbasis of H
    evals, U = eigh((H + dagger(H))/2.0)
    rho_e = dagger(U) @ rho @ U
    rho_e = np.diag(np.diag(rho_e))
    return U @ rho_e @ dagger(U)

# ---------- Gibbs states ----------
def gibbs_state(H, beta):
    # gamma = e^{-beta H}/Z (computed via eigen-decomposition)
    e, U = eigh((H + dagger(H))/2.0)
    e = np.real(e)
    g = np.exp(-beta * e)
    Z = np.sum(g)
    G = U @ np.diag(g/Z) @ dagger(U)
    # Numerical symmetrization
    return (G + dagger(G))/2.0

def choi_identity(d):
    """Create Choi representation of identity channel."""
    v = np.eye(d).reshape(-1, 1, order='F')  # vec(I) in column-major
    return v @ v.conj().T

def local_gp_A(rho, lam, gammaA, dA, dAp):
    """Apply local GP replacer on A subsystem."""
    rho_Ap = partial_trace(rho, (dA, dAp), keep=[1])
    out = (1 - lam) * rho + lam * np.kron(gammaA, rho_Ap)
    return 0.5 * (out + out.conj().T)

def local_gp_Ap(rho, lam, gammaAp, dA, dAp):
    """Apply local GP replacer on A' subsystem."""
    rho_A = partial_trace(rho, (dA, dAp), keep=[0])
    out = (1 - lam) * rho + lam * np.kron(rho_A, gammaAp)
    return 0.5 * (out + out.conj().T)

# ---------- Mutual information ----------
def mutual_information(rho, dims, tol=1e-12):
    # dims = [dA, dAp]
    S_AB = von_neumann_entropy(rho, tol=tol)
    rho_A = partial_trace(rho, dims, keep=[0], tol=tol)
    rho_B = partial_trace(rho, dims, keep=[1], tol=tol)
    return von_neumann_entropy(rho_A, tol=tol) + von_neumann_entropy(rho_B, tol=tol) - S_AB

# ---------- Relative entropy of coherence in energy basis ----------
def relative_entropy_of_coherence(rho, H, tol=1e-12):
    rho_deph = dephase_in_energy_basis(rho, H, tol=tol)
    return von_neumann_entropy(rho_deph, tol=tol) - von_neumann_entropy(rho, tol=tol)

# ---------- Choi / superoperator utilities ----------
def choi_to_super(J, d_in, d_out):
    """
    Reshuffle Choi J (d_out*d_in x d_out*d_in) to superoperator S (d_out^2 x d_in^2)
    Using index mapping: J_{(μ m),(ν n)} -> S_{(m n),(μ ν)} (col-vec convention)
    Implement via reshape+transpose.
    """
    J4 = J.reshape(d_out, d_in, d_out, d_in)  # indices: μ, m; ν, n
    S4 = np.transpose(J4, (1,3,0,2))          # m, n; μ, ν
    return S4.reshape(d_in*d_in, d_out*d_out).T  # column-stacking to match |ρ>> -> |Φ(ρ)>>

def apply_choi(J, rho_in, d_in, d_out):
    """
    Φ(ρ) = Tr_in[(ρ^T ⊗ I) J] (standard Choi action).
    Implemented by building superoperator and applying to vec.
    """
    S = choi_to_super(J, d_in, d_out)  # (d_out^2 x d_in^2)
    vec_in = rho_in.T.reshape(-1, 1)   # col-vec of ρ: vec(ρ) = (I ⊗ I)col(ρ). We need ρ^T under this convention.
    vec_out = S @ vec_in
    return vec_out.reshape(d_out, d_out).T

def apply_local_A(JA, rho_AA, dA, dAp):
    """
    ω = (G_A ⊗ id)(τ).
    Build S_A from J_A, then act as (S_A ⊗ I) on vec(τ) (col-vec).
    """
    S_A = choi_to_super(JA, dA, dA)            # dA^2 x dA^2
    I_Ap = np.eye(dAp*dAp)
    S_full = np.kron(S_A, I_Ap)                # (dA^2 * dAp^2) x (dA^2 * dAp^2)
    vec_tau = rho_AA.T.reshape(-1, 1)
    vec_out = S_full @ vec_tau
    return vec_out.reshape(dA*dAp, dA*dAp).T

def apply_local_Ap(JAp, rho_AA, dA, dAp):
    """
    τ' = (id ⊗ G_A')(ω)
    """
    S_Ap = choi_to_super(JAp, dAp, dAp)        # dAp^2 x dAp^2
    I_A   = np.eye(dA*dA)
    S_full = np.kron(I_A, S_Ap)
    vec_in = rho_AA.T.reshape(-1, 1)
    vec_out = S_full @ vec_in
    return vec_out.reshape(dA*dAp, dA*dAp).T

# ---------- LT checks and monotones ----------
def lt_membership(tau, gammaA, gammaAp, dims, tol=1e-8):
    dA, dAp = dims
    A  = partial_trace(tau, dims, keep=[0], tol=tol)
    Ap = partial_trace(tau, dims, keep=[1], tol=tol)
    okA  = np.allclose(A,  gammaA,  atol=tol)
    okAp = np.allclose(Ap, gammaAp, atol=tol)
    return okA and okAp, okA, okAp, A, Ap

def monotones(tau, gammaA, gammaAp, dims, H_A=None, H_Ap=None, tol=1e-12):
    GAxGAp = kron(gammaA, gammaAp)
    D_tau  = relative_entropy(tau, GAxGAp, tol=tol)
    I_tau  = mutual_information(tau, dims, tol=tol)
    C_tauA  = relative_entropy_of_coherence(partial_trace(tau, dims, keep=[0]), H_A) if H_A is not None else None
    C_tauAp = relative_entropy_of_coherence(partial_trace(tau, dims, keep=[1]), H_Ap) if H_Ap is not None else None
    return D_tau, I_tau, C_tauA, C_tauAp

# ---------- SDP: Global GP feasibility ----------
def check_global_gp_feasible(tau, tau_p, gammaA, gammaAp, dims, solver='SCS', tol=1e-7, verbose=False, eps_eq=1e-8):
    dA, dAp = dims
    d_in  = dA*dAp
    d_out = dA*dAp
    
    # Identity sanity check
    if norm(tau - tau_p, 'fro') <= 1e-10:
        return True, "identity_case"

    # Solver selection with fallbacks
    solver_actual = solver
    if solver.upper() == 'AUTO':
        for s in ['MOSEK', 'COSMO', 'SCS']:
            if s in cp.installed_solvers():
                solver_actual = s
                break
    
    if verbose:
        print(f"Global GP using solver: {solver_actual}")

    scs_kwargs = {"eps": tol, "max_iters": 200000, "alpha": 1.5, "scale": 5.0, 
                  "normalize": True, "use_indirect": False, "verbose": verbose} if solver_actual.upper() == "SCS" else {}

    J = cp.Variable((d_out*d_in, d_out*d_in), complex=True, name='J')  # Choi of global channel G: in=AA', out=AA'

    I_in = np.eye(d_in)
    GAxGAp = np.kron(gammaA, gammaAp)

    # Constraint helpers in CVXPY (use linear maps via reshape/traces):
    def choi_TP_constraints(J):
        # Tr_out J = I_in  (trace-preserving)
        # Implement blockwise: sum over output indices -> use kron identities
        # We'll enforce via partial trace using index reshapes.
        d = d_out
        din = d_in
        # reshape J to (d_out, d_in, d_out, d_in)
        # Tr_out means trace over first output index pair -> sum over μ=ν on axis 0,2
        # We implement with explicit sums (small-ish dims assumed).
        Jv = []
        for m in range(din):
            row = []
            for n in range(din):
                # entry (m,n) of Tr_out J
                s = 0
                for mu in range(d):
                    s += J[mu*din + m, mu*din + n]
                row.append(s)
            Jv.append(row)
        Tr_out = cp.vstack([cp.hstack(row) for row in Jv])
        return [Tr_out == I_in]

    def choi_apply(J, X):
        # Linear map in CVXPY: Y = Tr_in[(X^T \otimes I) J]
        d = d_out; din = d_in
        XT = X.T
        # Construct sum_{i,j} X^T_{ij} * J_{( :, i ), ( :, j )} as blocks
        blocks = []
        for i in range(din):
            acc = 0
            for j in range(din):
                # block corresponding to |i><j| on input
                block = J[i::din, j::din]  # picks rows/cols stepping by din -> shape (d, d)
                acc += XT[i, j] * block
            blocks.append(acc)
        # sum over i==j already in acc concatenation: now sum these blocks (they're all dxd matrices added)
        Y = 0
        for b in blocks:
            Y += b
        return Y

    constraints = []
    constraints += [J >> 0]  # CP

    # Trace-preserving:
    constraints += choi_TP_constraints(J)

    # Global GP: G(γ⊗γ) = γ⊗γ (relaxed equality)
    GAxGAp_mat = 0.5 * (GAxGAp + GAxGAp.conj().T)
    Y_gp = choi_apply(J, GAxGAp_mat)
    constraints += [cp.norm(Y_gp - GAxGAp_mat, 'fro') <= eps_eq]

    # Conversion: G(tau) = tau' (relaxed equality)
    tau_clean = 0.5 * (tau + tau.conj().T)
    tau_p_clean = 0.5 * (tau_p + tau_p.conj().T)
    Y_conv = choi_apply(J, tau_clean)
    constraints += [cp.norm(Y_conv - tau_p_clean, 'fro') <= eps_eq]

    prob = cp.Problem(cp.Minimize(0), constraints)
    
    try:
        res = prob.solve(solver=solver_actual, **scs_kwargs)
    except Exception as e:
        if verbose:
            print(f"Global GP solver error: {e}")
        return False, f"error: {str(e)}"
    
    feasible = (prob.status in ['optimal', 'optimal_inaccurate', 'infeasible_inaccurate']) and (prob.status != 'infeasible')
    if verbose:
        print(f"Global GP status: {prob.status}, feasible: {feasible}")
    return feasible, prob.status

# ---------- SDP: Local GP (two-step) feasibility ----------
def check_local_gp_feasible(tau, tau_p, gammaA, gammaAp, dims, solver='SCS', tol=1e-7,
                            verbose=False, omega_hint=None, eps_eq=1e-6):
    dA, dAp = dims
    
    # Identity sanity check
    if norm(tau - tau_p, 'fro') <= 1e-10:
        return True, "identity_case"
    
    # Solver selection with fallbacks
    solver_actual = solver
    if solver.upper() == 'AUTO':
        for s in ['MOSEK', 'COSMO', 'SCS']:
            if s in cp.installed_solvers():
                solver_actual = s
                break
    
    if verbose:
        print(f"LGP using solver: {solver_actual}")
    
    scs_kwargs = {"eps": tol, "max_iters": 200000, "alpha": 1.5, "scale": 5.0, 
                  "normalize": True, "use_indirect": False, "verbose": verbose} if solver_actual.upper() == "SCS" else {}

    def choi_apply_local(J, X_const, d):
        XT = X_const.T
        Y = 0
        for i in range(d):
            acc = 0
            for j in range(d):
                block = J[i::d, j::d]
                acc += XT[i, j] * block
            Y += acc
        return Y

    # ---------- STEP 1 ----------
    JA   = cp.Variable((dA*dA, dA*dA), complex=True, name='J_A')
    omega = cp.Variable((dA*dAp, dA*dAp), complex=True, name='omega')
    I_A = np.eye(dA)

    # Warm start J_A
    try:
        JA.value = choi_identity(dA)
    except:
        pass

    cons1 = [JA >> 0]
    # Tr_out J_A = I_A
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
    # Local GP on A (relaxed)
    cons1 += [cp.norm(choi_apply_local(JA, gammaA, dA) - gammaA, 'fro') <= eps_eq]

    # omega = (G_A ⊗ id)(tau)
    tau_blocks = tau.reshape(dA, dAp, dA, dAp)
    omega_expr = 0
    for i in range(dA):
        for j in range(dA):
            Eij = np.zeros((dA, dA), dtype=complex); Eij[i, j] = 1.0
            GA_Eij = choi_apply_local(JA, Eij, dA)
            Tij    = tau_blocks[i, :, j, :]
            omega_expr += cp.kron(GA_Eij, Tij)
    cons1 += [omega >> 0, cp.trace(omega) == 1, omega == omega_expr]

    # Objective based on omega_hint
    if omega_hint is not None:
        omega_target = 0.5 * (omega_hint + omega_hint.conj().T)
        obj1 = cp.Minimize(cp.norm(omega - omega_target, 'fro'))
    else:
        tau_target = 0.5 * (tau + tau.conj().T)
        obj1 = cp.Minimize(cp.norm(omega - tau_target, 'fro'))

    prob1 = cp.Problem(obj1, cons1)
    try:
        _ = prob1.solve(solver=solver_actual, **scs_kwargs)
    except Exception as e:
        if verbose:
            print(f"Step 1 solver error: {e}")
        return False, f"LGP step-1 error: {str(e)}"
    
    if prob1.status not in ['optimal', 'optimal_inaccurate']:
        if verbose:
            print(f"Step 1 status: {prob1.status}")
        return False, f"LGP step-1 {prob1.status}"

    omega_val = 0.5 * (omega.value + omega.value.conj().T)

    # ---------- STEP 2 (min residual) ----------
    JAp  = cp.Variable((dAp*dAp, dAp*dAp), complex=True, name='J_Ap')
    I_Ap = np.eye(dAp)

    # Warm start J_Ap
    try:
        JAp.value = choi_identity(dAp)
    except:
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
    cons2 += [cp.norm(choi_apply_local(JAp, gammaAp, dAp) - gammaAp, 'fro') <= eps_eq]

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
        _ = prob2.solve(solver=solver_actual, **scs_kwargs)
    except Exception as e:
        if verbose:
            print(f"Step 2 solver error: {e}")
        return False, f"LGP step-2 error: {str(e)}"
    
    if prob2.status not in ['optimal', 'optimal_inaccurate']:
        if verbose:
            print(f"Step 2 status: {prob2.status}")
        return False, f"LGP step-2 {prob2.status}"

    res = prob2.value if prob2.value is not None else np.inf
    feasible = (res <= eps_eq)
    
    if verbose:
        print(f"LGP residual: {res:.3e}, threshold: {eps_eq}")
    
    return feasible, (f"residual={res:.3e}" if res is not None else "optimal")
# ---------- Main driver ----------
def analyze_convertibility(H_A, H_Ap, beta, tau, tau_p, solver='SCS', tol=1e-7, verbose=False, **kwargs):
    """
    Inputs:
      H_A, H_Ap : numpy arrays (Hermitian)
      beta      : float
      tau, tau_p: bipartite AA' states (shape (dA*dAp, dA*dAp)), PSD, trace=1
    Returns dict with: LT checks, monotones, GP/LGP feasibility
    """
    dA  = H_A.shape[0]
    dAp = H_Ap.shape[0]
    dims = (dA, dAp)

    gammaA  = gibbs_state(H_A,  beta)
    gammaAp = gibbs_state(H_Ap, beta)

    LT_tau, ltA, ltAp, tauA, tauAp = lt_membership(tau,   gammaA, gammaAp, dims, tol=1e-8)
    LT_taup, ltApA, ltApAp, tA2, tAp2 = lt_membership(tau_p, gammaA, gammaAp, dims, tol=1e-8)

    D_tau,  I_tau,  C_A,  C_Ap  = monotones(tau,   gammaA, gammaAp, dims, H_A, H_Ap)
    D_taup, I_taup, C2_A, C2_Ap = monotones(tau_p, gammaA, gammaAp, dims, H_A, H_Ap)

    # Global GP SDP  
    gp_eps_eq = kwargs.pop('eps_eq', 1e-8)  # Default 1e-8 for global GP
    gp_feas, gp_status = check_global_gp_feasible(tau, tau_p, gammaA, gammaAp, dims, 
                                                   solver=solver, tol=tol, verbose=verbose, eps_eq=gp_eps_eq)

    # Local GP SDP
    lgp_eps_eq = kwargs.get('eps_eq', 1e-6)  # Default 1e-6 for local GP, can be overridden
    lgp_feas, lgp_status = check_local_gp_feasible(tau, tau_p, gammaA, gammaAp, dims,
                                                   solver=solver, tol=tol, verbose=verbose,
                                                   eps_eq=lgp_eps_eq, **kwargs)

    report = {
        "dims": dims,
        "beta": beta,
        "gammaA": gammaA, "gammaAp": gammaAp,
        "LT_tau": LT_tau, "LT_tau_breakdown": {"A": ltA, "Ap": ltAp},
        "LT_taup": LT_taup, "LT_taup_breakdown": {"A": ltApA, "Ap": ltApAp},
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

# ---------- Main execution and Tests ----------
if __name__ == "__main__":
    # ===== CONFIGURATION VARIABLES =====
    # Change these values to test different scenarios
    SEED = 42                    # Random seed for reproducibility
    SOLVER = 'MOSEK'            # Solver: 'MOSEK', 'COSMO', 'SCS', or 'AUTO'
    TOL = 1e-8                  # Solver tolerance
    EPS_EQ = 1e-6               # Equality constraint tolerance
    BETA = 1.0                  # Inverse temperature
    TEST_CASE = 'all'           # Test case: 'T0', 'T1', 'T2', or 'all'
    VERBOSE = False             # Verbose output
    
    # ===== SETUP =====
    # Set seed
    np.random.seed(SEED)
    
    # System dimensions
    dA = dAp = 2
    H_A  = np.diag([0.0, 1.0])
    H_Ap = np.diag([0.0, 1.5])
    beta = BETA

    def rand_state(d):
        X = np.random.randn(d,d) + 1j*np.random.randn(d,d)
        rho = X @ dagger(X)
        return rho / np.trace(rho)

    gammaA  = gibbs_state(H_A,  beta)
    gammaAp = gibbs_state(H_Ap, beta)
    
    # ===== TEST FUNCTIONS =====
    def run_test_t0():
        """Test T0: Identity Case"""
        print("\n=== Test T0: Identity Case ===")
        tau = rand_state(dA*dAp)
        tau_p = tau.copy()
        
        report = analyze_convertibility(H_A, H_Ap, beta, tau, tau_p, 
                                        solver=SOLVER, tol=TOL, 
                                        eps_eq=EPS_EQ, verbose=VERBOSE)
        
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> {report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> {report['monotones']['I_taup']:.4f}")
        return report

    def run_test_t1():
        """Test T1: Global Thermal Mix"""
        print("\n=== Test T1: Global Thermal Mix ===")
        tau = rand_state(dA*dAp)
        lam = 0.3
        GAxGAp = np.kron(gammaA, gammaAp)
        tau_p = (1 - lam) * tau + lam * GAxGAp
        tau_p = 0.5 * (tau_p + tau_p.conj().T)
        
        report = analyze_convertibility(H_A, H_Ap, beta, tau, tau_p, 
                                        solver=SOLVER, tol=TOL, 
                                        eps_eq=EPS_EQ, verbose=VERBOSE)
        
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> {report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> {report['monotones']['I_taup']:.4f}")
        return report

    def run_test_t2():
        """Test T2: Guaranteed LGP (with omega_hint)"""
        print("\n=== Test T2: Guaranteed LGP (with omega_hint) ===")
        tau = rand_state(dA*dAp)
        lamA, lamAp = 0.4, 0.5
        
        tau1  = local_gp_A(tau, lamA, gammaA, dA, dAp)
        tau_p = local_gp_Ap(tau1, lamAp, gammaAp, dA, dAp)
        
        report = analyze_convertibility(H_A, H_Ap, beta, tau, tau_p, 
                                        solver=SOLVER, tol=TOL, 
                                        eps_eq=EPS_EQ, omega_hint=tau1,
                                        verbose=VERBOSE)
        
        print(f"Global GP: {report['feasibility']['Global_GP']}")
        print(f"Local GP:  {report['feasibility']['Local_GP']}")
        print(f"D: {report['monotones']['D_tau_vs_gamma']:.4f} -> {report['monotones']['D_taup_vs_gamma']:.4f}")
        print(f"I: {report['monotones']['I_tau']:.4f} -> {report['monotones']['I_taup']:.4f}")
        return report

    # ===== RUN TESTS =====
    if TEST_CASE in ['T0', 'all']:
        run_test_t0()
    
    if TEST_CASE in ['T1', 'all']:
        run_test_t1()
    
    if TEST_CASE in ['T2', 'all']:
        run_test_t2()
    
    print("\nTests complete!")
