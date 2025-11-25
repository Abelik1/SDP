from sdp_system import LTSDPSystem, dagger, kron, partial_trace

import numpy as np
from scipy.linalg import eigh
from numpy.linalg import norm

# assuming LTSDPSystem is already imported / defined in this file
# from lt_sdp_system import LTSDPSystem, dagger, kron, partial_trace



# ==========================================================
#  1. State factory: build various test states systematically
# ==========================================================

class LTStateFactory:
    """
    Utility class for constructing states to feed into LTSDPSystem.

    Typical use:
      factory = LTStateFactory(system)
      tau     = factory.random_state()
      tfd     = factory.tfd_state()
      rho_cl  = factory.classical_LT_point(param=0.3)
    """

    def __init__(self, system):
        """
        system: LTSDPSystem instance (defines H_A, H_Ap, beta, gammaA, gammaAp, dims)
        """
        self.system = system
        self.dA, self.dAp = system.dims
        self.d = self.dA * self.dAp
        self.gammaA = system.gammaA
        self.gammaAp = system.gammaAp

    # ---------- general helpers ----------

    @staticmethod
    def random_state_dim(d, seed=None):
        """Random full-rank density matrix on dimension d."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        rho = X @ dagger(X)
        return rho / np.trace(rho)

    def random_state(self, seed=None):
        """Random full-rank state on A⊗A'."""
        return self.random_state_dim(self.d, seed=seed)

    # ---------- TFD-like states ----------

    def tfd_state(self):
        """
        Construct a TFD-like purification between A and A'.
        This assumes H_A == H_Ap (same spectrum & eigenbasis) so that
        gammaA == gammaAp and the TFD is locally thermal with marginal gamma.

        If H_A and H_Ap differ, this will raise a ValueError.
        """
        # Diagonalise H_A and H_Ap
        eA, UA = eigh((self.system.H_A + dagger(self.system.H_A)) / 2.0)
        eAp, UAp = eigh((self.system.H_Ap + dagger(self.system.H_Ap)) / 2.0)

        # Check compatibility: same dimension and (approximately) same spectrum
        if self.dA != self.dAp or not np.allclose(eA, eAp, atol=1e-10):
            raise ValueError(
                "TFD construction here assumes H_A and H_Ap have the same spectrum/basis."
            )

        beta = self.system.beta
        g = np.exp(-beta * eA)
        g = g / np.sum(g)
        # Build TFD in the joint eigenbasis: |TFD> = sum_n sqrt(g_n) |n>_A |n>_A'
        d = self.dA
        psi = np.zeros((d, d), dtype=complex)
        for n in range(d):
            psi[n, n] = np.sqrt(g[n])
        # Convert to full Hilbert space basis (if UA != I)
        psi_full = (UA @ psi @ UAp.T)  # shape (d, d)
        ket_TFD = psi_full.reshape(-1, 1)  # vector in A⊗A'
        rho_TFD = ket_TFD @ dagger(ket_TFD)
        return rho_TFD / np.trace(rho_TFD)

    # ---------- Classical LT states (2x2 case) ----------

    def classical_LT_point_qubit(self, a=None):
        """
        For dA = dAp = 2, generate a classical (diagonal) LT state with
        marginals gammaA, gammaAp (assumed equal).

        We use the 2x2 transportation polytope parameterisation:

          p(a) = [[a,         g0 - a],
                  [g0 - a,    g1 - g0 + a]]

        with a in [2g0 - 1, g0].

        If `a` is None, choose a random point in that interval.
        """
        if self.dA != 2 or self.dAp != 2:
            raise ValueError("classical_LT_point_qubit is only implemented for 2x2.")

        # Extract Gibbs probabilities from gammaA (assume diagonal in energy basis)
        # We diagonalise to get eigenvalues; ordering doesn't matter for diagonals.
        evals, U = eigh((self.gammaA + dagger(self.gammaA)) / 2.0)
        g = np.real(evals)
        # Sort descending just for consistency
        g = np.flip(np.sort(g))
        g0, g1 = g

        a_min = 2 * g0 - 1.0
        a_max = g0

        if a is None:
            rng = np.random.default_rng()
            a = rng.uniform(a_min, a_max)
        else:
            if not (a_min - 1e-10 <= a <= a_max + 1e-10):
                raise ValueError(f"a must lie in [{a_min}, {a_max}] for LT marginals.")

        p00 = a
        p01 = g0 - a
        p10 = g0 - a
        p11 = g1 - g0 + a

        p = np.array([p00, p01, p10, p11], dtype=float)
        if np.any(p < -1e-12):
            raise RuntimeError("Computed probabilities are negative: not LT.")

        # Diagonal state in computational / energy product basis
        rho_diag = np.diag(p)
        rho_diag = rho_diag / np.trace(rho_diag)
        return rho_diag

    # ---------- Generic classical LT sample (2x2) ----------

    def random_classical_LT_qubit(self):
        """Sample a random classical LT point in the 2x2 case."""
        return self.classical_LT_point_qubit(a=None)


# ==========================================================
#  2. Analysis class: wrap SDPs + factories into experiments
# ==========================================================

class LTAnalyzer:
    """
    High-level analysis class around an LTSDPSystem.

    Responsibilities:
      - Construct interesting state pairs (tau, tau'),
      - Call system.analyze_convertibility on them,
      - Compute distances to LT / classical LT,
      - Provide convenience routines for scanning families (e.g. mixtures).
    """

    def __init__(self, system):
        """
        system : LTSDPSystem instance
        """
        self.system = system
        self.factory = LTStateFactory(system)

    # ---------- Basic analysis of a single state ----------

    def analyze_single_state(self, rho, label=None, solver=None, tol=None, verbose=False):
        """
        Analyze a single state:
          - LT membership,
          - monotones D, I, coherence,
          - distance to LT set,
          - distance to classical LT set (2x2 only).

        Returns a dict with all of the above.
        """
        label = label or "state"
        dA, dAp = self.system.dims

        # LT membership & monotones
        LT_flag, ltA, ltAp, rhoA, rhoAp = self.system.lt_membership(rho, tol=1e-8)
        D_rho, I_rho, C_A, C_Ap = self.system.monotones(rho)

        # Projection to LT (distance to full LT set)
        sigma_LT, dist_LT, status_LT = self.system.closest_lt_state(
            rho, classical=False, solver=solver, tol=tol, verbose=verbose
        )

        # Projection to classical LT (if 2x2)
        sigma_cl, dist_cl, status_cl = None, None, None
        if dA == 2 and dAp == 2:
            sigma_cl, dist_cl, status_cl = self.system.closest_lt_state(
                rho, classical=True, solver=solver, tol=tol, verbose=verbose
            )

        report = {
            "label": label,
            "LT_membership": {
                "is_LT": LT_flag,
                "A_matches_gamma": ltA,
                "Ap_matches_gamma": ltAp,
                "rhoA": rhoA,
                "rhoAp": rhoAp,
            },
            "monotones": {
                "D_rho_vs_gamma": D_rho,
                "I_rho": I_rho,
                "C_rel_entropy_A": C_A,
                "C_rel_entropy_Ap": C_Ap,
            },
            "distance_to_LT": {
                "distance": dist_LT,
                "status": status_LT,
                "sigma_closest": sigma_LT,
            },
            "distance_to_classical_LT": {
                "distance": dist_cl,
                "status": status_cl,
                "sigma_closest": sigma_cl,
            },
        }
        return report

    # ---------- Analyze convertibility for one pair (tau -> tau') ----------

    def analyze_pair(
        self,
        tau,
        tau_p,
        label=None,
        solver=None,
        tol=None,
        verbose=False,
        omega_hint=None,
        eps_eq_global=None,
        eps_eq_local=None
    ):
        """
        Wraps system.analyze_convertibility with optional extra distances
        to classical LT for tau and tau'.

        Returns a dict with:
          - everything from LTSDPSystem.analyze_convertibility,
          - plus distances to classical LT for tau and tau_p.
        """
        label = label or "tau_to_taup"

        # Core convertibility analysis
        report_core = self.system.analyze_convertibility(
            tau,
            tau_p,
            solver=solver,
            tol=tol,
            verbose=verbose,
            eps_eq_global=eps_eq_global,
            eps_eq_local=eps_eq_local,
            omega_hint=omega_hint,
        )

        # Distances to classical LT (2x2 check)
        dA, dAp = self.system.dims
        dist_tau_cl, dist_taup_cl = None, None
        status_tau_cl, status_taup_cl = None, None

        if dA == 2 and dAp == 2:
            sigma_tau_cl, dist_tau_cl, status_tau_cl = self.system.closest_lt_state(
                tau, classical=True, solver=solver, tol=tol, verbose=verbose
            )
            sigma_taup_cl, dist_taup_cl, status_taup_cl = self.system.closest_lt_state(
                tau_p, classical=True, solver=solver, tol=tol, verbose=verbose
            )
        else:
            sigma_tau_cl = sigma_taup_cl = None

        # Distances to full LT
        sigma_tau_LT,  dist_tau_LT,  status_tau_LT  = self.system.closest_lt_state(
            tau, classical=False, solver=solver, tol=tol, verbose=verbose
        )
        sigma_taup_LT, dist_taup_LT, status_taup_LT = self.system.closest_lt_state(
            tau_p, classical=False, solver=solver, tol=tol, verbose=verbose
        )

        extra = {
            "label": label,
            "distances_tau": {
                "to_LT": {
                    "distance": dist_tau_LT,
                    "status": status_tau_LT,
                    "sigma_closest": sigma_tau_LT,
                },
                "to_classical_LT": {
                    "distance": dist_tau_cl,
                    "status": status_tau_cl,
                    "sigma_closest": sigma_tau_cl,
                },
            },
            "distances_taup": {
                "to_LT": {
                    "distance": dist_taup_LT,
                    "status": status_taup_LT,
                    "sigma_closest": sigma_taup_LT,
                },
                "to_classical_LT": {
                    "distance": dist_taup_cl,
                    "status": status_taup_cl,
                    "sigma_closest": sigma_taup_cl,
                },
            },
        }

        report_core["analysis_label"] = label
        report_core["extra_distances"] = extra
        return report_core

    # ---------- Example families / scans ----------

    def scan_classical_LT_line_qubit(self, num_points=21, solver=None, tol=None, verbose=False):
        """
        For dA=dAp=2: sample points along the classical LT line p(a),
        analyze each, and return a list of reports.

        This is useful to map out:
          - I(A:B) vs classical correlation strength,
          - distance to LT (should be 0 by definition),
          - how far classical points are from the TFD-like top layer.
        """
        if self.system.dims != (2, 2):
            raise ValueError("scan_classical_LT_line_qubit only valid for 2x2.")

        # Get interval [a_min, a_max] from factory
        # We'll hack it by calling classical_LT_point_qubit twice.
        # First call to get parameters:
        rho_sample = self.factory.classical_LT_point_qubit(a=None)
        # Recover g0 from gammaA
        evals, _ = eigh((self.system.gammaA + dagger(self.system.gammaA)) / 2.0)
        g = np.flip(np.sort(np.real(evals)))
        g0 = g[0]
        a_min = 2*g0 - 1.0
        a_max = g0

        a_values = np.linspace(a_min, a_max, num_points)
        reports = []
        for a in a_values:
            rho_cl = self.factory.classical_LT_point_qubit(a=a)
            rep = self.analyze_single_state(
                rho_cl,
                label=f"classical_LT_a={a:.4f}",
                solver=solver,
                tol=tol,
                verbose=verbose,
            )
            rep["a"] = a
            reports.append(rep)
        return reports

    def scan_mixture_with_gamma(self, rho, lam_list, solver=None, tol=None, verbose=False):
        """
        For a given state rho, consider:
          rho(lam) = (1-lam) rho + lam * gammaA⊗gammaAp

        Scan over lam_list and analyze each. Good for visualising how
        global athermality decays as you "thermalise" toward γ⊗γ.
        """
        GAxGAp = kron(self.system.gammaA, self.system.gammaAp)
        reports = []
        for lam in lam_list:
            mix = (1 - lam) * rho + lam * GAxGAp
            mix = 0.5 * (mix + mix.conj().T)
            rep = self.analyze_single_state(
                mix,
                label=f"mix_lambda={lam:.4f}",
                solver=solver,
                tol=tol,
                verbose=verbose,
            )
            rep["lambda"] = lam
            reports.append(rep)
        return reports

    def analyze_tfd_vs_dephased(self, solver=None, tol=None, verbose=False):
        """
        Convenience method:
          - build a TFD state (if possible),
          - dephase it in local energy bases (A and A'),
          - analyze both, and compare.

        This shows very cleanly:
          - TFD is LT and globally very athermal (coherent),
          - dephased version is classical LT on the same marginals.
        """
        tfd = self.factory.tfd_state()
        # dephase locally in energy basis
        eA, UA = eigh((self.system.H_A + dagger(self.system.H_A)) / 2.0)
        eAp, UAp = eigh((self.system.H_Ap + dagger(self.system.H_Ap)) / 2.0)

        # Projectors onto energy eigenbases
        U_tot = np.kron(UA, UAp)
        rho_e = dagger(U_tot) @ tfd @ U_tot
        rho_e_deph = np.diag(np.diag(rho_e))
        tfd_deph = U_tot @ rho_e_deph @ dagger(U_tot)

        rep_tfd = self.analyze_single_state(
            tfd, label="TFD", solver=solver, tol=tol, verbose=verbose
        )
        rep_deph = self.analyze_single_state(
            tfd_deph, label="TFD_dephased", solver=solver, tol=tol, verbose=verbose
        )

        return {"tfd": rep_tfd, "tfd_dephased": rep_deph}
