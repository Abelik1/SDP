import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

from sdp_system import (
    dagger,
    kron,
    gibbs_state,
    mutual_information,
    relative_entropy,
    relative_entropy_of_coherence,
    partial_trace,
)


# ==========================================
# LT state factory
# ==========================================

class LTStateFactory:
    def __init__(self, system):
        self.system = system

    def random_state_dim(self, d):
        X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho = X @ dagger(X)
        return rho / np.trace(rho)

    def random_state(self):
        dA, dAp = self.system.dims
        return self.random_state_dim(dA * dAp)

    def tfd_state(self):
        H_A = self.system.H_A
        H_Ap = self.system.H_Ap
        beta = self.system.beta

        # Need matching spectra for TFD-like construction
        wA, UA = eigh((H_A + dagger(H_A)) / 2.0)
        wAp, UAp = eigh((H_Ap + dagger(H_Ap)) / 2.0)

        wA = np.real(wA)
        wAp = np.real(wAp)

        if wA.shape != wAp.shape or norm(np.sort(wA) - np.sort(wAp)) > 1e-8:
            raise ValueError("TFD requires identical energy spectra. Use symmetric=True.")

        # Thermal probabilities
        wA = wA - np.min(wA)
        p = np.exp(-beta * wA)
        p = p / np.sum(p)

        # Build |TFD> in energy eigenbasis, then rotate back to computational basis
        d = len(p)
        psi = np.zeros((d * d,), dtype=complex)
        for i in range(d):
            psi[i * d + i] = np.sqrt(p[i])
        rho = np.outer(psi, psi.conj())

        U_tot = np.kron(UA, UAp)
        rho = U_tot @ rho @ dagger(U_tot)
        return 0.5 * (rho + dagger(rho))

    def classical_LT_point_qubit(self, a=None):
        # For d=2, classical LT line parameterized by a in [2g0-1, g0]
        gammaA = self.system.gammaA
        evals, _ = eigh((gammaA + dagger(gammaA)) / 2.0)
        g = np.flip(np.sort(np.real(evals)))
        g0 = g[0]
        a_min = 2 * g0 - 1.0
        a_max = g0

        if a is None:
            a = 0.5 * (a_min + a_max)

        a = float(a)
        a = min(max(a, a_min), a_max)

        # p00=a, p01=g0-a, p10=g0-a, p11=1-2g0+a
        p00 = a
        p01 = g0 - a
        p10 = g0 - a
        p11 = 1 - 2 * g0 + a

        diag = np.array([p00, p01, p10, p11], dtype=float)
        rho = np.diag(diag.astype(complex))
        return rho / np.trace(rho)

    def random_classical_LT_qubit(self):
        gammaA = self.system.gammaA
        evals, _ = eigh((gammaA + dagger(gammaA)) / 2.0)
        g = np.flip(np.sort(np.real(evals)))
        g0 = g[0]
        a_min = 2 * g0 - 1.0
        a_max = g0
        a = np.random.uniform(a_min, a_max)
        return self.classical_LT_point_qubit(a=a)


# ==========================================
# High-level analysis / experiment helper
# ==========================================

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

    def sample_extremal_lt_states(
        self,
        num_samples=20,
        classical=False,
        solver=None,
        tol=None,
        verbose=False,
    ):
        """
        Sample extremal LT (or classical LT) states by maximising Tr(K rho)
        over the LT set using extremal_lt_state(K, classical=...).

        Returns a list of dicts with rho, K, status, and monotones.
        """
        dA, dAp = self.system.dims
        d = dA * dAp

        def random_Hermitian():
            X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
            H = X + dagger(X)
            return 0.5 * H  # ensure Hermitian

        extremals = []
        for _ in range(num_samples):
            K = random_Hermitian()
            rho_ext, opt_val, status = self.system.extremal_lt_state(
                K,
                classical=classical,
                solver=solver,
                tol=tol,
                verbose=verbose,
            )
            if rho_ext is None:
                continue

            D_rho, I_rho, C_A, C_Ap = self.system.monotones(rho_ext)

            extremals.append(
                {
                    "rho": rho_ext,
                    "K": K,
                    "opt_val": opt_val,
                    "status": status,
                    "monotones": {
                        "D_rho_vs_gamma": D_rho,
                        "I_rho": I_rho,
                        "C_A": C_A,
                        "C_Ap": C_Ap,
                    },
                }
            )

        return extremals
