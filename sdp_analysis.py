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
    # ------------------------------------------
    # Structured LT family generators (2x2 only)
    # ------------------------------------------

    def lt_ray_state_pauli(self, label: str, p: float) -> np.ndarray:
        """ρ(p)=γ⊗γ + p C0 with C0=(1/4)σ_i⊗σ_j."""
        C0 = self.system.qubit_C0_from_pauli_label(label)
        return self.system.lt_ray_state(C0, p)

    def lt_diagT_state(self, tx: float, ty: float, tz: float) -> np.ndarray:
        """ρ=γ⊗γ + (1/4)(tx XX + ty YY + tz ZZ)."""
        C = self.system.qubit_C_from_diag_T(tx=tx, ty=ty, tz=tz)
        return self.system.lt_ray_state(C, 1.0)


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
    # ==========================================
    # Structured LT families + monotone validation
    # ==========================================

    def scan_lt_ray_family_pauli(
        self,
        label: str = "XX",
        num_points: int = 21,
        p_min: float | None = None,
        p_max: float | None = None,
        p_shrink: float = 0.98,
        include_negative: bool = False,
        tol_psd: float = 1e-12,
    ) -> dict:
        """
        Build a 1D LT ray ρ(p)=γ⊗γ + p C0 with C0=(1/4)σ_i⊗σ_j.

        - If (p_min,p_max) not provided, uses analytic PSD bounds from the whitening condition.
        - By default scans p∈[0,p_max]. Set include_negative=True to scan symmetric range.

        Returns dict with keys: p_list, states, C0, p_bounds.
        """
        if self.system.dims != (2, 2):
            raise ValueError("scan_lt_ray_family_pauli requires dims=(2,2)")

        C0 = self.system.qubit_C0_from_pauli_label(label)
        p_lo, p_hi = self.system.lt_ray_p_bounds(C0, tol=tol_psd)

        if p_min is None or p_max is None:
            p_min_eff = p_lo if include_negative else 0.0
            p_max_eff = p_hi
        else:
            p_min_eff = float(p_min)
            p_max_eff = float(p_max)

        p_min_eff *= p_shrink
        p_max_eff *= p_shrink

        p_list = np.linspace(p_min_eff, p_max_eff, int(num_points))
        states = [self.system.lt_ray_state(C0, float(p)) for p in p_list]

        return {
            "family": "ray_pauli",
            "label": label,
            "C0": C0,
            "p_bounds": (p_lo, p_hi),
            "p_list": p_list,
            "states": states,
        }

    def scan_lt_diagT_family(
        self,
        t0: tuple[float, float, float] = (1.0, 0.0, 0.0),
        num_points: int = 21,
        p_min: float | None = None,
        p_max: float | None = None,
        p_shrink: float = 0.98,
        include_negative: bool = False,
        tol_psd: float = 1e-12,
    ) -> dict:
        """
        Diagonal correlation-tensor ray:
          ρ(p) = γ⊗γ + p * C0,
          C0 = (1/4)(t0x XX + t0y YY + t0z ZZ).

        Returns dict with keys: p_list, states, C0, t0, p_bounds.
        """
        if self.system.dims != (2, 2):
            raise ValueError("scan_lt_diagT_family requires dims=(2,2)")

        t0x, t0y, t0z = [float(x) for x in t0]
        C0 = self.system.qubit_C_from_diag_T(tx=t0x, ty=t0y, tz=t0z)
        p_lo, p_hi = self.system.lt_ray_p_bounds(C0, tol=tol_psd)

        if p_min is None or p_max is None:
            p_min_eff = p_lo if include_negative else 0.0
            p_max_eff = p_hi
        else:
            p_min_eff = float(p_min)
            p_max_eff = float(p_max)

        p_min_eff *= p_shrink
        p_max_eff *= p_shrink

        p_list = np.linspace(p_min_eff, p_max_eff, int(num_points))
        states = [self.system.lt_ray_state(C0, float(p)) for p in p_list]

        return {
            "family": "diagT_ray",
            "t0": (t0x, t0y, t0z),
            "C0": C0,
            "p_bounds": (p_lo, p_hi),
            "p_list": p_list,
            "states": states,
        }

    def compute_family_observables(self, states: list[np.ndarray]) -> dict:
        """Compute I, D, ||C||_1/2, ||C||_F, and singular values of Pauli correlation tensor T."""
        sys = self.system

        I = []
        D = []
        C1 = []      # 1/2||C||_1
        CF = []      # ||C||_F
        Tsvals = []  # singular values of 3x3 correlation tensor
        Tdiag = []   # diag entries (diagnostic)

        for rho in states:
            Dv, Iv, _, _ = sys.monotones(rho)
            cm = sys.correlation_metrics(rho)
            I.append(float(Iv))
            D.append(float(Dv))
            C1.append(float(cm["C_trace_dist"]))
            CF.append(float(cm["C_fro"]))

            if sys.dims == (2, 2):
                T = sys.qubit_correlation_tensor_T(rho, use_C=True)
                s = np.linalg.svd(T, compute_uv=False)
                Tsvals.append(np.sort(np.real(s))[::-1])
                Tdiag.append(np.real(np.diag(T)))
            else:
                Tsvals.append(None)
                Tdiag.append(None)

        return {
            "I": np.array(I, dtype=float),
            "D": np.array(D, dtype=float),
            "C_trace_dist": np.array(C1, dtype=float),
            "C_fro": np.array(CF, dtype=float),
            "T_svals": Tsvals,
            "T_diag": Tdiag,
        }

    def validate_local_gp_monotones_on_ray(
        self,
        p_list: np.ndarray,
        states: list[np.ndarray],
        pair_mode: str = "adjacent",
        solver: str | None = None,
        tol: float | None = None,
        eps_map_local: float | None = None,
        eps_gibbs: float | None = None,
        mono_tol: float = 1e-9,
        verbose: bool = False,
    ) -> dict:
        """
        For p1>p2, test local GP feasibility + verify monotone inequalities on feasible edges.

        Checks on feasible edges:
          - I decreases:        I(p1) >= I(p2)
          - ||C||_1 decreases: ||C1|| >= ||C2||   (here stored as 0.5||C||_1)
          - svals(T) contract: svals(T1) majorize svals(T2)

        Returns dict with all edge results and any violations.
        """
        sys = self.system
        obs = self.compute_family_observables(states)
        I = obs["I"]
        C1 = obs["C_trace_dist"]
        Ts = obs["T_svals"]

        idx_pairs = []
        n = len(states)
        if pair_mode.lower() == "all":
            for i in range(n):
                for j in range(n):
                    if p_list[i] > p_list[j]:
                        idx_pairs.append((i, j))
        elif pair_mode.lower() == "adjacent":
            for i in range(1, n):
                if p_list[i] > p_list[i - 1]:
                    idx_pairs.append((i, i - 1))
                else:
                    idx_pairs.append((i - 1, i))
        else:
            raise ValueError("pair_mode must be 'adjacent' or 'all'")

        edges = []
        violations = []

        for (i, j) in idx_pairs:
            src = states[i]
            tgt = states[j]

            l_ok, l_status, l_det = sys.check_local_gp_feasible(
                src,
                tgt,
                solver=solver,
                tol=tol,
                eps_map=eps_map_local,
                eps_gibbs=eps_gibbs,
                verbose=verbose,
                return_details=True,
            )
            res = float(l_det.get("residual", np.inf))

            edge = {
                "i": i,
                "j": j,
                "p_i": float(p_list[i]),
                "p_j": float(p_list[j]),
                "local_feasible": bool(l_ok),
                "status": str(l_status),
                "residual": res,
                "I_i": float(I[i]),
                "I_j": float(I[j]),
                "C1_i": float(C1[i]),
                "C1_j": float(C1[j]),
            }

            if sys.dims == (2, 2):
                s_i = np.array(Ts[i], dtype=float)
                s_j = np.array(Ts[j], dtype=float)
                edge["T_svals_i"] = s_i
                edge["T_svals_j"] = s_j

            if l_ok:
                ok_I = (I[i] + mono_tol >= I[j])
                ok_C = (C1[i] + mono_tol >= C1[j])
                ok_T = True
                if sys.dims == (2, 2):
                    ok_T = sys.majorization_holds(s_i, s_j, tol=mono_tol)

                edge["ineq_ok"] = bool(ok_I and ok_C and ok_T)
                edge["ineq_detail"] = {
                    "I_contracts": bool(ok_I),
                    "C1_contracts": bool(ok_C),
                    "T_svals_majorize": bool(ok_T),
                }
                if not edge["ineq_ok"]:
                    violations.append(edge)

            edges.append(edge)

        return {
            "p_list": p_list,
            "observables": obs,
            "edges": edges,
            "violations": violations,
        }




    # ==========================================
    # NEW: Characterise qubit diag-T C-space (2D/3D)
    # ==========================================

    def diagT_plane_boundary(
        self,
        axes: tuple[str, str] = ("x", "z"),
        num_angles: int = 181,
        include_negative: bool = True,
        tol: float = 1e-12,
    ) -> dict:
        """
        Boundary of the feasible LT slice in a 2D plane of diag correlation coordinates.

        We restrict to qubit-diag correlations:
          C = (1/4)(tx XX + ty YY + tz ZZ),
        but only two coordinates are active, chosen by `axes`, e.g. ('x','z') means ty=0.

        For each direction u(θ) in that plane, compute ray bounds p_min,p_max for
          ρ(p) = γ⊗γ + p C(u),
        and convert to boundary coordinates t = p*u.

        Returns:
          {
            "axes": axes,
            "theta": (num_angles,),
            "u": (num_angles,3),
            "p_min": (num_angles,),
            "p_max": (num_angles,),
            "t_min": (num_angles,3),   # boundary points for p_min
            "t_max": (num_angles,3),   # boundary points for p_max
          }
        """
        system = self.system
        if system.dims != (2, 2):
            raise ValueError("diagT_plane_boundary requires dims=(2,2)")

        axmap = {"x": 0, "y": 1, "z": 2}
        a1 = str(axes[0]).lower().strip()
        a2 = str(axes[1]).lower().strip()
        if a1 not in axmap or a2 not in axmap or a1 == a2:
            raise ValueError("axes must be two distinct elements from {'x','y','z'}")

        k1 = axmap[a1]
        k2 = axmap[a2]

        theta = np.linspace(0.0, 2.0 * np.pi, int(num_angles), endpoint=True)
        u = np.zeros((theta.size, 3), dtype=float)
        u[:, k1] = np.cos(theta)
        u[:, k2] = np.sin(theta)

        p_min = np.zeros(theta.size, dtype=float)
        p_max = np.zeros(theta.size, dtype=float)

        for i in range(theta.size):
            tx, ty, tz = u[i]
            C0 = system.qubit_C_from_diag_T(tx, ty, tz)
            pmin, pmax = system.lt_ray_p_bounds(C0, tol=tol)
            p_min[i] = pmin
            p_max[i] = pmax

        t_min = p_min[:, None] * u
        t_max = p_max[:, None] * u

        out = {
            "axes": (a1, a2),
            "theta": theta,
            "u": u,
            "p_min": p_min,
            "p_max": p_max,
            "t_min": t_min,
            "t_max": t_max,
        }
        if not include_negative:
            # keep the "positive" side only (p_max)
            out["t_min"] = None
            out["p_min"] = None
        return out

    @staticmethod
    def _fibonacci_sphere(num_dirs: int) -> np.ndarray:
        """Return (num_dirs,3) approximately-uniform unit vectors on S^2."""
        n = int(num_dirs)
        if n <= 0:
            return np.zeros((0, 3), dtype=float)
        i = np.arange(n, dtype=float) + 0.5
        phi = 2.0 * np.pi * i / ((1.0 + 5.0 ** 0.5) / 2.0)
        z = 1.0 - 2.0 * i / n
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        u = np.stack([x, y, z], axis=1)
        return u

    def diagT_3d_boundary(
        self,
        num_dirs: int = 600,
        include_negative: bool = True,
        tol: float = 1e-12,
    ) -> dict:
        """
        Boundary of the feasible LT slice in full (tx,ty,tz) diag correlation space.

        For each unit direction u on S^2, compute ray bounds p_min,p_max for
          ρ(p) = γ⊗γ + p C(u)
        where C(u) = (1/4)(u_x XX + u_y YY + u_z ZZ), and convert to boundary points
          t = p*u.

        Returns:
          {
            "u": (num_dirs,3),
            "p_min": (num_dirs,),
            "p_max": (num_dirs,),
            "t_min": (num_dirs,3),
            "t_max": (num_dirs,3),
          }
        """
        system = self.system
        if system.dims != (2, 2):
            raise ValueError("diagT_3d_boundary requires dims=(2,2)")

        u = self._fibonacci_sphere(num_dirs)
        p_min = np.zeros(u.shape[0], dtype=float)
        p_max = np.zeros(u.shape[0], dtype=float)

        for i in range(u.shape[0]):
            tx, ty, tz = u[i]
            C0 = system.qubit_C_from_diag_T(tx, ty, tz)
            pmin, pmax = system.lt_ray_p_bounds(C0, tol=tol)
            p_min[i] = pmin
            p_max[i] = pmax

        t_min = p_min[:, None] * u
        t_max = p_max[:, None] * u

        out = {
            "u": u,
            "p_min": p_min,
            "p_max": p_max,
            "t_min": t_min,
            "t_max": t_max,
        }
        if not include_negative:
            out["t_min"] = None
            out["p_min"] = None
        return out
