import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt

from sdp_system import LTSDPSystem, dagger
from sdp_analysis import LTAnalyzer
from sdp_gui import LTSDPWindow


# =========================
# File/plot utilities
# =========================

def ensure_png_dir():
    folder = "png"
    os.makedirs(folder, exist_ok=True)
    return folder

def save_plot(fig, filename):
    folder = ensure_png_dir()
    full_path = os.path.join(folder, filename)
    fig.savefig(full_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return full_path

def log_info(title, text):
    print(f"\n[INFO] {title}\n{text}\n")

def log_warning(title, text):
    print(f"\n⚠️ [WARNING] {title}\n{text}\n")

def log_error(title, text):
    print(f"\n☠️ [ERROR] {title}\n{text}\n")


# =========================
# Config parsing
# =========================

def parse_variables_string(var_str):
    """Parse 'k1=v1, k2=v2' into dict with int/float conversion."""
    out = {}
    if not var_str:
        return out
    for chunk in var_str.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            elif "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except Exception:
            out[k] = v
    return out


# =========================
# System builder
# =========================

def default_hamiltonian(d, scale=1.0):
    """Default non-degenerate Hamiltonian: diag(0,1,2,...)*(scale)."""
    return np.diag(scale * np.arange(d, dtype=float))

def build_system_and_analyzer(
    dA=2,
    dAp=2,
    beta=1.0,
    solver="SCS",
    tol=1e-7,
    symmetric=True,
    eps_eq_global=1e-8,
    eps_eq_local=1e-6,
    eps_gibbs=1e-8,
):
    dA = int(dA); dAp = int(dAp)
    if symmetric and dA == dAp:
        H_A = default_hamiltonian(dA, scale=1.0)
        H_Ap = H_A.copy()
    else:
        H_A = default_hamiltonian(dA, scale=1.0)
        # If dimensions match but "symmetric" unchecked, make them slightly different.
        scale_ap = 1.0 if dA != dAp else 1.3
        H_Ap = default_hamiltonian(dAp, scale=scale_ap)

    system = LTSDPSystem(
        H_A, H_Ap, beta,
        solver=solver,
        tol=tol,
        eps_eq_global=eps_eq_global,
        eps_eq_local=eps_eq_local,
        eps_gibbs=eps_gibbs,
    )
    analyzer = LTAnalyzer(system)
    return system, analyzer


# =========================
# State/geometry helpers
# =========================

def random_state(d):
    X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = X @ dagger(X)
    rho = rho / np.trace(rho)
    return 0.5 * (rho + dagger(rho))

def paulis():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz

def embed_state_3d(system, rho, rng=None):
    """Return 3D coordinates for plotting points. For 2-qubits uses <σi⊗σi>; otherwise random observables."""
    dA, dAp = system.dims
    d = dA * dAp
    if (dA, dAp) == (2, 2):
        sx, sy, sz = paulis()
        X = np.kron(sx, sx)
        Y = np.kron(sy, sy)
        Z = np.kron(sz, sz)
        coords = [
            float(np.real(np.trace(rho @ X))),
            float(np.real(np.trace(rho @ Y))),
            float(np.real(np.trace(rho @ Z))),
        ]
        return np.array(coords)

    # General dims: use 3 fixed random Hermitians (seeded) so plot is stable
    if rng is None:
        rng = np.random.default_rng(0)
    Os = []
    for _ in range(3):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = 0.5 * (A + dagger(A))
        Os.append(H / (np.linalg.norm(H, "fro") + 1e-12))
    coords = [float(np.real(np.trace(rho @ O))) for O in Os]
    return np.array(coords)


# =========================
# Main GUI entry
# =========================

def main():
    # default starting system (the GUI can override all of this)
    np.random.seed(0)
    system, analyzer = build_system_and_analyzer(
        dA=2, dAp=2, beta=1.0, solver="SCS", tol=1e-7,
        symmetric=True,
        eps_eq_global=1e-8,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    )

    app = QtWidgets.QApplication(sys.argv)

    def backend_run(config):
        nonlocal system, analyzer

        eq_id = config.get("selected_equation_id")
        eq_name = config.get("selected_equation_name", "")
        vars_str = config.get("variables_str", config.get("variables", ""))
        custom_func_text = config.get("custom_function", "")

        print("\n[backend_run] Selected equation:", eq_id, f"({eq_name})")
        print("[backend_run] Variables string:", vars_str)
        print("[backend_run] Custom text length:", len(custom_func_text))

        vars_dict = parse_variables_string(vars_str)
        print("[backend_run] Parsed variables:", vars_dict)

        # -----------------------
        # Pull UI-controlled vars
        # -----------------------
        dA = int(vars_dict.get("dA", system.dA))
        dAp = int(vars_dict.get("dAp", system.dAp))
        beta = float(vars_dict.get("beta", system.beta))

        symmetric_flag = bool(vars_dict.get("symmetric", True))
        if symmetric_flag and dA != dAp:
            symmetric_flag = False  # can't be symmetric if dims differ

        reset_system = bool(vars_dict.get("reset_system", False))

        solver_override = str(vars_dict.get("solver", system.solver_default))
        eps_eq_global = float(vars_dict.get("eps_eq_global", system.eps_eq_global))
        eps_eq_local = float(vars_dict.get("eps_eq_local", system.eps_eq_local))
        eps_gibbs = float(vars_dict.get("eps_gibbs", getattr(system, "eps_gibbs", 1e-8)))

        # -----------------------
        # Rebuild system if needed
        # -----------------------
        needs_rebuild = (
            reset_system
            or (system.dA != dA)
            or (system.dAp != dAp)
            or (abs(system.beta - beta) > 1e-15)
            or (str(system.solver_default) != solver_override)
            or (not np.allclose(system.H_Ap, system.H_A) and symmetric_flag)
        )

        if needs_rebuild:
            log_info(
                "System rebuild",
                f"dA={dA}, dAp={dAp}, beta={beta}, solver={solver_override}, symmetric={symmetric_flag}\n"
                f"eps_eq_global={eps_eq_global}, eps_eq_local={eps_eq_local}, eps_gibbs={eps_gibbs}"
            )
            system, analyzer = build_system_and_analyzer(
                dA=dA,
                dAp=dAp,
                beta=beta,
                solver=solver_override,
                tol=system.tol_default,
                symmetric=symmetric_flag,
                eps_eq_global=eps_eq_global,
                eps_eq_local=eps_eq_local,
                eps_gibbs=eps_gibbs,
            )
        else:
            # Always update eps values even if no rebuild
            system.eps_eq_global = eps_eq_global
            system.eps_eq_local = eps_eq_local
            system.eps_gibbs = eps_gibbs
            system.solver_default = solver_override

        # seed (negative = don't touch global rng)
        seed = int(vars_dict.get("seed", -1))
        if seed >= 0:
            np.random.seed(seed)

        d = system.dA * system.dAp

        # Helper: global dephasing in energy eigenbasis
        def dephase_global_in_energy_basis(rho):
            H_A = system.H_A
            H_Ap = system.H_Ap
            eA, UA = np.linalg.eigh((H_A + dagger(H_A)) / 2.0)
            eAp, UAp = np.linalg.eigh((H_Ap + dagger(H_Ap)) / 2.0)
            U_tot = np.kron(UA, UAp)
            rho_e = dagger(U_tot) @ rho @ U_tot
            rho_e_deph = np.diag(np.diag(rho_e))
            return 0.5 * (U_tot @ rho_e_deph @ dagger(U_tot) + dagger(U_tot @ rho_e_deph @ dagger(U_tot)))

        # ====================================================
        # Dispatch
        # ====================================================

        if eq_id == "tfd_vs_dephased":
            try:
                results = analyzer.analyze_tfd_vs_dephased(
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                rep_tfd = results["tfd"]
                rep_deph = results["tfd_dephased"]

                tfd_state = analyzer.factory.tfd_state()
                tfd_deph = rep_deph["distance_to_LT"]["sigma_closest"]  # should be itself (LT)
                if tfd_deph is None:
                    tfd_deph = dephase_global_in_energy_basis(tfd_state)

                conv_report = analyzer.analyze_pair(
                    tfd_state,
                    tfd_deph,
                    label="TFD_to_TFD_deph",
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                    eps_eq_global=system.eps_eq_global,
                    eps_eq_local=system.eps_eq_local,
                )

                gp = conv_report["feasibility"]["Global_GP"]
                lgp = conv_report["feasibility"]["Local_GP"]

                text = (
                    "TFD vs dephased TFD\n\n"
                    "=== Monotones ===\n"
                    f"TFD: I(A:B)={rep_tfd['monotones']['I_rho']:.4f}, D(ρ||γ⊗γ)={rep_tfd['monotones']['D_rho_vs_gamma']:.4f}\n"
                    f"Dephased: I(A:B)={rep_deph['monotones']['I_rho']:.4f}, D(ρ||γ⊗γ)={rep_deph['monotones']['D_rho_vs_gamma']:.4f}\n\n"
                    "=== Convertibility TFD → dephased ===\n"
                    f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                    f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n"
                )
                log_info("TFD vs Dephased TFD", text)
                print("Finished TFD vs dephased analysis.")
            except Exception as e:
                log_error(
                    "TFD Error",
                    "TFD analysis failed. This typically means symmetric=True is required (H_A == H_A').\n\n"
                    f"Details:\n{e}"
                )

        elif eq_id == "random_pair_gp_lgp":
            tau = random_state(d)
            tau_p = random_state(d)
            report = analyzer.analyze_pair(
                tau,
                tau_p,
                label="random_pair",
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            gp = report["feasibility"]["Global_GP"]
            lgp = report["feasibility"]["Local_GP"]

            text = (
                "Random τ → τ' convertibility\n\n"
                f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n\n"
                f"D_tau  = {report['monotones']['D_tau_vs_gamma']:.4f}\n"
                f"D_taup = {report['monotones']['D_taup_vs_gamma']:.4f}\n"
                f"I_tau  = {report['monotones']['I_tau']:.4f}\n"
                f"I_taup = {report['monotones']['I_taup']:.4f}\n"
            )
            log_info("Random Pair GP / LGP Test", text)

            fig, ax = plt.subplots()
            ax.scatter(
                [report["monotones"]["D_tau_vs_gamma"], report["monotones"]["D_taup_vs_gamma"]],
                [report["monotones"]["I_tau"], report["monotones"]["I_taup"]],
            )
            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_title("Random pair in (D,I) plane")
            ax.annotate("τ", (report["monotones"]["D_tau_vs_gamma"], report["monotones"]["I_tau"]))
            ax.annotate("τ'", (report["monotones"]["D_taup_vs_gamma"], report["monotones"]["I_taup"]))
            path = save_plot(fig, "random_pair_DI.png")
            log_info("Random Pair Plot", f"Saved:\n{path}")

        elif eq_id == "mix_with_gamma":
            # Choose a "source" state: TFD if symmetric and dims match, else random.
            try:
                rho0 = analyzer.factory.tfd_state()
                label = "TFD"
            except Exception:
                rho0 = random_state(d)
                label = "random"

            lam_grid = np.linspace(0.0, 1.0, int(vars_dict.get("num_samples", 25)))
            reports = analyzer.scan_mixture_with_gamma(
                rho0,
                lam_grid,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )

            lams = [r["lambda"] for r in reports]
            Dvals = [r["monotones"]["D_rho_vs_gamma"] for r in reports]
            Ivals = [r["monotones"]["I_rho"] for r in reports]
            distLT = [r["distance_to_LT"]["distance"] for r in reports]

            fig, ax = plt.subplots()
            ax.plot(lams, Dvals, label="D(ρ||γ⊗γ)")
            ax.plot(lams, Ivals, label="I(A:B)")
            ax.plot(lams, distLT, label="dist_to_LT")
            ax.set_xlabel("λ in (1−λ)ρ + λ γ⊗γ")
            ax.set_title(f"Mix with γ⊗γ starting from {label}")
            ax.legend()
            path = save_plot(fig, "mix_with_gamma.png")
            log_info("Mixture with γ⊗γ", f"Saved plot:\n{path}")
            print("Finished mixture with gamma analysis.")

        elif eq_id == "closest_lt_distance":
            rho = random_state(d)
            rep = analyzer.analyze_single_state(
                rho,
                label="random_rho",
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            text = (
                "Distance to LT (random state)\n\n"
                f"Is LT? {rep['LT_membership']['is_LT']}\n"
                f"Dist to LT (trace): {rep['distance_to_LT']['distance']:.4e} (status={rep['distance_to_LT']['status']})\n"
            )
            if rep["distance_to_classical_LT"]["distance"] is not None:
                text += (
                    f"Dist to classical LT (trace): {rep['distance_to_classical_LT']['distance']:.4e} "
                    f"(status={rep['distance_to_classical_LT']['status']})\n"
                )
            log_info("Closest LT Distance", text)
            print("Finished closest LT distance analysis.") 

        elif eq_id == "lt_region_geometry":
            num_samples = int(vars_dict.get("num_samples", 50))
            classical_flag = bool(vars_dict.get("classical", False))

            extremals = analyzer.sample_extremal_lt_states(
                num_samples=num_samples,
                classical=classical_flag,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            if not extremals:
                log_warning("LT Region Geometry", "No extremal LT states found (all SDPs failed).")
                return

            D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in extremals]
            I_vals = [rep["monotones"]["I_rho"] for rep in extremals]

            fig, ax = plt.subplots()
            ax.scatter(D_vals, I_vals)
            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_title("Extremal LT boundary (support-function samples)")
            path = save_plot(fig, "lt_region_geometry_DI.png")
            log_info("LT Region Geometry", f"Sampled {len(extremals)} extremals. Saved:\n{path}")
            print("Finished LT region geometry analysis.")

        elif eq_id == "lt_interior_geometry":
            num_samples = int(vars_dict.get("num_samples", 200))

            projected = []
            for _ in range(num_samples):
                rho = random_state(d)

                sigma_LT, _, st1 = system.closest_lt_state(
                    rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                sigma_cl, dist_cl, st2 = system.closest_lt_state(
                    rho, classical=True, solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                if sigma_LT is None or sigma_cl is None:
                    continue

                D_val, I_val, _, _ = system.monotones(sigma_LT)
                projected.append({"rho": sigma_LT, "D": D_val, "I": I_val, "dist_classical": dist_cl})

            if not projected:
                log_warning("LT Interior Geometry", "No LT projections succeeded.")
                return

            D_vals = [p["D"] for p in projected]
            I_vals = [p["I"] for p in projected]
            Z_vals = [p["dist_classical"] for p in projected]

            # 3D plot (D, I, dist_to_classical)
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(D_vals, I_vals, Z_vals)
            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_zlabel("dist_to_classical_LT")
            ax.set_title("LT interior: random → LT projection")
            fig.colorbar(sc, ax=ax, pad=0.1, label="Distance to classical LT")
            path = save_plot(fig, "lt_interior_geometry_3d.png")
            log_info("LT Interior Geometry", f"Projected {len(projected)} states. Saved:\n{path}")
            print("Finished LT interior geometry analysis.")

        elif eq_id == "lt_geometry_combined":
            # Final figure: interior + boundary + classical line (if qubits)
            num_samples = int(vars_dict.get("num_samples", 200))
            classical_flag = bool(vars_dict.get("classical", False))

            n_interior = max(20, num_samples)
            n_boundary = max(20, min(100, num_samples // 2))

            # Interior points via projection
            interior = []
            for _ in range(n_interior):
                rho = random_state(d)
                sigma_LT, _, _ = system.closest_lt_state(
                    rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False
                )
                if sigma_LT is None:
                    continue
                D_val, I_val, _, _ = system.monotones(sigma_LT)
                interior.append((D_val, I_val))

            # Boundary extremals
            extremals = analyzer.sample_extremal_lt_states(
                num_samples=n_boundary,
                classical=classical_flag,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            boundary = [(rep["monotones"]["D_rho_vs_gamma"], rep["monotones"]["I_rho"]) for rep in extremals]

            fig, ax = plt.subplots()
            if interior:
                ax.scatter([p[0] for p in interior], [p[1] for p in interior], alpha=0.25, label="interior (proj)")
            if boundary:
                ax.scatter([p[0] for p in boundary], [p[1] for p in boundary], alpha=0.9, label="boundary (extremal)")

            # classical LT line for qubits (if available)
            if system.dims == (2, 2):
                reports_cl = analyzer.scan_classical_LT_line_qubit(
                    num_points=60,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                D_cl = [rep["monotones"]["D_rho_vs_gamma"] for rep in reports_cl]
                I_cl = [rep["monotones"]["I_rho"] for rep in reports_cl]
                ax.plot(D_cl, I_cl, linestyle="--", linewidth=1.5, label="classical LT line")

            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_title("LT geometry: boundary + interior")
            ax.legend()
            path = save_plot(fig, "lt_geometry_combined.png")
            log_info("LT Geometry Combined", f"Saved:\n{path}")
            print("Finished LT geometry combined analysis.")

        elif eq_id == "lt_convertibility_graph":
            # Build an LT ensemble and compute GP vs LGP reachability graphs
            num_samples = int(vars_dict.get("num_samples", 25))
            N = max(8, min(30, num_samples))  # clamp for runtime
            N = 8  # TEMP OVERRIDE FOR TESTING
            states = []
            labels = []

            # 1) Classical LT points (qubits only)
            if system.dims == (2, 2):
                n_cl = min(8, max(2, N // 3))
                reports_cl = analyzer.scan_classical_LT_line_qubit(num_points=n_cl, solver=system.solver_default, tol=system.tol_default, verbose=False)
                for rep in reports_cl:
                    a = rep.get("a", None)
                    rho_cl = analyzer.factory.classical_LT_point_qubit(a=a)
                    states.append(rho_cl)
                    labels.append(f"cl a={a:.2f}")

            # 2) Extremal LT boundary samples
            n_ext = min(10, max(3, N // 3))
            extremals = analyzer.sample_extremal_lt_states(num_samples=n_ext, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
            for k, rep in enumerate(extremals):
                states.append(rep["rho"])
                labels.append(f"ext {k}")

            # 3) Interior LT points via projection
            while len(states) < N:
                rho = random_state(d)
                sigma_LT, _, _ = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
                if sigma_LT is None:
                    continue
                states.append(sigma_LT)
                labels.append(f"proj {len(states)-1}")

            N = len(states)
            log_info("Convertibility Graph", f"Testing pairwise reachability on N={N} LT states (this can take a moment).")

            G = np.zeros((N, N), dtype=int)
            L = np.zeros((N, N), dtype=int)

            # Pairwise convertibility
            for i in range(N):
                for j in range(N):
                    if i == j:
                        G[i, j] = 1
                        L[i, j] = 1
                        continue
                    g_ok, _ = system.check_global_gp_feasible(states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False)
                    l_ok, _ = system.check_local_gp_feasible(states[i], states[j], solver=system.solver_default, tol=system.tol_default, verbose=False)
                    G[i, j] = 1 if g_ok else 0
                    L[i, j] = 1 if l_ok else 0

            # Incomparability under LGP
            unordered = 0
            incomparable = 0
            for i in range(N):
                for j in range(i + 1, N):
                    unordered += 1
                    if (L[i, j] == 0) and (L[j, i] == 0):
                        incomparable += 1
            inc_rate = incomparable / unordered if unordered else 0.0

            # Heatmaps
            fig, ax = plt.subplots()
            ax.imshow(G, interpolation="nearest", aspect="auto")
            ax.set_title("Adjacency: Global GP (i → j)")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            pathG = save_plot(fig, "convertibility_global_heatmap.png")

            fig, ax = plt.subplots()
            ax.imshow(L, interpolation="nearest", aspect="auto")
            ax.set_title(f"Adjacency: Local GP (i → j)  | incomparability={inc_rate:.2%}")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            pathL = save_plot(fig, "convertibility_local_heatmap.png")

            # Directed graph in 3D (use local graph)
            coords_rng = np.random.default_rng(0)
            coords = np.array([embed_state_3d(system, r, rng=coords_rng) for r in states])

            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

            # draw edges if not too dense
            if N <= 25:
                for i in range(N):
                    for j in range(N):
                        if i != j and L[i, j] == 1:
                            ax.plot(
                                [coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                [coords[i, 2], coords[j, 2]],
                                linewidth=0.6,
                                alpha=0.35,
                            )

            ax.set_title("Local GP reachability graph (embedded in 3D)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            pathGraph = save_plot(fig, "convertibility_local_graph_3d.png")

            log_info(
                "Convertibility Graph Results",
                f"Incomparability rate (Local GP): {inc_rate:.2%}\n"
                f"Saved heatmaps:\n- {pathG}\n- {pathL}\nSaved graph:\n- {pathGraph}"
            )
            print("Finished LT convertibility graph analysis.")

        elif eq_id == "extract_global_channel":
            # Try to find one mapping where global GP is feasible, then dump the Choi matrix.
            # Prefer TFD->dephased if available.
            candidates = []
            try:
                tau = analyzer.factory.tfd_state()
                tau_p = dephase_global_in_energy_basis(tau)
                candidates.append(("TFD→dephased", tau, tau_p))
            except Exception:
                pass

            # Add some LT→LT projected candidates
            for _ in range(6):
                r1 = random_state(d)
                r2 = random_state(d)
                t1, _, _ = system.closest_lt_state(r1, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
                t2, _, _ = system.closest_lt_state(r2, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
                if t1 is not None and t2 is not None:
                    candidates.append(("projLT→projLT", t1, t2))

            picked = None
            details = None
            for name, tau, tau_p in candidates:
                ok, status, det = system.check_global_gp_feasible(
                    tau, tau_p,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    return_details=True
                )
                if ok and det.get("J") is not None:
                    picked = (name, tau, tau_p, status)
                    details = det
                    break

            if picked is None:
                log_warning(
                    "Extract Global Channel",
                    "Couldn't find a feasible global GP mapping in the quick candidate set. "
                    "Try increasing num_samples or relaxing eps_eq_global / eps_gibbs."
                )
                return

            name, tau, tau_p, status = picked
            J = details["J"]

            # Verify CPTP approximately: TP is enforced; CP => J ⪰ 0 (numerical)
            # Verify mapping and Gibbs preservation directly
            GAxGAp = np.kron(system.gammaA, system.gammaAp)
            Yg = system.choi_apply_numpy(J, GAxGAp, d_in=d, d_out=d)
            Ym = system.choi_apply_numpy(J, 0.5*(tau+dagger(tau)), d_in=d, d_out=d)

            gibbs_err = np.linalg.norm(Yg - GAxGAp, "fro")
            map_err = np.linalg.norm(Ym - 0.5*(tau_p+dagger(tau_p)), "fro")

            # Save Choi matrix
            ensure_png_dir()
            choi_path = os.path.join("png", "global_gp_choi.npy")
            np.save(choi_path, J)

            text = (
                f"Picked mapping: {name}\n"
                f"Status: {status}\n\n"
                f"Choi saved: {choi_path}\n"
                f"Gibbs error ||Φ(γ⊗γ)-γ⊗γ||_F = {gibbs_err:.3e}\n"
                f"Map   error ||Φ(τ)-τ'||_F       = {map_err:.3e}\n"
            )
            log_info("Extract Global GP Channel", text)
            print("Finished global channel extraction analysis.")

        elif eq_id == "sanity_checks":
            # Produce a small 'capstone-grade' table of errors + monotone change.
            # Use one mapping (prefer TFD->dephased).
            try:
                tau = analyzer.factory.tfd_state()
                tau_p = dephase_global_in_energy_basis(tau)
                label = "TFD→dephased"
            except Exception:
                # fallback: two LT projected states
                r1 = random_state(d)
                r2 = random_state(d)
                tau, _, _ = system.closest_lt_state(r1, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
                tau_p, _, _ = system.closest_lt_state(r2, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
                label = "projLT→projLT"

            if tau is None or tau_p is None:
                log_warning("Sanity checks", "Could not construct test states.")
                return

            # LT errors
            tauA = analyzer.system.lt_membership(tau)[3]
            tauAp = analyzer.system.lt_membership(tau)[4]
            lt_err_A = np.linalg.norm(tauA - system.gammaA, "fro")
            lt_err_Ap = np.linalg.norm(tauAp - system.gammaAp, "fro")

            # Global GP channel residuals
            g_ok, g_status, g_det = system.check_global_gp_feasible(
                tau, tau_p, solver=system.solver_default, tol=system.tol_default, return_details=True
            )
            J = g_det.get("J", None)
            if J is None:
                log_warning("Sanity checks", "Global GP channel solve failed (no J). Try relaxing tolerances.")
                return

            GAxGAp = np.kron(system.gammaA, system.gammaAp)
            Yg = system.choi_apply_numpy(J, GAxGAp, d_in=d, d_out=d)
            Ym = system.choi_apply_numpy(J, 0.5*(tau+dagger(tau)), d_in=d, d_out=d)

            gp_err = np.linalg.norm(Yg - GAxGAp, "fro")
            map_err = np.linalg.norm(Ym - 0.5*(tau_p+dagger(tau_p)), "fro")

            # Local GP residual (gap score)
            l_ok, l_status, l_det = system.check_local_gp_feasible(
                tau, tau_p, solver=system.solver_default, tol=system.tol_default, return_details=True
            )
            l_res = float(l_det.get("residual", np.inf))

            # Monotone change
            D_tau, I_tau, _, _ = system.monotones(tau)
            D_taup, I_taup, _, _ = system.monotones(tau_p)
            dD = D_tau - D_taup

            # Table-like print
            lines = [
                f"Mapping: {label}",
                "",
                f"LT error ||tau_A - gamma||_F      = {lt_err_A:.3e}",
                f"LT error ||tau_A' - gamma||_F     = {lt_err_Ap:.3e}",
                "",
                f"Global GP feasible? {g_ok} | {g_status}",
                f"GP constraint error ||Φ(γ⊗γ)-γ⊗γ||_F = {gp_err:.3e}",
                f"Map error ||Φ(tau)-tau'||_F           = {map_err:.3e}",
                "",
                f"Local GP feasible?  {l_ok} | {l_status}",
                f"Local best residual (step-2 objective) = {l_res:.3e}",
                "",
                f"Monotones: D(tau)={D_tau:.4f}, D(tau')={D_taup:.4f}, ΔD = {dD:.4f} (should be ≥ 0 under GP)",
                f"          I(tau)={I_tau:.4f}, I(tau')={I_taup:.4f}",
            ]
            text = "\n".join(lines)
            log_info("Sanity checks", text)
            print("Finished sanity checks analysis.")

        else:
            log_warning(
                "No backend attached",
                "No specific backend implemented for this experiment id yet.\n\n"
                f"ID: {eq_id}\nName: {eq_name}\n\n"
                f"Variables: {vars_dict}"
            )

    # Create and show the GUI window
    window = LTSDPWindow(run_callback=backend_run)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
