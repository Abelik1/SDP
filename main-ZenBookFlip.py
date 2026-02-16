import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  # <-- add this

from sdp_system import LTSDPSystem, dagger
from sdp_analysis import LTAnalyzer
from sdp_gui import LTSDPWindow
def ensure_png_dir():
    """
    Ensure that the ./png directory exists and return its path.
    """
    folder = "png"
    os.makedirs(folder, exist_ok=True)
    return folder


def save_plot(fig, filename):
    """
    Save a Matplotlib figure into ./png and close it.

    Returns the full path of the saved file.
    """
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


def parse_variables_string(var_str):
    """
    Parse a simple 'k1=v1, k2=v2' string into a dict.
    Tries to convert values to int or float when possible.
    Very forgiving: ignores malformed chunks.
    """
    vars_dict = {}
    if not var_str:
        return vars_dict

    chunks = var_str.split(',')
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            continue
        k, v = chunk.split('=', 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # try int / float / fallback to string
        try:
            if '.' in v or 'e' in v.lower():
                val = float(v)
            else:
                val = int(v)
        except Exception:
            val = v
        vars_dict[k] = val
    return vars_dict


def build_system_and_analyzer(beta=1.0, solver="SCS", tol=1e-7, symmetric=True):
    """
    Convenience to create LTSDPSystem and LTAnalyzer.

    If symmetric=True, set H_Ap = H_A (needed for TFD tests).
    """
    dA = dAp = 2
    H_A = np.diag([0.0, 1.0])
    if symmetric:
        H_Ap = H_A.copy()
    else:
        H_Ap = np.diag([0.0, 1.5])

    system = LTSDPSystem(H_A, H_Ap, beta, solver=solver, tol=tol)
    analyzer = LTAnalyzer(system)
    return system, analyzer


def main():
    # ---- basic global config ----
    SEED = 0
    SOLVER = "MOSEK"       # or "MOSEK" if you have it
    TOL = 1e-7
    BETA = 1.0

    np.random.seed(SEED)

    # Start in the symmetric case so TFD is well-defined by default.
    system, analyzer = build_system_and_analyzer(
        beta=BETA, solver=SOLVER, tol=TOL, symmetric=True
    )

    app = QtWidgets.QApplication(sys.argv)

    def backend_run(config):
        """
        Called whenever the user presses 'Run' in the GUI.

        config:
          {
            "selected_equation_id": ...,
            "selected_equation_name": ...,
            "module_type": ...,
            "variables": "beta=1.0, d=2",
            "custom_function": "..."
          }
        """
        nonlocal system, analyzer

        eq_id = config.get("selected_equation_id")
        eq_name = config.get("selected_equation_name")
        vars_str = config.get("variables", "")
        custom_func_text = config.get("custom_function", "")
        variables_str = config.get("variables_str", config.get("variables", ""))
        print("\n[backend_run] Selected equation:", eq_id, f"({eq_name})")
        print("[backend_run] Variables string:", vars_str)
        print("[backend_run] Custom text length:", len(custom_func_text))

        vars_dict = parse_variables_string(vars_str)
        print("[backend_run] Parsed variables:", vars_dict)

        # Optional overrides from GUI
        beta_override = vars_dict.get("beta", None)
        symmetric_flag = bool(vars_dict.get("symmetric", 1))  # default True

        # Rebuild system/analyzer if beta or symmetry is changed
        if (beta_override is not None) or vars_dict.get("reset_system", None) is not None:
            beta_new = float(beta_override) if beta_override is not None else system.beta
            print(f"[backend_run] Rebuilding system with beta={beta_new}, "
                  f"symmetric={symmetric_flag}")
            system, analyzer = build_system_and_analyzer(
                beta=beta_new,
                solver=system.solver_default,
                tol=system.tol_default,
                symmetric=symmetric_flag,
            )

        dA, dAp = system.dims

        def rand_state(d):
            X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
            rho = X @ dagger(X)
            return rho / np.trace(rho)

        # Helper: global dephasing in energy basis of H_A + H_Ap
        def dephase_global_in_energy_basis(rho):
            H_A = system.H_A
            H_Ap = system.H_Ap
            eA, UA = np.linalg.eigh((H_A + dagger(H_A)) / 2.0)
            eAp, UAp = np.linalg.eigh((H_Ap + dagger(H_Ap)) / 2.0)
            U_tot = np.kron(UA, UAp)
            rho_e = dagger(U_tot) @ rho @ U_tot
            rho_e_deph = np.diag(np.diag(rho_e))
            return U_tot @ rho_e_deph @ dagger(U_tot)

        # ====================================================
        # Dispatch based on selected equation (Phase 1–3)
        # ====================================================

        # ---------- Phase 2 + Phase 3A: TFD vs dephased + convertibility ----------
        if eq_id == "tfd_vs_dephased":
            try:
                # Phase 2: analyze TFD & dephased
                results = analyzer.analyze_tfd_vs_dephased(
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                rep_tfd = results["tfd"]
                rep_deph = results["tfd_dephased"]

                # Phase 3A: convertibility TFD -> dephased TFD
                tfd_state   = analyzer.factory.tfd_state()
                tfd_deph    = rep_deph["rho"] if "rho" in rep_deph else tfd_state  # or rebuild as in analyze_tfd_vs_dephased
                conv_report = analyzer.analyze_pair(
                    tfd_state,
                    tfd_deph,
                    label="TFD_to_TFD_deph",
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )

                gp  = conv_report["feasibility"]["Global_GP"]
                lgp = conv_report["feasibility"]["Local_GP"]

                text = (
                    "TFD vs dephased TFD\n\n"
                    "=== Monotones (Phase 2) ===\n"
                    f"TFD:\n"
                    f"  I(A:B)       = {rep_tfd['monotones']['I_rho']:.4f}\n"
                    f"  D(ρ||γ⊗γ)    = {rep_tfd['monotones']['D_rho_vs_gamma']:.4f}\n"
                    f"  Dist to LT   = {rep_tfd['distance_to_LT']['distance']:.4e}\n"
                    f"  Dist to classical LT = "
                    f"{rep_tfd['distance_to_classical_LT']['distance']:.4e}\n\n"
                    f"Dephased TFD:\n"
                    f"  I(A:B)       = {rep_deph['monotones']['I_rho']:.4f}\n"
                    f"  D(ρ||γ⊗γ)    = {rep_deph['monotones']['D_rho_vs_gamma']:.4f}\n"
                    f"  Dist to LT   = {rep_deph['distance_to_LT']['distance']:.4e}\n"
                    f"  Dist to classical LT = "
                    f"{rep_deph['distance_to_classical_LT']['distance']:.4e}\n\n"
                    "=== Convertibility TFD → dephased (Phase 3A) ===\n"
                    f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                    f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n"
                )

                log_info("TFD vs Dephased TFD (Phase 2 + 3A)", text)
                print("Finished Processing")

            except Exception as e:
                log_error(
                    "TFD Error",
                    "TFD analysis failed.\n\n"
                    "Most likely H_A and H_A' are not compatible (need symmetric=True).\n\n"
                    f"Details:\n{e}",
                )

        # ---------- Phase 1A + 1B + 3B (classical LT line + random + classical pair) ----------
        elif eq_id == "classical_LT_line":
            if system.dims != (2, 2):
                log_warning(
                    "Unsupported dimension",
                    "Classical LT line scan is currently only implemented for 2×2.",
                )
            else:
                num_points = int(vars_dict.get("num_points", 11))

                # Phase 1A: scan classical LT line
                reports = analyzer.scan_classical_LT_line_qubit(
                    num_points=num_points,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                lines = []
                for rep in reports:
                    a = rep["a"]
                    I_val = rep["monotones"]["I_rho"]
                    lines.append(f"a={a:.4f}, I(A:B)={I_val:.4f}")
                text = "=== Phase 1A: Classical LT line scan (2×2) ===\n\n"
                text += "\n".join(lines[:20])  # show first 20 points

                # Phase 1B: random state analysis (distance to LT, etc.)
                rho_rand = rand_state(dA * dAp)
                rep_rand = analyzer.analyze_single_state(
                    rho_rand,
                    label="random_state",
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                text += "\n\n=== Phase 1B: Random state analysis ===\n"
                text += (
                    f"Is LT?       {rep_rand['LT_membership']['is_LT']}\n"
                    f"D(ρ||γ⊗γ)    = {rep_rand['monotones']['D_rho_vs_gamma']:.4f}\n"
                    f"I(A:B)       = {rep_rand['monotones']['I_rho']:.4f}\n"
                    f"Dist to LT   = {rep_rand['distance_to_LT']['distance']:.4e}\n"
                    f"Dist to classical LT = "
                    f"{rep_rand['distance_to_classical_LT']['distance']:.4e}\n"
                )

                # Phase 3B (classical pair): pick two a values and test convertibility
                if len(reports) >= 2:
                    rep1 = reports[0]
                    rep2 = reports[-1]
                    a1 = rep1["a"]
                    a2 = rep2["a"]
                    rho1 = analyzer.factory.classical_LT_point_qubit(a1)
                    rho2 = analyzer.factory.classical_LT_point_qubit(a2)
                    pair_rep = analyzer.analyze_pair(
                        rho1,
                        rho2,
                        label="classical_pair",
                        solver=system.solver_default,
                        tol=system.tol_default,
                        verbose=False,
                    )
                    gp = pair_rep["feasibility"]["Global_GP"]
                    lgp = pair_rep["feasibility"]["Local_GP"]

                    text += "\n=== Phase 3B: Classical LT pair convertibility ===\n"
                    text += (
                        f"a1={a1:.4f}, a2={a2:.4f}\n"
                        f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                        f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n"
                    )

                log_info("Classical LT line + random + classical pair", text)
                ensure_png_dir()

                # Plot I(A:B) for TFD vs dephased
                fig_I, ax_I = plt.subplots()
                labels = ["TFD", "Dephased TFD"]
                I_vals = [
                    rep_tfd["monotones"]["I_rho"],
                    rep_deph["monotones"]["I_rho"],
                ]
                ax_I.bar(labels, I_vals)
                ax_I.set_ylabel("I(A:B)")
                ax_I.set_title("Phase 2: I(A:B) for TFD vs dephased TFD")
                path_I = save_plot(fig_I, "phase2_tfd_vs_dephased_I.png")

                # Plot D(ρ || γ⊗γ) for TFD vs dephased
                fig_D, ax_D = plt.subplots()
                D_vals = [
                    rep_tfd["monotones"]["D_rho_vs_gamma"],
                    rep_deph["monotones"]["D_rho_vs_gamma"],
                ]
                ax_D.bar(labels, D_vals)
                ax_D.set_ylabel("D(ρ || γ⊗γ)")
                ax_D.set_title("Phase 2: D(ρ || γ⊗γ) for TFD vs dephased TFD")
                path_D = save_plot(fig_D, "phase2_tfd_vs_dephased_D.png")
                
                
                 # ---------- Plots for Phase 1A/1B/3B ----------
                ensure_png_dir()

                # (a) Classical LT line: I(A:B) vs a
                a_vals = [rep["a"] for rep in reports]
                I_vals = [rep["monotones"]["I_rho"] for rep in reports]

                fig_line, ax_line = plt.subplots()
                ax_line.plot(a_vals, I_vals, marker="o")
                ax_line.set_xlabel("a")
                ax_line.set_ylabel("I(A:B)")
                ax_line.set_title("Phase 1A: Classical LT line (2×2)")
                path_line = save_plot(fig_line, "phase1_classical_LT_line_I.png")

                # (b) Random state in (D,I) plane
                D_rand = rep_rand["monotones"]["D_rho_vs_gamma"]
                I_rand = rep_rand["monotones"]["I_rho"]

                fig_rand, ax_rand = plt.subplots()
                ax_rand.scatter([D_rand], [I_rand])
                ax_rand.set_xlabel("D(ρ || γ⊗γ)")
                ax_rand.set_ylabel("I(A:B)")
                ax_rand.set_title("Phase 1B: Random state in (D,I) plane")
                path_rand = save_plot(fig_rand, "phase1_random_state_DI.png")

                # (c) Classical pair endpoints in (D,I) (if we had at least 2 points)
                path_pair = None
                if len(reports) >= 2:
                    rep1 = reports[0]
                    rep2 = reports[-1]
                    a1 = rep1["a"]
                    a2 = rep2["a"]
                    D1 = rep1["monotones"]["D_rho_vs_gamma"]
                    I1 = rep1["monotones"]["I_rho"]
                    D2 = rep2["monotones"]["D_rho_vs_gamma"]
                    I2 = rep2["monotones"]["I_rho"]

                    fig_pair, ax_pair = plt.subplots()
                    ax_pair.scatter([D1, D2], [I1, I2])
                    ax_pair.set_xlabel("D(ρ || γ⊗γ)")
                    ax_pair.set_ylabel("I(A:B)")
                    ax_pair.set_title("Phase 3B: Classical endpoints in (D,I)")
                    ax_pair.annotate(f"a1={a1:.3f}", (D1, I1))
                    ax_pair.annotate(f"a2={a2:.3f}", (D2, I2))
                    path_pair = save_plot(fig_pair, "phase3_classical_pair_DI.png")

                log_info(
                    "Classical LT plots",
                    "Saved Phase 1/3 plots:\n"
                    f"{path_line}\n"
                    f"{path_rand}\n"
                    + (f"{path_pair}\n" if path_pair is not None else ""),
                )

                log_info("Classical LT line + random + classical pair", text)
                print("Finished Processsing")
                log_info(
                    "TFD vs dephased plots",
                    f"Saved Phase 2 plots:\n{path_I}\n{path_D}",
                )

                

        # ---------- Phase 3B: random pair GP/LGP ----------
        elif eq_id == "random_pair_gp_lgp":
            tau = rand_state(dA * dAp)
            tau_p = rand_state(dA * dAp)
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
                "Random τ → τ' convertibility (Phase 3B)\n\n"
                f"Global GP: feasible={gp['feasible']} (status={gp['status']})\n"
                f"Local  GP: feasible={lgp['feasible']} (status={lgp['status']})\n\n"
                f"D_tau  = {report['monotones']['D_tau_vs_gamma']:.4f}\n"
                f"D_taup = {report['monotones']['D_taup_vs_gamma']:.4f}\n"
                f"I_tau  = {report['monotones']['I_tau']:.4f}\n"
                f"I_taup = {report['monotones']['I_taup']:.4f}\n"
            )

            log_info("Random Pair GP / LGP Test", text)
                        # ---------- Plot for Phase 3B: random pair in (D,I) ----------
            ensure_png_dir()

            D_tau = report["monotones"]["D_tau_vs_gamma"]
            D_taup = report["monotones"]["D_taup_vs_gamma"]
            I_tau = report["monotones"]["I_tau"]
            I_taup = report["monotones"]["I_taup"]

            fig, ax = plt.subplots()
            ax.scatter([D_tau, D_taup], [I_tau, I_taup])
            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_title("Phase 3B: Random pair in (D,I) plane")
            ax.annotate("τ", (D_tau, I_tau))
            ax.annotate("τ'", (D_taup, I_taup))
            path = save_plot(fig, "phase3_random_pair_DI.png")

            log_info(
                "Random Pair Plot",
                f"Saved Phase 3B random pair plot:\n{path}",
            )

            print("Finished Processing")


        # ---------- Phase 3C: mixture with γ⊗γ ----------
        elif eq_id == "mix_with_gamma":
            rho = rand_state(dA * dAp)
            num_points = int(vars_dict.get("num_points", 6))
            lam_list = np.linspace(0.0, 1.0, num_points)

            reports = analyzer.scan_mixture_with_gamma(
                rho,
                lam_list,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )
            lines = []
            for rep in reports:
                lam = rep["lambda"]
                D_val = rep["monotones"]["D_rho_vs_gamma"]
                I_val = rep["monotones"]["I_rho"]
                lines.append(f"λ={lam:.2f}: D={D_val:.4f}, I={I_val:.4f}")

            text = "Mixture with γ⊗γ (Phase 3C)\n\n" + "\n".join(lines)
            # ---------- Plot for Phase 3C: ρ(λ) mixture path ----------
            ensure_png_dir()

            lam_vals = [rep["lambda"] for rep in reports]
            D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in reports]
            I_vals = [rep["monotones"]["I_rho"] for rep in reports]

            fig, ax = plt.subplots()
            ax.plot(lam_vals, D_vals, marker="o", label="D(ρ(λ) || γ⊗γ)")
            ax.plot(lam_vals, I_vals, marker="x", label="I(A:B)")
            ax.set_xlabel("λ")
            ax.set_title("Phase 3C: Mixture with γ⊗γ")
            ax.legend()
            path = save_plot(fig, "phase3C_mixture_with_gamma.png")

            log_info(
                "Mixture with γ⊗γ plots",
                f"Saved Phase 3C plot:\n{path}",
            )

            log_info("Mixture with γ⊗γ", text)
            print("Finished Processing")
        # ---------- Phase 4: Extremal LT geometry ----------
        elif eq_id == "extremal_LT_boundary":
            # Optional parameters from the Variables box:
            # e.g. "num_samples=40, classical=0"
            num_samples = int(vars_dict.get("num_samples", 40))
            classical_flag = bool(int(vars_dict.get("classical", 0)))

            extremals = analyzer.sample_extremal_lt_states(
                num_samples=num_samples,
                classical=classical_flag,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )

            if not extremals:
                text = "No extremal LT states found (all SDPs failed)."
                log_warning("Extremal LT Geometry", text)
                print("Finished Processing")
                return

            D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in extremals]
            I_vals = [rep["monotones"]["I_rho"] for rep in extremals]

            # Optionally overlay classical LT line if 2×2
            D_classical = None
            I_classical = None
            if system.dims == (2, 2):
                reports_cl = analyzer.scan_classical_LT_line_qubit(
                    num_points=21,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                D_classical = [
                    rep["monotones"]["D_rho_vs_gamma"] for rep in reports_cl
                ]
                I_classical = [rep["monotones"]["I_rho"] for rep in reports_cl]

            # Plot extremal points in (D,I)
            ensure_png_dir()
            fig, ax = plt.subplots()
            ax.scatter(D_vals, I_vals, alpha=0.8, label="Extremal LT")

            if D_classical is not None:
                ax.plot(
                    D_classical,
                    I_classical,
                    marker="o",
                    linestyle="--",
                    label="Classical LT line",
                )

            ax.set_xlabel("D(ρ || γ⊗γ)")
            ax.set_ylabel("I(A:B)")
            ax.set_title("Phase 4: Extremal LT geometry")
            ax.legend()

            path = save_plot(fig, "phase4_extremal_LT_geometry.png")

            text = (
                f"Collected {len(extremals)} extremal LT states.\n"
                f"Plot saved to: {path}"
            )
            log_info("Extremal LT boundary", text)
            print("Finished Processing")
            return
        elif eq_id == "lt_region_geometry":
            # -------------------------------
            # Parameters (can be set in GUI)
            # -------------------------------
            num_samples = int(vars_dict.get("num_samples", 50))
            classical_flag = bool(int(vars_dict.get("classical", 0)))

            log_info(
                "LT Region Geometry",
                f"Sampling {num_samples} extremal LT states "
                f"({'classical' if classical_flag else 'quantum'})"
            )

            # -------------------------------
            # Sample extremal LT states
            # -------------------------------
            extremals = analyzer.sample_extremal_lt_states(
                num_samples=num_samples,
                classical=classical_flag,
                solver=system.solver_default,
                tol=system.tol_default,
                verbose=False,
            )

            if not extremals:
                log_warning(
                    "LT Region Geometry",
                    "No extremal LT states found (all SDPs failed)."
                )
                return

            # -------------------------------
            # Extract monotones for plotting
            # -------------------------------
            D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in extremals]
            I_vals = [rep["monotones"]["I_rho"] for rep in extremals]

            # -------------------------------
            # Optional: overlay classical LT line (2×2 only)
            # -------------------------------
            D_classical, I_classical = None, None
            if system.dims == (2, 2) and not classical_flag:
                reports_cl = analyzer.scan_classical_LT_line_qubit(
                    num_points=30,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                D_classical = [
                    rep["monotones"]["D_rho_vs_gamma"] for rep in reports_cl
                ]
                I_classical = [
                    rep["monotones"]["I_rho"] for rep in reports_cl
                ]

            # -------------------------------
            # Extract monotones for plotting
            # -------------------------------
            D_vals = [rep["monotones"]["D_rho_vs_gamma"] for rep in extremals]
            I_vals = [rep["monotones"]["I_rho"] for rep in extremals]
            C_vals = [
                rep["monotones"]["C_A"] + rep["monotones"]["C_Ap"]
                for rep in extremals
            ]

            # -------------------------------
            # Optional: classical LT curve (z = 0)
            # -------------------------------
            D_classical, I_classical = None, None
            if system.dims == (2, 2) and not classical_flag:
                reports_cl = analyzer.scan_classical_LT_line_qubit(
                    num_points=30,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )
                D_classical = [
                    rep["monotones"]["D_rho_vs_gamma"] for rep in reports_cl
                ]
                I_classical = [
                    rep["monotones"]["I_rho"] for rep in reports_cl
                ]
                C_classical = [0.0 for _ in reports_cl]

            # -------------------------------
            # 3D plot
            # -------------------------------
            ensure_png_dir()

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")

            sc = ax.scatter(
                D_vals,
                I_vals,
                C_vals,
                c=C_vals,
                depthshade=True,
                s=50,
                label="Extremal LT states"
            )

            if D_classical is not None:
                ax.plot(
                    D_classical,
                    I_classical,
                    C_classical,
                    linestyle="--",
                    marker="o",
                    label="Classical LT boundary (C = 0)"
                )

            ax.set_xlabel(r"$D(\rho \| \gamma \otimes \gamma)$")
            ax.set_ylabel(r"$I(A\!:\!B)$")
            ax.set_zlabel(r"$C_A + C_{A'}$")

            ax.set_title("Geometry of the Locally Thermal State Space")

            fig.colorbar(sc, ax=ax, pad=0.1, label="Total coherence")

            ax.legend()

            path = save_plot(fig, "lt_region_geometry_3d.png")
            
            log_info(
                "LT Region Geometry",
                f"Collected {len(extremals)} extremal LT states.\n"
                f"3D plot saved to:\n{path}"
            )
            fig.show()
            print("Finished Processing")
        elif eq_id == "lt_interior_geometry":
            # --------------------------------
            # Parameters
            # --------------------------------
            num_samples = int(vars_dict.get("num_samples", 200))
            seed = int(vars_dict.get("seed", 0))

            if seed >= 0:
                np.random.seed(seed)

            log_info(
                "LT Interior Geometry",
                f"Sampling {num_samples} random states and projecting to LT set"
            )

            # --------------------------------
            # Generate random states
            # --------------------------------
            dA, dAp = system.dims
            d = dA * dAp

            def random_state():
                X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
                rho = X @ dagger(X)
                return rho / np.trace(rho)

            projected_states = []

            for _ in range(num_samples):
                rho = random_state()

                # Project to full LT
                sigma_LT, _, status_LT = system.closest_lt_state(
                    rho,
                    classical=False,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )

                # Project to classical LT
                sigma_cl, dist_cl, status_cl = system.closest_lt_state(
                    rho,
                    classical=True,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    verbose=False,
                )

                if sigma_LT is None or sigma_cl is None:
                    continue

                # Compute monotones
                D_val, I_val, C_A, C_Ap = system.monotones(sigma_LT)

                projected_states.append({
                    "D": D_val,
                    "I": I_val,
                    "dist_classical": dist_cl,
                })

            if not projected_states:
                log_warning(
                    "LT Interior Geometry",
                    "No LT projections succeeded."
                )
                return

            # --------------------------------
            # Extract data
            # --------------------------------
            D_vals = [p["D"] for p in projected_states]
            I_vals = [p["I"] for p in projected_states]
            Z_vals = [p["dist_classical"] for p in projected_states]

            # --------------------------------
            # Plot (interactive 3D)
            # --------------------------------
            ensure_png_dir()

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")

            sc = ax.scatter(
                D_vals,
                I_vals,
                Z_vals,
                c=Z_vals,
                cmap="viridis",
                s=45,
                depthshade=True,
                label="Projected LT states"
            )

            ax.set_xlabel(r"$D(\sigma \| \gamma \otimes \gamma)$")
            ax.set_ylabel(r"$I(A\!:\!B)$")
            ax.set_zlabel(r"$\frac{1}{2}\|\sigma - \sigma_{\mathrm{classical}}\|_1$")

            ax.set_title("Interior Geometry of the Locally Thermal State Space")

            fig.colorbar(sc, ax=ax, pad=0.1, label="Distance to classical LT")

            path = save_plot(fig, "lt_interior_geometry_3d.png")

            log_info(
                "LT Interior Geometry",
                f"Projected {len(projected_states)} random states.\n"
                f"3D plot saved to:\n{path}"
            )

            fig.show()
            print("Finished Processing")

        else:
            # Fallback: just show the config
            log_warning(
                "No backend attached",
                "No specific backend implemented for this equation id yet.\n\n"
                f"ID: {eq_id}\nName: {eq_name}\n\n"
                f"Variables: {vars_dict}\n"
                "(Check main.py to add behaviour for this case.)",
            )

    # ---- Create and show the GUI window ----
    window = LTSDPWindow(run_callback=backend_run)
    window.show()

    # ---- Run the Qt event loop ----
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
