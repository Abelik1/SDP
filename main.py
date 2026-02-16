import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import json

from sdp_system import LTSDPSystem, dagger, mutual_information
from sdp_analysis import LTAnalyzer
from sdp_gui import LTSDPWindow
from utils import (
    parse_variables_string,
    build_system_and_analyzer,
    save_plot,
    log_info,
    log_warning,
    log_error,
    random_state,
    embed_state_3d,
)

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
    eps_eq_global=1e-6,
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

def plot_lt_family_scan(p_list, observables, title_prefix, filename_prefix):
    """Save a compact set of plots for an LT family scan (I, ||C||_1/2, ||C||_F, T singular values)."""
    import matplotlib.pyplot as plt

    p_list = np.array(p_list, dtype=float)

    # I(p)
    fig = plt.figure()
    plt.plot(p_list, observables["I"], marker="o", linestyle="-")
    plt.xlabel("p")
    plt.ylabel("I(A:B)")
    plt.title(f"{title_prefix} — mutual information")
    save_plot(fig, f"{filename_prefix}_I_vs_p.png")
    plt.close(fig)

    # ||C||_1/2 and ||C||_F
    fig = plt.figure()
    plt.plot(p_list, observables["C_trace_dist"], marker="o", linestyle="-", label="0.5||C||_1")
    plt.plot(p_list, observables["C_fro"], marker="s", linestyle="--", label="||C||_F")
    plt.xlabel("p")
    plt.ylabel("correlation size")
    plt.title(f"{title_prefix} — correlation norms")
    plt.legend()
    save_plot(fig, f"{filename_prefix}_C_norms_vs_p.png")
    plt.close(fig)

    # T singular values (if present)
    Ts = observables.get("T_svals", None)
    if Ts and Ts[0] is not None:
        s0 = np.array([s[0] for s in Ts], dtype=float)
        s1 = np.array([s[1] for s in Ts], dtype=float)
        s2 = np.array([s[2] for s in Ts], dtype=float)
        fig = plt.figure()
        plt.plot(p_list, s0, marker="o", linestyle="-", label="s1")
        plt.plot(p_list, s1, marker="s", linestyle="--", label="s2")
        plt.plot(p_list, s2, marker="^", linestyle=":", label="s3")
        plt.xlabel("p")
        plt.ylabel("svals(T)")
        plt.title(f"{title_prefix} — singular values of correlation tensor T")
        plt.legend()
        save_plot(fig, f"{filename_prefix}_T_svals_vs_p.png")
        plt.close(fig)
# =========================
# Structured LT family experiments (no GUI changes)
# =========================

def _invsqrt_psd(mat, tol=1e-12):
    """Hermitian PSD inverse square root via eigendecomposition."""
    H = 0.5 * (mat + mat.conj().T)
    w, U = np.linalg.eigh(H)
    w = np.real(w)
    w[w < tol] = tol
    return U @ np.diag(w ** (-0.5)) @ dagger(U)

def _whiten_C(system, C0, tol=1e-12):
    """C~ = (γ^{-1/2}⊗γ'^{-1/2}) C0 (γ^{-1/2}⊗γ'^{-1/2})."""
    GinvA = _invsqrt_psd(system.gammaA, tol=tol)
    GinvB = _invsqrt_psd(system.gammaAp, tol=tol)
    W = np.kron(GinvA, GinvB)
    C0h = 0.5 * (C0 + C0.conj().T)
    Ct = W @ C0h @ W
    return 0.5 * (Ct + Ct.conj().T)

def _lt_ray_p_bounds(system, C0, tol=1e-12):
    """
    Analytic PSD interval for ρ(p)=γ⊗γ + p C0 from eigenvalues of C~.
    Returns (p_min, p_max) such that ρ(p)⪰0 for p in [p_min, p_max].
    """
    Ct = _whiten_C(system, C0, tol=tol)
    lam = np.linalg.eigvalsh(0.5 * (Ct + Ct.conj().T))
    lam = np.real(lam)

    p_min = -np.inf
    p_max = +np.inf
    for x in lam:
        if x > tol:
            p_min = max(p_min, -1.0 / x)
        elif x < -tol:
            p_max = min(p_max, -1.0 / x)

    # In the (rare) unbounded case, clamp to keep scans sane.
    if not np.isfinite(p_min):
        p_min = -1e6
    if not np.isfinite(p_max):
        p_max = +1e6
    return float(p_min), float(p_max)

def _C0_from_pauli_pair(label):
    """
    C0 for dims=(2,2): (1/4) σ_i⊗σ_j where label in {XX,YY,ZZ,XY,XZ,YZ,...}.
    This is traceless and has zero marginals.
    """
    sx, sy, sz = paulis()
    P = {"X": sx, "Y": sy, "Z": sz}
    lab = str(label).strip().upper()
    if len(lab) != 2 or lab[0] not in P or lab[1] not in P:
        raise ValueError("label must be one of XX, YY, ZZ, XY, XZ, YZ")
    return 0.25 * np.kron(P[lab[0]], P[lab[1]])

def _C0_from_diagT(tx, ty, tz):
    """C0 = (1/4)(tx XX + ty YY + tz ZZ) for dims=(2,2)."""
    sx, sy, sz = paulis()
    XX = np.kron(sx, sx)
    YY = np.kron(sy, sy)
    ZZ = np.kron(sz, sz)
    return 0.25 * (float(tx) * XX + float(ty) * YY + float(tz) * ZZ)

def _lt_state_on_ray(system, C0, p):
    """ρ(p)=γ⊗γ + p C0 (Hermitian, trace-normalized)."""
    G = np.kron(system.gammaA, system.gammaAp)
    rho = G + float(p) * 0.5 * (C0 + C0.conj().T)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho = rho / tr
    return 0.5 * (rho + rho.conj().T)

def _qubit_corr_tensor_T(system, rho):
    """
    T_{ij}=Tr(C σ_i⊗σ_j), i,j∈{x,y,z}, where C=rho-γ⊗γ.
    Returns real 3x3 tensor.
    """
    if system.dims != (2, 2):
        raise ValueError("Correlation tensor T implemented for dims=(2,2) only.")
    sx, sy, sz = paulis()
    sig = [sx, sy, sz]
    G = np.kron(system.gammaA, system.gammaAp)
    C = 0.5 * ((rho - G) + (rho - G).conj().T)

    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            O = np.kron(sig[i], sig[j])
            T[i, j] = float(np.real(np.trace(C @ O)))
    return T

def _structured_family_hierarchy_run(system, analyzer, spec):
    """
    Custom experiment:
      - build LT family along a ray ρ(p)=γ⊗γ + p C0 (Pauli pair) OR diagT direction
      - compute observables (I, 0.5||C||1, ||C||F, svals(T))
      - build convertibility adjacency (global + local GP)
      - validate monotone inequalities on feasible edges (tol=1e-8 default)
      - compare whether componentwise svals contraction predicts local infeasibility
    Outputs saved in ./png/.
    """
    if system.dims != (2, 2):
        raise ValueError("This structured-family experiment is implemented for symmetric qubits dims=(2,2).")

    family = str(spec.get("family", "ray")).strip().lower()
    num_p = int(spec.get("num_p", spec.get("num_points", 21)))
    include_negative = bool(spec.get("include_negative", False))
    pair_mode = str(spec.get("pair_mode", "decreasing")).strip().lower()  # all | decreasing | adjacent
    p_shrink = float(spec.get("p_shrink", 0.98))
    mono_tol = float(spec.get("mono_tol", 1e-8))

    # Direction C0
    tag_parts = []
    if family in ("ray", "ray_pauli", "pauli"):
        label = str(spec.get("label", "XX"))
        C0 = _C0_from_pauli_pair(label)
        tag_parts = ["ray", label.upper()]
    elif family in ("diagt", "diag_t", "diagt_ray"):
        tx = float(spec.get("tx", 1.0))
        ty = float(spec.get("ty", 0.0))
        tz = float(spec.get("tz", 0.0))
        C0 = _C0_from_diagT(tx, ty, tz)
        tag_parts = ["diagT", f"{tx:g}", f"{ty:g}", f"{tz:g}"]
    else:
        raise ValueError("family must be 'ray' (Pauli pair) or 'diagT' (tx,ty,tz).")

    # Analytic PSD bounds
    p_min, p_max = _lt_ray_p_bounds(system, C0, tol=1e-12)
    if not include_negative:
        p_min = max(0.0, p_min)

    # interior scan
    p_lo = p_min * p_shrink
    p_hi = p_max * p_shrink
    p_list = np.linspace(p_lo, p_hi, num_p, dtype=float)

    states = [_lt_state_on_ray(system, C0, float(p)) for p in p_list]

    # Observables
    I = np.zeros(num_p, dtype=float)
    C1 = np.zeros(num_p, dtype=float)   # 0.5||C||_1
    CF = np.zeros(num_p, dtype=float)   # ||C||_F
    svals = np.zeros((num_p, 3), dtype=float)

    for k, rho in enumerate(states):
        Dk, Ik, _, _ = system.monotones(rho, tol=1e-12)
        cm = system.correlation_metrics(rho, tol=1e-12)
        I[k] = float(Ik)
        C1[k] = float(cm["C_trace_dist"])
        CF[k] = float(cm["C_fro"])
        T = _qubit_corr_tensor_T(system, rho)
        sv = np.linalg.svd(T, compute_uv=False)
        svals[k, :] = np.sort(np.real(sv))[::-1]

    # Pair selection
    pairs = []
    if pair_mode == "all":
        for i in range(num_p):
            for j in range(num_p):
                if i != j:
                    pairs.append((i, j))
    elif pair_mode == "decreasing":
        for i in range(num_p):
            for j in range(num_p):
                if p_list[i] > p_list[j]:
                    pairs.append((i, j))
    elif pair_mode == "adjacent":
        for k in range(1, num_p):
            # directed from higher p to lower p
            i, j = (k, k - 1) if p_list[k] > p_list[k - 1] else (k - 1, k)
            pairs.append((i, j))
    else:
        raise ValueError("pair_mode must be one of: all | decreasing | adjacent")

    # Convertibility adjacency
    A_global = np.zeros((num_p, num_p), dtype=int)
    A_local = np.zeros((num_p, num_p), dtype=int)

    for (i, j) in pairs:
        tau = states[i]
        tau_p = states[j]

        try:
            g_ok, g_status = system.check_global_gp_feasible(
                tau, tau_p,
                solver=system.solver_default,
                tol=system.tol_default,
                eps_map=system.eps_eq_global,
                eps_gibbs=system.eps_gibbs,
                verbose=False,
            )
            A_global[i, j] = 1 if bool(g_ok) else 0
        except Exception:
            A_global[i, j] = 0

        try:
            l_ok, l_status = system.check_local_gp_feasible(
                tau, tau_p,
                solver=system.solver_default,
                tol=system.tol_default,
                eps_map=system.eps_eq_local,
                eps_gibbs=system.eps_gibbs,
                verbose=False,
                return_details=False,
            )
            A_local[i, j] = 1 if bool(l_ok) else 0
        except Exception:
            A_local[i, j] = 0

    # Monotone validation on feasible edges (global OR local)
    violations = []
    for (i, j) in pairs:
        if A_local[i, j] != 1 and A_global[i, j] != 1:
            continue
        ok_I = (I[i] + mono_tol >= I[j])
        ok_C = (C1[i] + mono_tol >= C1[j])
        ok_S = bool(np.all(svals[i, :] + mono_tol >= svals[j, :]))  # componentwise contraction

        if not (ok_I and ok_C and ok_S):
            violations.append({
                "i": int(i), "j": int(j),
                "p_i": float(p_list[i]), "p_j": float(p_list[j]),
                "feasible_global": int(A_global[i, j]),
                "feasible_local": int(A_local[i, j]),
                "I_i": float(I[i]), "I_j": float(I[j]),
                "C1_i": float(C1[i]), "C1_j": float(C1[j]),
                "s_i": svals[i, :].copy(), "s_j": svals[j, :].copy(),
                "ok_I": bool(ok_I), "ok_C": bool(ok_C), "ok_S": bool(ok_S),
            })

    # Predictor quality: pred_feasible := componentwise svals contraction
    TP = FP = TN = FN = 0
    for (i, j) in pairs:
        pred = bool(np.all(svals[i, :] + mono_tol >= svals[j, :]))
        actual = bool(A_local[i, j] == 1)
        if pred and actual:
            TP += 1
        elif pred and (not actual):
            FP += 1
        elif (not pred) and (not actual):
            TN += 1
        else:
            FN += 1

    def _safe_div(a, b):
        return float(a) / float(b) if b else float("nan")

    accuracy = _safe_div(TP + TN, TP + TN + FP + FN)
    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    specificity = _safe_div(TN, TN + FP)

    # Plot singular values vs p
    fig, ax = plt.subplots()
    ax.plot(p_list, svals[:, 0], marker="o", linestyle="-", label="s1")
    ax.plot(p_list, svals[:, 1], marker="s", linestyle="--", label="s2")
    ax.plot(p_list, svals[:, 2], marker="^", linestyle=":", label="s3")
    ax.set_xlabel("p")
    ax.set_ylabel("singular values of T")
    ax.set_title("Correlation tensor singular values vs p")
    ax.legend()
    tag = "_".join(tag_parts).replace("-", "m").replace(".", "p")
    sv_path = save_plot(fig, f"{tag}_T_svals_vs_p.png")

    # Save adjacency matrices
    folder = ensure_png_dir()
    g_path = os.path.join(folder, f"{tag}_adj_global.npy")
    l_path = os.path.join(folder, f"{tag}_adj_local.npy")
    np.save(g_path, A_global)
    np.save(l_path, A_local)

    # Save violations
    v_path = os.path.join(folder, f"{tag}_monotone_violations.json")
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(violations, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    # Console summary
    lines = []
    lines.append("LT Structured-Family Hierarchy (custom)")
    lines.append("")
    lines.append(f"Family: {family}")
    if family in ("ray", "ray_pauli", "pauli"):
        lines.append(f"Direction: C0 = (1/4){tag_parts[1]}")
    else:
        lines.append(f"Direction: C0 = (1/4)(tx XX + ty YY + tz ZZ) with tx,ty,tz = {tag_parts[1:]}")
    lines.append(f"β = {system.beta}, dims = {system.dims}, symmetric = True assumed")
    lines.append("")
    lines.append("Analytic PSD interval from C~ eigenvalues:")
    lines.append(f"  p ∈ [{p_min:.6g}, {p_max:.6g}] ; scanned interior shrink={p_shrink} -> [{p_lo:.6g}, {p_hi:.6g}]")
    lines.append(f"num_p = {num_p}, include_negative = {include_negative}, pair_mode = {pair_mode}")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  - singular-value plot: {sv_path}")
    lines.append(f"  - global adjacency .npy: {g_path}")
    lines.append(f"  - local  adjacency .npy: {l_path}")
    lines.append(f"  - monotone violations : {v_path} (count={len(violations)})")
    lines.append("")
    lines.append("Local feasibility vs svals(T) componentwise contraction predictor:")
    lines.append(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    lines.append(f"  accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}")

    return "\n".join(lines), {"A_global": A_global, "A_local": A_local, "violations": violations}

# =========================
# Main GUI entry
# =======================
from experiments import backend_run

def main():
    # default starting system (the GUI can override all of this)
    np.random.seed(0)
    system, analyzer = build_system_and_analyzer(
        dA=2, dAp=2, beta=1.0, solver="SCS", tol=1e-7,
        symmetric=True,
        eps_eq_global=1e-6,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    )

    app = QtWidgets.QApplication(sys.argv)

    def run_callback(config):
        backend_run(config, system, analyzer)
    
    # Create and show the GUI window
    window = LTSDPWindow(run_callback=run_callback)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
