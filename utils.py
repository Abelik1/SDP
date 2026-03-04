# utils.py

import os
import numpy as np
import matplotlib.pyplot as plt

from ltgp.system import LTGPSystem
from sdp_system import dagger
from sdp_analysis import LTAnalyzer


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


def parse_variables_string(var_str):
    """Parse 'k1=v1, k2=v2' into dict with int/float/bool conversion."""
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


def default_hamiltonian(d, scale=1.0):
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
    dA = int(dA)
    dAp = int(dAp)

    if symmetric and dA == dAp:
        H_A = default_hamiltonian(dA, scale=1.0)
        H_Ap = H_A.copy()
    else:
        H_A = default_hamiltonian(dA, scale=1.0)
        scale_ap = 1.0 if dA != dAp else 1.3
        H_Ap = default_hamiltonian(dAp, scale=scale_ap)

    system = LTGPSystem(
        H_A,
        H_Ap,
        beta,
        solver=solver,
        tol=tol,
        eps_eq_global=eps_eq_global,
        eps_eq_local=eps_eq_local,
        eps_gibbs=eps_gibbs,
    )
    analyzer = LTAnalyzer(system)
    return system, analyzer


def random_state(d):
    X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = X @ dagger(X)
    rho = rho / np.trace(rho)
    return 0.5 * (rho + dagger(rho))


def embed_state_3d(system, rho, rng=None):
    dA, dAp = system.dims
    d = dA * dAp

    if (dA, dAp) == (2, 2):
        B = system.correlation_C(rho)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.kron(sx, sx)
        Y = np.kron(sy, sy)
        Z = np.kron(sz, sz)
        return np.array([
            float(np.real(np.trace(B @ X))),
            float(np.real(np.trace(B @ Y))),
            float(np.real(np.trace(B @ Z))),
        ])

    if rng is None:
        rng = np.random.default_rng(0)

    Os = []
    for _ in range(3):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = 0.5 * (A + dagger(A))
        Os.append(H / (np.linalg.norm(H, "fro") + 1e-12))

    return np.array([float(np.real(np.trace(rho @ O))) for O in Os])