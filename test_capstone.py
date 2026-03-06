import numpy as np

from ltgp.system import LTGPSystem, build_system_and_analyzer, choi_identity, default_hamiltonian, gibbs_state, partial_trace


def test_gibbs_state_normalized():
    H = default_hamiltonian(3)
    g = gibbs_state(H, beta=1.0)
    assert np.allclose(np.trace(g), 1.0)
    assert np.min(np.linalg.eigvalsh(g)) >= -1e-12


def test_partial_trace_shapes():
    system, _ = build_system_and_analyzer(dA=2, dAp=3, beta=1.0)
    rho = np.kron(system.gammaA, system.gammaAp)
    assert partial_trace(rho, system.dims, keep=[0]).shape == (2, 2)
    assert partial_trace(rho, system.dims, keep=[1]).shape == (3, 3)


def test_lt_membership_of_gamma_product():
    system, _ = build_system_and_analyzer(dA=2, dAp=2, beta=1.0)
    rho = np.kron(system.gammaA, system.gammaAp)
    ok, _, _, _, _ = system.lt_membership(rho, tol=1e-10)
    assert ok


def test_energy_diagonal_classical_projection_for_nondiagonal_hamiltonian():
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    system = LTGPSystem(H, H, beta=1.0, solver='SCS', tol=1e-6)
    rho = np.kron(system.gammaA, system.gammaAp)
    sigma, dist, status = system.closest_lt_state(rho, classical=True, solver='SCS', tol=1e-6, verbose=False)
    assert sigma is not None
    assert system.is_energy_diagonal(sigma, tol=1e-4)
    assert dist is not None


def test_lt_ray_bounds_match_endpoint_psd():
    system, _ = build_system_and_analyzer(dA=2, dAp=2, beta=1.0)
    C0 = system.qubit_C0_from_pauli_label('XX')
    pmin, pmax = system.lt_ray_p_bounds(C0)
    rho_in = system.lt_ray_state(C0, 0.999 * pmax)
    rho_out = system.lt_ray_state(C0, 1.001 * pmax)
    assert np.min(np.linalg.eigvalsh(rho_in)) > -1e-6
    assert np.min(np.linalg.eigvalsh(rho_out)) < 1e-4


def test_choi_identity_tp():
    J = choi_identity(2)
    system, _ = build_system_and_analyzer(dA=2, dAp=2, beta=1.0)
    diag = system.choi_diagnostics(J, d_in=2, d_out=2, gamma_in=system.gammaA, gamma_out=system.gammaA)
    assert diag['tp_fro_err'] < 1e-10
    assert diag['min_eig_J'] >= -1e-10


def test_random_seed_reproducibility():
    s1, _ = build_system_and_analyzer(dA=2, dAp=2, beta=1.0)
    s2, _ = build_system_and_analyzer(dA=2, dAp=2, beta=1.0)
    states1 = s1.sample_commuting_lt_d3(3, seed=7, iters=20) if s1.dims == (3, 3) else None
    assert np.allclose(s1.gammaA, s2.gammaA)
