# Capstone Experiments Overview  
Hierarchy of Locally Thermal States under Global and Local Gibbs-Preserving Operations  

Symmetric bipartite system:  A ⊗ A′  
Identical Hamiltonians H_A = H_A′  
Inverse temperature β  

Thermal state:
γ_X = e^{-βH_X} / Z  

Global reference Gibbs state:
γ_{AA′} = γ_A ⊗ γ_A′  

------------------------------------------------------------
CORE DEFINITIONS
------------------------------------------------------------

Locally Thermal (LT) set:

Tr_{A′}(ρ) = γ_A  
Tr_A(ρ) = γ_A′  

Equivalent structural decomposition:

ρ = γ ⊗ γ + C  

with:

Tr_A(C) = 0  
Tr_B(C) = 0  
ρ ⪰ 0  

The LT set is therefore:
- An affine subspace (zero-marginal constraint)
- Intersected with the PSD cone
- Convex
- High-dimensional in state space

Thermodynamic monotone:

D(ρ || γ⊗γ)

On the LT set:

D(ρ || γ⊗γ) = I(A:B)

Thus LT states collapse to a 1D manifold in (D, I) coordinates,
but remain high-dimensional operationally.

------------------------------------------------------------
EXPERIMENT 1  
LT Membership Verification
------------------------------------------------------------

Given arbitrary ρ:

Check:
- Tr_B(ρ) == γ_A
- Tr_A(ρ) == γ_A′

Uses exact linear constraints (dimension-independent).

Conclusion:
Implementation characterises the full LT set,
not a restricted ansatz.

------------------------------------------------------------
EXPERIMENT 2  
TFD → Dephased TFD (Global GP vs Local GP)
------------------------------------------------------------

Initial state: Thermofield Double (TFD)

|TFD⟩ = Σ_i √g_i |E_i⟩_A ⊗ |E_i⟩_{A′}

Properties:
- Pure
- Locally thermal
- Maximally correlated in energy basis

Dephased state:

τ_deph = Σ_i g_i |E_i E_i⟩⟨E_i E_i|

Test reachability under:
- Global GP
- Local GP

Results:
- Global GP: feasible
- Local GP: infeasible

Conclusion:
Local Gibbs-preserving operations induce
a strict partial order.

------------------------------------------------------------
EXPERIMENT 3  
Projection onto LT (Trace-Distance SDP)
------------------------------------------------------------

Given arbitrary ρ:

Compute:

min_{σ ∈ LT} ½ ||ρ − σ||₁

Also compute projection onto classical LT.

Observations:
- Random states typically not LT.
- Distance to classical LT > distance to LT.

Conclusion:
classical LT ⊂ LT strictly.
Quantum correlations expand LT set.

------------------------------------------------------------
EXPERIMENT 4  
C-Decomposition Diagnostics
------------------------------------------------------------

For LT states:

C = ρ − γ⊗γ

Compute:

- Frobenius norm ||C||_F
- Trace distance ½||C||₁
- Zero-marginal residuals
- Operator-Schmidt singular values of C

Purpose:
Study full correlation structure,
not just scalar monotones.

Observation:
Most LT states have small ||C||,
strongly correlated states lie near extremal boundary.

------------------------------------------------------------
EXPERIMENT 5  
LT Extremal Boundary via Support-Function SDP
------------------------------------------------------------

Solve:

max_{ρ ∈ LT} Tr(Kρ)

for random Hermitian K.

Maps extremal LT states.

Plot in:
- (D, I)
- correlation metrics space

Observation:
All LT states satisfy I = D,
but boundary geometry is nontrivial in full operator space.

------------------------------------------------------------
EXPERIMENT 6  
LT Interior Sampling
------------------------------------------------------------

Project random states onto LT.

Analyse:

- Distribution of D
- Distribution of ||C||
- Operator-Schmidt spectra

Observation:
Interior LT states cluster near low D.
Strong quantum LT states are rare.

Conclusion:
Thermodynamically 1D,
operationally high-dimensional.

------------------------------------------------------------
EXPERIMENT 7  
Ray Family: ρ(p) = γ⊗γ + pC₀
------------------------------------------------------------

Structured LT family.

Positivity bound computed analytically via whitening:

C̃ = (γ^{-1/2}⊗γ^{-1/2}) C₀ (γ^{-1/2}⊗γ^{-1/2})

ρ(p) ⪰ 0  ⇔  I + p C̃ ⪰ 0

Compute valid interval [p_min, p_max].

Used to test:

- Global GP convertibility
- Local GP convertibility
- Monotone contraction hypotheses

------------------------------------------------------------
EXPERIMENT 8  
Diagonal Correlation Tensor Family (Qubit Case)
------------------------------------------------------------

C = (1/4)(t_x XX + t_y YY + t_z ZZ)

Explore structured tensor directions.

Compute:

- Correlation tensor T
- Singular values s_k(T)
- Mapping T → T′ under Local GP

Observation:
Local GP acts as:

T′ = M_A T M_B^T

Local hierarchy constrained by tensor structure.

------------------------------------------------------------
EXPERIMENT 9  
Operator-Schmidt Spectrum of C
------------------------------------------------------------

Reshape C across A|A′:

C → matrix of shape (d_A², d_A′²)

Compute singular values.

Used as structural signature.

Hypothesis tested:
Local GP may contract singular values
beyond scalar monotones.

------------------------------------------------------------
EXPERIMENT 10  
LT Convertibility Graph (Global GP vs Local GP)
------------------------------------------------------------

Fix finite LT set.
Test pairwise reachability via SDP.

Global GP:
- Nearly total preorder
- Governed by D = I

Local GP:
- Highly fragmented
- Large incomparability rate

Conclusion:
Locality introduces non-scalar constraints.

------------------------------------------------------------
EXPERIMENT 11  
Explicit Global GP Channel Extraction
------------------------------------------------------------

Extract Choi matrix J_Φ for TFD → dephased mapping.

Verify:
- J_Φ ⪰ 0
- Trace-preserving
- Φ(γ⊗γ) = γ⊗γ
- Φ(τ_TFD) = τ_deph

Residual errors ~ 10⁻¹⁰.

Conclusion:
Existence upgraded to explicit channel construction.

------------------------------------------------------------
EXPERIMENT 12  
Numerical Robustness Checks
------------------------------------------------------------

Verify:
- Gibbs preservation residual
- Mapping residual
- Monotone non-increase
- Solver stability

Conclusion:
Observed structure is physical,
not numerical artefact.

------------------------------------------------------------
OVERALL STRUCTURE OF RESULTS
------------------------------------------------------------

1. LT set is fully characterised via SDP:
   exact marginal constraints + PSD.

2. Thermodynamically:
   LT collapses to I = D.

3. Globally:
   GP hierarchy nearly total,
   governed by scalar monotone.

4. Locally:
   Highly fragmented partial order.

5. Correlation tensor structure (C-decomposition)
   reveals constraints invisible to scalar monotones.

Core Thesis:

Local Gibbs-preserving operations impose structural
constraints on the full correlation operator C,
producing genuine operational incomparability
within the locally thermal manifold.
