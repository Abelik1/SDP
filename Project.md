# Capstone Project Summary  
## Hierarchy of Locally Thermal Bipartite States under Global and Local Gibbs-Preserving Operations

---

# 0. System Setup

We consider a symmetric bipartite quantum system:

\[
\mathcal{H}_A \otimes \mathcal{H}_B
\]

with identical Hamiltonians:

\[
H_A = H_B
\]

Thermal (Gibbs) state:

\[
\gamma = \frac{e^{-\beta H}}{Z}
\]

Global equilibrium reference state:

\[
\gamma_{AB} = \gamma \otimes \gamma
\]

Allowed operations:

- **Global Gibbs-preserving (GP):**
  \[
  \Phi(\gamma_{AB}) = \gamma_{AB}
  \]
  with \(\Phi\) completely positive and trace-preserving (CPTP).

- **Local Gibbs-preserving (Local GP):**
  \[
  \Phi = \Phi_A \otimes \Phi_B,
  \quad
  \Phi_X(\gamma) = \gamma
  \]

Monotone used:

\[
D(\rho \Vert \gamma \otimes \gamma)
\]

On the locally thermal (LT) set:

\[
D(\rho \Vert \gamma \otimes \gamma) = I(A:B)
\]

---

# Objective 1  
## Identify and Characterise the Set of Locally Thermal States

### Definition

A bipartite state is **locally thermal (LT)** if:

\[
\mathrm{Tr}_A(\rho) = \gamma,
\quad
\mathrm{Tr}_B(\rho) = \gamma
\]

### Structural Form

Every LT state can be written as:

\[
\rho = \gamma \otimes \gamma + C
\]

with:

\[
\mathrm{Tr}_A C = 0,
\quad
\mathrm{Tr}_B C = 0,
\quad
\rho \succeq 0
\]

Thus LT is:

- An affine subspace (marginal constraints),
- Intersected with the positive semidefinite (PSD) cone,
- Convex.

### Qubit Case Representation

For qubits:

\[
C = \sum_{i,j \in \{x,y,z\}} t_{ij} \, \sigma_i \otimes \sigma_j
\]

So LT states are parameterised by a real \(3 \times 3\) correlation matrix \(T = [t_{ij}]\).

---

### Computational Work Completed

- Projection onto LT via trace-norm SDP.
- Convexity verification.
- Extremal boundary mapping via support-function optimisation.
- Interior sampling.
- Demonstrated classical LT ⊂ LT strictly.

---

### Structured LT Families (New Phase)

#### Ray Family

\[
\rho(p) = \gamma \otimes \gamma + p C_0
\]

Analytic positivity bound:

Define

\[
\tilde C =
(\gamma^{-1/2} \otimes \gamma^{-1/2})
C_0
(\gamma^{-1/2} \otimes \gamma^{-1/2})
\]

Then:

\[
\rho(p) \succeq 0
\iff
I + p \tilde C \succeq 0
\]

So:

\[
p \in
\left[
\max_{\lambda_i>0} \left(-\frac{1}{\lambda_i}\right),
\min_{\lambda_i<0} \left(-\frac{1}{\lambda_i}\right)
\right]
\]

---

# Objective 2  
## Construct the Hierarchy under Global Gibbs-Preserving Operations

Definition:

\[
\rho \succ \sigma
\quad \text{if} \quad
\exists \Phi \text{ (GP)} \text{ such that } \Phi(\rho) = \sigma
\]

### Monotonicity

\[
D(\rho \Vert \gamma \otimes \gamma)
\ge
D(\Phi(\rho) \Vert \gamma \otimes \gamma)
\]

On LT:

\[
I(A:B) \text{ is monotone}
\]

---

### Computational Results

- Pairwise GP feasibility via SDP.
- Convertibility graph constructed.
- Near-total preorder observed.
- Explicit GP channel constructed (TFD → dephased).
- Numerical monotone validation performed.

**Conclusion:**  
Global GP hierarchy is largely governed by the scalar monotone \(I(A:B)\).

---

# Objective 3  
## Hierarchy under Local Gibbs-Preserving Operations

Restrict:

\[
\Phi = \Phi_A \otimes \Phi_B
\]

### Structural Transformation (Qubit Case)

Each qubit CPTP map acts on Bloch vectors:

\[
r \mapsto M r + t
\]

Fixing \(\gamma\) implies:

\[
t = (I - M) r_\gamma
\]

On LT states:

\[
T \mapsto T' = M_A T M_B^T
\]

This is the key structural equation.

---

### Computational Results

- Pairwise Local GP feasibility via SDP.
- Convertibility graph constructed.
- ~96% incomparability observed.
- Scalar monotone insufficient.

**Central Result:**  
Locality induces a highly fragmented partial order.

---

# Structural Hypotheses Under Investigation

For feasible local GP edges:

1. Mutual information monotonicity:
   \[
   I(\rho) \ge I(\rho')
   \]

2. Trace-distance contractivity:
   \[
   \| \rho - \gamma \otimes \gamma \|_1
   \ge
   \| \rho' - \gamma \otimes \gamma \|_1
   \]

   equivalently:

   \[
   \|C'\|_1 \le \|C\|_1
   \]

3. Singular value contraction (conjecture):

   Let \(s_k(T)\) be singular values of \(T\):

   \[
   s_k(T') \le s_k(T)
   \quad \forall k
   \]

These are tested numerically using structured LT families.

---

# Current Project Status

Objective 1 (Set characterisation): ~85% complete  
Objective 2 (Global hierarchy): ~90% complete  
Objective 3 (Local hierarchy): ~80% complete  
Structural invariant testing: in progress  

---

# Final Thesis Claim

1. The locally thermal manifold is a convex affine subset parameterised by correlation operators \(C\).
2. Under global Gibbs-preserving operations, the hierarchy is largely governed by a scalar thermodynamic monotone.
3. Under local Gibbs-preserving operations, the hierarchy becomes highly fragmented.
4. This fragmentation cannot be explained by scalar monotones alone and is linked to structural constraints on the correlation tensor.

---

# Remaining Steps

1. Finalise structured-family experiments (ray and diagonal-T).
2. Validate monotone contraction hypotheses numerically.
3. Compare predictive power of:
   - Mutual information
   - Trace norm
   - Singular values
4. Write structural explanation section tying tensor transformation law to observed incomparability.
