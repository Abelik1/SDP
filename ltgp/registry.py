from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


GROUPS_ORDER = [
    "A) LT Geometry",
    "B) State Families",
    "C) Convertibility",
    "D) Monotones & Invariants",
    "E) Utilities & Diagnostics",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """Metadata used by the GUI.
    - eq_id must match the backend dispatch id
    - tags are used for search
    """
    eq_id: str
    title: str
    description: str
    group: str
    tags: List[str] = field(default_factory=list)


def get_catalog() -> Dict[str, List[ExperimentSpec]]:
    items: List[ExperimentSpec] = [
        # -------- A) LT Geometry --------
        ExperimentSpec(
            "lt_region_geometry",
            "Extremal boundary (support function)",
            "Sample extremal locally-thermal states via support-function SDPs and plot boundary projections.",
            "A) LT Geometry",
            tags=["boundary", "support", "sdp"],
        ),
        ExperimentSpec(
            "lt_interior_geometry",
            "Interior (random → LT projection)",
            "Project random states onto LT via trace-norm SDP and visualise interior geometry.",
            "A) LT Geometry",
            tags=["interior", "projection", "trace-norm"],
        ),
        ExperimentSpec(
            "lt_geometry_combined",
            "Boundary + interior (final figure)",
            "Overlay interior LT points with boundary extremals; optional classical LT line for qubits.",
            "A) LT Geometry",
            tags=["figure", "boundary", "interior"],
        ),
        ExperimentSpec(
            "closest_lt_distance",
            "Distance to LT (trace norm)",
            "Compute min_{σ∈LT} 1/2 ||ρ-σ||_1 using the standard SDP (P,N ⪰ 0).",
            "A) LT Geometry",
            tags=["distance", "projection"],
        ),

        # -------- B) State Families --------
        ExperimentSpec(
            "tfd_vs_dephased",
            "TFD → dephased TFD",
            "Compare convertibility and monotones for a TFD-like pure LT state vs its dephased version.",
            "B) State Families",
            tags=["tfd", "dephasing"],
        ),
        ExperimentSpec(
            "mix_with_gamma",
            "Thermalisation path (1−λ)ρ + λγ⊗γ",
            "Mix a state with global Gibbs and track monotones / distances.",
            "B) State Families",
            tags=["mixing", "thermalisation"],
        ),
        ExperimentSpec(
            "lt_family_ray_validation",
            "LT family: Pauli ray ρ(p)=γ⊗γ+pC0 (qubits)",
            "Scan an LT ray family and test local-GP feasibility + monotone inequalities (I, ||C||, svals(T)).",
            "B) State Families",
            tags=["ray", "pauli", "qubit"],
        ),
        ExperimentSpec(
            "lt_family_diagT_validation",
            "LT family: diagonal-T ray (qubits)",
            "Scan a diagonal correlation-tensor ray ρ(p)=γ⊗γ + p*(tx XX+ty YY+tz ZZ)/4.",
            "B) State Families",
            tags=["diagT", "ray", "qubit"],
        ),
        ExperimentSpec(
            "lt_C_diagT_plane_characterise",
            "+C feasible region: 2D diag-T plane (qubits)",
            "Characterise feasible +C region in a diag-T plane; sample boundary and interior.",
            "B) State Families",
            tags=["diagT", "plane", "boundary"],
        ),
        ExperimentSpec(
            "lt_C_diagT_3d_characterise",
            "+C feasible region: 3D diag-T space (qubits)",
            "Approximate feasible +C region in full diag-T space and test convertibility on interior points.",
            "B) State Families",
            tags=["diagT", "3d"],
        ),
        ExperimentSpec(
            "d3_commuting_sampling",
            "d=3 commuting LT subclass (transport polytope)",
            "Sample energy-diagonal LT qutrit states via Sinkhorn scaling (row/col sums = γ) and compute monotones.",
            "B) State Families",
            tags=["qutrit", "commuting", "polytope", "sinkhorn"],
        ),

        # -------- C) Convertibility --------
        ExperimentSpec(
            "random_pair_gp_lgp",
            "Random τ → τ' (GP vs LGP)",
            "Sample a random pair and test global GP vs local GP (heuristic).",
            "C) Convertibility",
            tags=["random", "convertibility"],
        ),
        ExperimentSpec(
            "lt_convertibility_graph",
            "Convertibility graph (global vs local)",
            "Generate an LT ensemble and test pairwise convertibility; output adjacency and a directed graph plot.",
            "C) Convertibility",
            tags=["graph", "adjacency"],
        ),
        ExperimentSpec(
            "local_gp_ppt_relax",
            "Local GP outer relaxation: PPT-Choi test",
            "Convex outer relaxation: allow global channel with Choi PPT across (AoutAin)|(A'outA'in), plus GP + mapping constraints.",
            "C) Convertibility",
            tags=["ppt", "relaxation", "choi"],
        ),
        ExperimentSpec(
            "extract_global_channel",
            "Extract a global GP channel (Choi)",
            "Solve a global GP Choi SDP to find a concrete channel; verify CPTP + Gibbs-fixing numerically.",
            "C) Convertibility",
            tags=["choi", "channel"],
        ),
        ExperimentSpec(
            "extract_local_channels",
            "Extract local channels (JA,JAp)",
            "Run the two-step local GP solver and save JA, JAp (plus intermediate ω).",
            "C) Convertibility",
            tags=["local", "choi", "channels"],
        ),

        # -------- D) Monotones & Invariants --------
        ExperimentSpec(
            "sanity_checks",
            "Sanity checks table",
            "Generate a compact table of LT errors, GP errors, mapping errors, and monotone changes for example mappings.",
            "D) Monotones & Invariants",
            tags=["monotones", "table"],
        ),

        # -------- E) Utilities & Diagnostics --------
        ExperimentSpec(
            "local_gp_closure_test",
            "LT closure under random local GP channels",
            "Generate LT states, apply random local GP channels, and verify outputs remain LT (numerically).",
            "E) Utilities & Diagnostics",
            tags=["closure", "random", "local"],
        ),
        ExperimentSpec(
            "custom",
            "Custom (backend-defined)",
            "Pass JSON through to the backend.",
            "E) Utilities & Diagnostics",
            tags=["custom"],
        ),
    ]

    grouped: Dict[str, List[ExperimentSpec]] = {g: [] for g in GROUPS_ORDER}
    for it in items:
        grouped.setdefault(it.group, []).append(it)
    return grouped


def find_by_id(eq_id: str) -> Optional[ExperimentSpec]:
    for _, items in get_catalog().items():
        for it in items:
            if it.eq_id == eq_id:
                return it
    return None