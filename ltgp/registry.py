from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

GROUPS_ORDER = [
    'A) LT Geometry',
    'B) State Families',
    'C) Convertibility',
    'D) Monotones & Invariants',
    'E) Utilities & Diagnostics',
]


@dataclass(frozen=True)
class ExperimentSpec:
    eq_id: str
    title: str
    description: str
    group: str
    tags: List[str] = field(default_factory=list)


def get_catalog() -> Dict[str, List[ExperimentSpec]]:
    items = [
        ExperimentSpec('closest_lt_distance', 'Distance to LT (trace norm)', 'Trace-distance projection onto LT or energy-diagonal LT.', 'A) LT Geometry', ['distance', 'projection']),
        ExperimentSpec('lt_region_geometry', 'Extremal LT boundary', 'Support-function extremals sampling the LT boundary.', 'A) LT Geometry', ['boundary', 'support', 'sdp']),
        ExperimentSpec('lt_interior_geometry', 'Interior LT cloud', 'Random states projected to LT for interior geometry plots.', 'A) LT Geometry', ['interior', 'projection']),
        ExperimentSpec('lt_geometry_combined', 'Boundary + interior', 'Overlay boundary and interior LT point clouds.', 'A) LT Geometry', ['figure', 'geometry']),
        ExperimentSpec('tfd_vs_dephased', 'TFD vs dephased TFD', 'Compare coherent LT and commuting LT exemplars.', 'B) State Families', ['tfd', 'dephasing']),
        ExperimentSpec('mix_with_gamma', 'Thermalisation path', 'Track monotones along (1-λ)ρ + λγ⊗γ.', 'B) State Families', ['mixing', 'thermalisation']),
        ExperimentSpec('lt_family_ray_validation', 'Qubit LT Pauli ray', 'Analytic positivity bounds, monotones, and convertibility along ρ(p)=γ⊗γ+pC0.', 'B) State Families', ['ray', 'qubit', 'pauli']),
        ExperimentSpec('lt_family_diagT_validation', 'Qubit diagonal-T ray', 'Diagonal-T slice with exact boundary interval and convertibility audits.', 'B) State Families', ['diagT', 'qubit']),
        ExperimentSpec('d3_commuting_sampling', 'Qutrit commuting LT subclass', 'Transport-polytope / Sinkhorn sampling of qutrit commuting LT states.', 'B) State Families', ['qutrit', 'commuting', 'polytope']),
        ExperimentSpec('random_pair_gp_lgp', 'Random pair GP vs local GP', 'Random source/target pair with global GP, verified local GP, and PPT outer test.', 'C) Convertibility', ['random', 'convertibility']),
        ExperimentSpec('lt_convertibility_graph', 'Convertibility graph', 'Adjacency comparison of global GP, verified local GP, and PPT-local outer relaxation.', 'C) Convertibility', ['graph', 'adjacency', 'verified']),
        ExperimentSpec('local_gp_ppt_relax', 'PPT outer relaxation', 'Necessary condition for local GP based on PPT Choi relaxation.', 'C) Convertibility', ['ppt', 'relaxation']),
        ExperimentSpec('extract_global_channel', 'Extract global GP channel', 'Return an explicit global Gibbs-preserving Choi matrix and diagnostics.', 'C) Convertibility', ['channel', 'choi']),
        ExperimentSpec('sanity_checks', 'Sanity checks table', 'Compact report of LT errors, GP errors, and monotone ladders.', 'D) Monotones & Invariants', ['table', 'diagnostics']),
        ExperimentSpec('separable_vs_entangled_lt', 'PPT/separability in LT', 'Classify LT samples by PPT; exact for 2x2 and 2x3, diagnostic otherwise.', 'D) Monotones & Invariants', ['ppt', 'separable']),
        ExperimentSpec('local_gp_closure_test', 'LT closure under local GP', 'Apply random verified local GP channels and check LT closure numerically.', 'E) Utilities & Diagnostics', ['closure', 'local']),
        ExperimentSpec('verified_local_edge_audit', 'Verified local edge audit', 'Audit heuristic vs verified local edges on a small ensemble.', 'E) Utilities & Diagnostics', ['audit', 'verified']),
        ExperimentSpec('custom', 'Custom JSON', 'Pass a backend-defined custom payload.', 'E) Utilities & Diagnostics', ['custom']),
    ]
    grouped = {g: [] for g in GROUPS_ORDER}
    for it in items:
        grouped.setdefault(it.group, []).append(it)
    return grouped


def find_by_id(eq_id: str) -> Optional[ExperimentSpec]:
    for _, items in get_catalog().items():
        for it in items:
            if it.eq_id == eq_id:
                return it
    return None
