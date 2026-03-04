"""LTGP: research toolkit for locally-thermal (LT) bipartite states and Gibbs-preserving operations.

Key entry points:
- ltgp.system.LTGPSystem: core math + SDPs (extends your LTSDPSystem)
- ltgp.registry: experiment grouping metadata for the GUI
- ltgp.ui.LTGPMainWindow: grouped/searchable GUI
- ltgp.backend.backend_run: run wrapper (new experiments + legacy fallback)
"""

from .system import LTGPSystem  # noqa: F401