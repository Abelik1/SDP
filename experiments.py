"""Top-level experiment entry point kept for compatibility.

All experiment implementations now live in ltgp.experiments_ext.
"""

from ltgp.experiments_ext import dispatch_experiment as backend_dispatch

__all__ = ['backend_dispatch']
