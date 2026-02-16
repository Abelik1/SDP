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
