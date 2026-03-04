import sys
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets

from utils import build_system_and_analyzer
from ltgp.ui import LTGPMainWindow
from ltgp.backend import backend_run


def main():
    np.random.seed(0)
    system, analyzer = build_system_and_analyzer(
        dA=2,
        dAp=2,
        beta=1.0,
        solver="SCS",
        tol=1e-7,
        symmetric=True,
        eps_eq_global=1e-6,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    )

    app = QtWidgets.QApplication(sys.argv)

    def run_callback(config):
        return backend_run(config, system, analyzer)

    window = LTGPMainWindow(run_callback=run_callback)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()