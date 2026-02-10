import json
from dataclasses import dataclass
from typing import Any, Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QMessageBox,
)


@dataclass
class EquationItem:
    eq_id: str
    title: str
    description: str


def _kv(k: str, v: Any) -> str:
    if isinstance(v, bool):
        return f"{k}={'true' if v else 'false'}"
    return f"{k}={v}"


class LTSDPWindow(QWidget):
    """
    Minimal Qt GUI for selecting experiments + setting basic knobs.
    A run_callback(cfg_dict) is called when "Run" is pressed.
    """

    def __init__(self, run_callback=None, parent=None):
        super().__init__(parent)
        self.run_callback = run_callback

        self.setWindowTitle("LT / GP SDP Playground")
        self.setMinimumWidth(900)
        self._build_ui()
        self._refresh_details()

    # -------------------------
    # UI build
    # -------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Top row: experiment selector + run
        top = QHBoxLayout()
        root.addLayout(top)

        self.eq_combo = QComboBox()
        self.eq_items = self._default_equations()
        for item in self.eq_items:
            self.eq_combo.addItem(item.title, item.eq_id)
        self.eq_combo.currentIndexChanged.connect(self._refresh_details)
        top.addWidget(QLabel("Experiment:"))
        top.addWidget(self.eq_combo, 1)

        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._on_run)
        top.addWidget(run_btn)

        # Split: left controls / right details
        split = QHBoxLayout()
        root.addLayout(split, 1)

        left = QVBoxLayout()
        split.addLayout(left, 0)

        right = QVBoxLayout()
        split.addLayout(right, 1)

        # Controls group
        controls_box = QGroupBox("Controls")
        left.addWidget(controls_box)
        form = QFormLayout(controls_box)

        self.module_combo = QComboBox()
        self.module_combo.addItems(["Preset", "Custom"])
        self.module_combo.currentIndexChanged.connect(self._refresh_details)
        form.addRow("Mode:", self.module_combo)

        self.custom_text = QTextEdit()
        self.custom_text.setPlaceholderText(
            "Custom mode: put JSON here (backend-defined).\n"
            "Example: {\"type\":\"global_gp\",\"constraints\":...}"
        )
        self.custom_text.setMinimumHeight(100)
        form.addRow("Custom JSON:", self.custom_text)

        self.dA = QSpinBox()
        self.dA.setRange(2, 8)
        self.dA.setValue(2)
        form.addRow("dA:", self.dA)

        self.dAp = QSpinBox()
        self.dAp.setRange(2, 8)
        self.dAp.setValue(2)
        form.addRow("dA':", self.dAp)

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.01, 50.0)
        self.beta.setSingleStep(0.1)
        self.beta.setValue(1.0)
        form.addRow("β:", self.beta)

        self.eps_eq_global = QDoubleSpinBox()
        self.eps_eq_global.setDecimals(12)
        self.eps_eq_global.setRange(0.0, 1.0)
        self.eps_eq_global.setSingleStep(1e-6)
        self.eps_eq_global.setValue(1e-8)
        form.addRow("ε map (global):", self.eps_eq_global)

        self.eps_eq_local = QDoubleSpinBox()
        self.eps_eq_local.setDecimals(12)
        self.eps_eq_local.setRange(0.0, 1.0)
        self.eps_eq_local.setSingleStep(1e-6)
        self.eps_eq_local.setValue(1e-6)
        form.addRow("ε map (local):", self.eps_eq_local)

        self.eps_gibbs = QDoubleSpinBox()
        self.eps_gibbs.setDecimals(12)
        self.eps_gibbs.setRange(0.0, 1.0)
        self.eps_gibbs.setSingleStep(1e-6)
        self.eps_gibbs.setValue(1e-8)
        form.addRow("ε Gibbs:", self.eps_gibbs)

        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 2000)
        self.num_samples.setValue(50)
        form.addRow("num_samples:", self.num_samples)

        self.seed = QSpinBox()
        self.seed.setRange(-1, 10_000_000)
        self.seed.setValue(0)
        form.addRow("seed:", self.seed)

        self.symmetric = QCheckBox("Use symmetric H (H_A = H_A') when possible")
        self.symmetric.setChecked(True)
        form.addRow("", self.symmetric)

        self.classical = QCheckBox("Classical/diagonal restriction (where applicable)")
        self.classical.setChecked(False)
        form.addRow("", self.classical)

        self.reset_system = QCheckBox("Reset system on Run (force rebuild)")
        self.reset_system.setChecked(False)
        form.addRow("", self.reset_system)

        self.solver = QComboBox()
        self.solver.addItems(["SCS", "CVXOPT", "MOSEK"])
        form.addRow("Solver hint:", self.solver)

        # Variables string preview
        self.vars_preview_line = QLineEdit()
        self.vars_preview_line.setReadOnly(True)
        self.vars_preview_line.setStyleSheet("QLineEdit { background: #f3f3f3; }")
        left.addWidget(QLabel("Variables string (sent to backend):"))
        left.addWidget(self.vars_preview_line)

        preview_btn = QPushButton("Preview variables")
        preview_btn.clicked.connect(self._preview_variables_string)
        left.addWidget(preview_btn)

        # Details pane
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(300)
        right.addWidget(QLabel("Experiment details:"))
        right.addWidget(self.details, 1)

        self.howto = QTextEdit()
        self.howto.setReadOnly(True)
        self.howto.setMinimumHeight(150)
        right.addWidget(QLabel("Notes:"))
        right.addWidget(self.howto, 0)

        # Initial preview
        self._preview_variables_string()

    # -------------------------
    # Experiment catalog
    # -------------------------

    def _default_equations(self) -> list[EquationItem]:
        # Keep IDs stable with whatever your backend switch uses.
        return [
            EquationItem(
                "tfd_vs_dephased",
                "TFD → Dephased TFD (GP vs LGP)",
                "Build a thermo-field-double-like correlated state τ and compare convertibility to a dephased version."
            ),
            EquationItem(
                "random_pair_gp_lgp",
                "Random τ → τ' (GP vs LGP)",
                "Sample random pairs and test feasibility under global and (heuristic) local GP."
            ),
            EquationItem(
                "mix_with_gamma",
                "Thermalisation path: (1−λ)ρ + λ γ⊗γ",
                "Mix a chosen state with the global Gibbs state γ⊗γ and track how monotones and LT distances change with λ."
            ),
            EquationItem(
                "closest_lt_distance",
                "Distance to LT (trace norm SDP)",
                "Compute min_{σ∈LT} 1/2 ||ρ - σ||_1 using the standard trace-norm SDP with P,N ⪰ 0."
            ),
            EquationItem(
                "lt_region_geometry",
                "LT Geometry: Extremal Boundary",
                "Sample extremal locally thermal states (support function SDPs) and plot the boundary in a monotone projection."
            ),
            EquationItem(
                "lt_interior_geometry",
                "LT Geometry: Interior (Random → LT Projection)",
                "Project random quantum states onto the locally thermal set and visualise the interior geometry."
            ),
            EquationItem(
                "lt_geometry_combined",
                "LT Geometry: Boundary + Interior (Final Figure)",
                "Overlay interior LT points (random→LT projection) with boundary extremals and (optionally) the classical LT line."
            ),
            EquationItem(
                "lt_convertibility_graph",
                "LT Convertibility Graph (GP vs LGP)",
                "Generate a small LT ensemble and test pairwise convertibility under global GP vs local GP, outputting adjacency heatmaps and a directed graph plot."
            ),
            EquationItem(
                "extract_global_channel",
                "Extract a Global GP Channel (Choi + optional Kraus)",
                "For one successful GP mapping, output the Choi matrix and verify CPTP + Gibbs-preserving numerically; optionally extract Kraus operators."
            ),
            EquationItem(
                "sanity_checks",
                "Numerical sanity checks table",
                "Generate a compact table of LT errors, GP errors, mapping errors, and monotone changes for one or more example mappings."
            ),
            EquationItem(
                "custom",
                "Custom (backend-defined)",
                "In Custom mode, the JSON spec is passed through verbatim so the backend can build any SDP it wants."
            ),
        ]

    # -------------------------
    # Refresh details
    # -------------------------

    def _refresh_details(self):
        eq_id = self.eq_combo.currentData()
        item = next((x for x in self.eq_items if x.eq_id == eq_id), None)

        if item is None:
            self.details.setPlainText("Unknown experiment id.")
        else:
            self.details.setPlainText(f"{item.title}\n\n{item.description}")

        # Enable/disable custom JSON box based on mode
        custom_mode = (self.module_combo.currentIndex() == 1)
        self.custom_text.setEnabled(custom_mode)

        if custom_mode:
            self.howto.setPlainText(
                "Custom mode: the JSON is passed directly to the backend.\n"
                "If your backend doesn't support custom inputs yet, it will be ignored.\n\n"
                "Tip: keep an eye on the console logs printed by backend_run."
            )
        else:
            self.howto.setPlainText(
                "Preset mode: use the controls on the left.\n"
                "If you change dA/dA'/β/eps/solver and it seems ignored, tick 'Reset system'.\n"
                "Also check the variables string preview to confirm what is being sent."
            )

    # -------------------------
    # Variables string
    # -------------------------

    def _build_variables_string(self) -> str:
        # Backwards-compatible key=value string.
        pairs = []
        pairs.append(_kv("dA", int(self.dA.value())))
        pairs.append(_kv("dAp", int(self.dAp.value())))
        pairs.append(_kv("beta", float(self.beta.value())))
        pairs.append(_kv("eps_eq_global", float(self.eps_eq_global.value())))
        pairs.append(_kv("eps_eq_local", float(self.eps_eq_local.value())))
        pairs.append(_kv("eps_gibbs", float(self.eps_gibbs.value())))
        pairs.append(_kv("num_samples", int(self.num_samples.value())))
        pairs.append(_kv("seed", int(self.seed.value())))
        pairs.append(_kv("symmetric", self.symmetric.isChecked()))
        pairs.append(_kv("classical", self.classical.isChecked()))
        pairs.append(_kv("reset_system", self.reset_system.isChecked()))

        # Solver hint (backend may ignore)
        solver_map = {
            0: "SCS",
            1: "CVXOPT",
            2: "MOSEK",
        }
        pairs.append(_kv("solver", solver_map.get(self.solver.currentIndex(), "SCS")))

        return ", ".join(pairs)

    def _build_config(self) -> Dict[str, Any]:
        eq_id = self.eq_combo.currentData()
        module_type = "custom" if self.module_combo.currentIndex() == 1 else "preset"

        variables_str = self._build_variables_string()
        custom_function = self.custom_text.toPlainText().strip()

        # Validate JSON if in custom mode (but still pass through even if invalid, with warning).
        if module_type == "custom" and custom_function:
            try:
                json.loads(custom_function)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Custom JSON looks invalid",
                    "Your Custom SDP input is not valid JSON.\n\n"
                    f"Error: {e}\n\n"
                    "It will still be passed to the backend as a raw string."
                )

        return {
            "selected_equation_id": eq_id,
            "selected_equation_name": self.eq_combo.currentText(),
            "module_type": module_type,
            "variables_str": variables_str,
            "custom_function": custom_function,
        }

    def _preview_variables_string(self):
        s = self._build_variables_string()
        self.vars_preview_line.setText(s)

    def _on_run(self):
        cfg = self._build_config()
        # Keep preview line up-to-date when running
        self.vars_preview_line.setText(cfg["variables_str"])
        if self.run_callback:
            self.run_callback(cfg)


# -------------------------
# Manual test (optional)
# -------------------------

def _demo():
    import sys

    def _run(cfg):
        # Replace with real backend call.
        QMessageBox.information(None, "Run pressed", json.dumps(cfg, indent=2))

    app = QApplication(sys.argv)
    w = LTSDPWindow(_run)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _demo()
