
"""
sdp_gui.py — Revamped UI for LT / GP SDP Experiments

Goals:
- Clean, structured UX: no "mystery" free-text variables required.
- Fixed parameter widgets (spinboxes, checkboxes, dropdowns).
- Custom SDP input supported as JSON (shown with examples + validation).
- Backwards compatible: still emits `variables_str` in key=value CSV format
  so existing backend code using `parse_variables_string` keeps working.

Expected config keys produced on Run:
  - selected_equation_id: str
  - module_type: str
  - variables_str: str   (e.g. "beta=1.0, num_samples=30, symmetric=1")
  - custom_function: str (JSON text; may be empty if not in Custom mode)

If your backend expects additional keys, add them in `_build_config()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit,
    QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox, QTextEdit, QSplitter, QMessageBox,
    QSizePolicy
)

import json


# -------------------------
# Helpers
# -------------------------

def _kv(key: str, value: Any) -> str:
    """Serialize key=value for the legacy variables_str channel."""
    if isinstance(value, bool):
        return f"{key}={1 if value else 0}"
    return f"{key}={value}"

def _pretty_json_default() -> str:
    example = {
        "task": "convertibility",
        "tau": "tfd",
        "tau_p": "dephase_global(tau)",
        "check_global": True,
        "check_local": True,
        "eps_eq_global": 1e-8,
        "eps_eq_local": 1e-6
    }
    return json.dumps(example, indent=2)

@dataclass
class EquationItem:
    eq_id: str
    title: str
    description: str


# -------------------------
# Main Window
# -------------------------

class LTSDPWindow(QMainWindow):
    """
    A clean UI that:
    - lets you pick a preset experiment (Equation)
    - choose module type / algorithm mode
    - set parameters via widgets
    - (optionally) provide Custom JSON spec
    """

    def __init__(self, run_callback, equations: Optional[list[EquationItem]] = None, parent=None):
        super().__init__(parent)
        self.run_callback = run_callback
        self.setWindowTitle("LT / GP SDP — Experiments")
        self.resize(1100, 720)

        self.equations = equations or self._default_equations()

        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QLabel("Locally Thermal (LT) & Gibbs-Preserving (GP) SDP Toolkit")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        outer.addWidget(header)

        sub = QLabel(
            "Pick an experiment preset, set parameters, then Run. "
            "For Custom mode, paste a JSON spec in the Custom box."
        )
        sub.setStyleSheet("color: #555;")
        outer.addWidget(sub)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(splitter, 1)

        # Left: experiment selection + help
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        splitter.addWidget(left)

        self.eq_combo = QComboBox()
        for item in self.equations:
            self.eq_combo.addItem(item.title, item.eq_id)
        self.eq_combo.currentIndexChanged.connect(self._refresh_details)
        left_layout.addWidget(self._boxed("Experiment", self.eq_combo))

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(220)
        left_layout.addWidget(self._boxed("Description", self.details))

        self.howto = QTextEdit()
        self.howto.setReadOnly(True)
        self.howto.setMinimumHeight(220)
        self.howto.setText(self._howto_text())
        left_layout.addWidget(self._boxed("How to use inputs", self.howto))

        # Right: parameters + custom JSON
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Module selection row
        self.module_combo = QComboBox()
        self.module_combo.addItems([
            "Preset (recommended)",
            "Custom (JSON spec)"
        ])
        self.module_combo.currentIndexChanged.connect(self._toggle_custom)
        right_layout.addWidget(self._boxed("Mode", self.module_combo))

        # Parameter grid
        right_layout.addWidget(self._params_box())

        # Custom JSON box
        self.custom_text = QTextEdit()
        self.custom_text.setPlaceholderText("Paste Custom JSON spec here (only used in Custom mode).")
        self.custom_text.setText(_pretty_json_default())
        self.custom_box = self._boxed("Custom SDP input (JSON)", self.custom_text)
        right_layout.addWidget(self.custom_box)

        # Action bar
        actions = QHBoxLayout()
        actions.addStretch(1)

        self.preview_vars = QPushButton("Preview variables_str")
        self.preview_vars.clicked.connect(self._preview_variables_string)
        actions.addWidget(self.preview_vars)

        self.run_btn = QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._on_run)
        actions.addWidget(self.run_btn)

        right_layout.addLayout(actions)

        self._refresh_details()
        self._toggle_custom()

    # -------------------------
    # UI builders
    # -------------------------

    def _boxed(self, title: str, widget: QWidget) -> QGroupBox:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget)
        return box

    def _params_box(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        grid = QGridLayout(box)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        r = 0

        # Dimensions
        self.dA = QSpinBox(); self.dA.setRange(2, 16); self.dA.setValue(2)
        self.dAp = QSpinBox(); self.dAp.setRange(2, 16); self.dAp.setValue(2)
        grid.addWidget(QLabel("dA"), r, 0); grid.addWidget(self.dA, r, 1)
        grid.addWidget(QLabel("dA'"), r, 2); grid.addWidget(self.dAp, r, 3)
        r += 1

        # Thermo params
        self.beta = QDoubleSpinBox(); self.beta.setDecimals(6); self.beta.setRange(0.0, 1000.0); self.beta.setValue(1.0)
        self.beta.setSingleStep(0.1)
        grid.addWidget(QLabel("β (inverse temperature)"), r, 0); grid.addWidget(self.beta, r, 1, 1, 3)
        r += 1

        # Tolerances
        self.eps_eq_global = QDoubleSpinBox(); self.eps_eq_global.setDecimals(12); self.eps_eq_global.setRange(0.0, 1.0); self.eps_eq_global.setValue(1e-8)
        self.eps_eq_global.setSingleStep(1e-8)
        self.eps_eq_local = QDoubleSpinBox(); self.eps_eq_local.setDecimals(12); self.eps_eq_local.setRange(0.0, 1.0); self.eps_eq_local.setValue(1e-6)
        self.eps_eq_local.setSingleStep(1e-6)
        grid.addWidget(QLabel("ε_eq (global GP)"), r, 0); grid.addWidget(self.eps_eq_global, r, 1)
        grid.addWidget(QLabel("ε_eq (local GP)"), r, 2); grid.addWidget(self.eps_eq_local, r, 3)
        r += 1

        self.eps_gibbs = QDoubleSpinBox(); self.eps_gibbs.setDecimals(12); self.eps_gibbs.setRange(0.0, 1.0); self.eps_gibbs.setValue(1e-8)
        self.eps_gibbs.setSingleStep(1e-8)
        grid.addWidget(QLabel("ε_Gibbs (GP constraint)"), r, 0); grid.addWidget(self.eps_gibbs, r, 1, 1, 3)
        r += 1

        # Sampling / misc
        self.num_samples = QSpinBox(); self.num_samples.setRange(1, 5000); self.num_samples.setValue(40)
        grid.addWidget(QLabel("num_samples"), r, 0); grid.addWidget(self.num_samples, r, 1)

        self.seed = QSpinBox(); self.seed.setRange(-1, 10_000_000); self.seed.setValue(0)
        grid.addWidget(QLabel("seed"), r, 2); grid.addWidget(self.seed, r, 3)
        r += 1

        # Algorithm toggles
        self.symmetric = QCheckBox("symmetric (γA = γA')")
        self.symmetric.setChecked(True)
        self.classical = QCheckBox("classical LT (diagonal only)")
        self.reset_system = QCheckBox("reset_system (rebuild Gibbs states)")
        self.reset_system.setChecked(False)

        grid.addWidget(self.symmetric, r, 0, 1, 2)
        grid.addWidget(self.classical, r, 2, 1, 2)
        r += 1
        grid.addWidget(self.reset_system, r, 0, 1, 4)
        r += 1

        # Solver dropdown (backend may ignore; included for future use)
        self.solver = QComboBox()
        self.solver.addItems(["SCS (default)", "CVXOPT", "MOSEK (if installed)"])
        grid.addWidget(QLabel("solver"), r, 0); grid.addWidget(self.solver, r, 1, 1, 3)
        r += 1

        # Compact status line
        self.vars_preview_line = QLineEdit()
        self.vars_preview_line.setReadOnly(True)
        self.vars_preview_line.setPlaceholderText("variables_str preview will appear here.")
        grid.addWidget(QLabel("variables_str"), r, 0)
        grid.addWidget(self.vars_preview_line, r, 1, 1, 3)

        return box

    # -------------------------
    # Text blocks
    # -------------------------

    def _howto_text(self) -> str:
        return (
            "This UI writes a legacy `variables_str` (comma-separated key=value) so your existing backend\n"
            "can keep using `parse_variables_string`.\n\n"
            "Common variables:\n"
            "  beta=1.0                inverse temperature\n"
            "  dA=2, dAp=2             local dimensions\n"
            "  eps_eq_global=1e-8      conversion tolerance for global GP\n"
            "  eps_eq_local=1e-6       conversion tolerance for local GP\n"
            "  eps_gibbs=1e-8          tolerance for GP constraint G(γ)=γ\n"
            "  num_samples=40          sampling size (for random/extremal scans)\n"
            "  seed=0                  RNG seed\n"
            "  symmetric=1             enforce γA=γA'\n"
            "  classical=1             diagonal (classical) LT variant\n"
            "  reset_system=1          force recompute Gibbs states when beta/dims change\n\n"
            "Custom mode:\n"
            "- Paste a JSON spec in the Custom box. Your backend can parse it instead of `variables_str`.\n"
            "- Recommended fields: task, tau, tau_p, check_global, check_local, eps_eq_global, eps_eq_local.\n"
        )

    # -------------------------
    # Behaviors
    # -------------------------

    def _default_equations(self) -> list[EquationItem]:
        # Keep IDs stable with whatever your backend_run switch uses.
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
                "closest_lt_distance",
                "Distance to LT (trace norm SDP)",
                "Compute min_{σ∈LT} 1/2 ||ρ - σ||_1 using the standard SDP with P,N ⪰ 0."
            ),
            EquationItem(
                "lt_region_geometry",
                "LT Region Geometry (Extremal Boundary)",
                "Sample extremal locally thermal states and plot their geometric region"
            ),
            EquationItem(
                "lt_interior_geometry",
                "LT Interior Geometry (Random → LT Projection)",
                "Project random quantum states onto the locally thermal set and visualise the interior geometry."
            ),
            EquationItem(
                "custom",
                "Custom (backend-defined)",
                "In Custom mode, the JSON spec is passed through verbatim so the backend can build any SDP it wants."
            ),
        ]

    def _refresh_details(self):
        eq_id = self.eq_combo.currentData()
        item = next((x for x in self.equations if x.eq_id == eq_id), None)
        if not item:
            self.details.setText("")
            return
        self.details.setText(
            f"{item.title}\n\n"
            f"ID: {item.eq_id}\n\n"
            f"{item.description}\n\n"
            "Outputs depend on your backend preset implementation."
        )

    def _toggle_custom(self):
        is_custom = (self.module_combo.currentIndex() == 1)  # Custom (JSON spec)
        self.custom_box.setEnabled(is_custom)
        self.custom_text.setEnabled(is_custom)

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
