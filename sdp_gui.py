# sdp_gui.py
#
# PyQt5 GUI for selecting equations and configuring SDP runs.
# You can import LTSDPWindow and launch_gui(run_callback) from main.py.

from PyQt5 import QtWidgets, QtCore


class LTSDPWindow(QtWidgets.QMainWindow):
    """
    Main window for LT/SDP front-end.

    Parameters
    ----------
    run_callback : callable or None
        Function called when "Run" is pressed.
        Signature: run_callback(config: dict)
        where config includes:
          - "selected_equation_id"
          - "selected_equation_name"
          - "module_type"
          - "variables"
          - "custom_function"
    """

    def __init__(self, run_callback=None, parent=None):
        super().__init__(parent)
        self.run_callback = run_callback

        self.setWindowTitle("LT / SDP Explorer")
        self.resize(900, 600)

        self._init_equation_data()
        self._build_ui()
        self._connect_signals()

    # ---------- Equation data ----------

    def _init_equation_data(self):
        """
        Define a small set of example "equations" to populate the table.
        You can edit/extend this as you like.
        """
        self.equations = [
            {
                "id": "tfd_vs_dephased",
                "name": "TFD vs dephased TFD",
                "category": "Pure LT",
                "summary": "Compare pure TFD to its dephased (classical LT) version.",
                "details": (
                    "Investigates the difference between a pure thermofield double (TFD) "
                    "state and its dephased version in the energy basis.\n\n"
                    "Metrics of interest:\n"
                    " - Mutual information I(A:B)\n"
                    " - Relative entropy D(ρ || γ⊗γ)\n"
                    " - Distance to classical LT subset\n"
                    " - Distance to full LT set"
                ),
            },
            {
                "id": "classical_LT_line",
                "name": "Classical LT line (2×2)",
                "category": "Classical LT",
                "summary": "Scan the 2×2 LT transportation polytope p(a).",
                "details": (
                    "Scans the full set of classical locally thermal states in the 2×2 case, "
                    "parameterised by a single parameter a.\n\n"
                    "Each point satisfies:\n"
                    " - Fixed Gibbs marginals on A and A'\n"
                    " - Varying classical correlations\n\n"
                    "Useful to map I(A:B) as a function of classical correlation strength."
                ),
            },
            {
                "id": "random_pair_gp_lgp",
                "name": "Random pair (GP vs LGP)",
                "category": "Convertibility",
                "summary": "Test random τ → τ' under global and local GP.",
                "details": (
                    "Samples random bipartite states τ and τ', and checks whether there exists:\n"
                    " - A global Gibbs-preserving map G such that G(τ) = τ'\n"
                    " - A composition of local GP maps on A and A' achieving the same\n\n"
                    "Useful to probe differences between global and local GPOs."
                ),
            },
            {
                "id": "mix_with_gamma",
                "name": "Mixture with γ⊗γ",
                "category": "Thermalisation path",
                "summary": "Study ρ(λ) = (1-λ)ρ + λ γ⊗γ.",
                "details": (
                    "Takes a reference LT state ρ and studies the family:\n"
                    "   ρ(λ) = (1 - λ) ρ + λ (γ⊗γ)\n"
                    "for λ ∈ [0,1].\n\n"
                    "Useful to visualise decay of mutual information and free energy "
                    "along a simple thermalisation path inside the LT slice."
                ),
            },
            {
                "id": "extremal_LT_boundary",
                "name": "Extremal LT boundary (Phase 4)",
                "category": "Geometry",
                "summary": "Sample extremal LT states using support function SDPs.",
                "details": (
                    "Uses random Hermitian directions K and solves the extremal LT SDP:\n"
                    "   maximise Tr(K ρ) over LT (or classical LT) states.\n\n"
                    "Collects the resulting extremal states and plots them in the\n"
                    "plane of (D(ρ || γ⊗γ), I(A:B)), optionally overlaying the\n"
                    "classical LT line for comparison."
                ),
            },

        ]

    # ---------- UI construction ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # --- Header row with title + Run button (top-right) ---
        header_layout = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("Locally Thermal / SDP Explorer")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Run the selected equation with the current parameters")
        header_layout.addWidget(self.run_button)

        main_layout.addLayout(header_layout)

        # --- Middle: splitter with equation table (left) and details (right) ---
        mid_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: equations "table" using QTreeWidget
        self.eq_tree = QtWidgets.QTreeWidget()
        self.eq_tree.setHeaderLabels(["Name", "Category", "Summary"])
        self.eq_tree.setColumnWidth(0, 200)
        self.eq_tree.setColumnWidth(1, 130)
        self.eq_tree.setAlternatingRowColors(True)
        self.eq_tree.setRootIsDecorated(True)  # allows expansion arrow

        # Populate tree
        for eq in self.equations:
            item = QtWidgets.QTreeWidgetItem([
                eq["name"],
                eq["category"],
                eq["summary"],
            ])
            item.setData(0, QtCore.Qt.UserRole, eq["id"])  # store ID
            # Add a child item for details (so clicking triangle "expands")
            child = QtWidgets.QTreeWidgetItem(["Details", "", "Double-click or see right panel"])
            item.addChild(child)
            self.eq_tree.addTopLevelItem(item)

        mid_splitter.addWidget(self.eq_tree)

        # Right: details text
        self.details_box = QtWidgets.QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setPlaceholderText("Select an equation on the left to see details here.")
        mid_splitter.addWidget(self.details_box)

        mid_splitter.setStretchFactor(0, 2)
        mid_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(mid_splitter, stretch=5)

        # --- Bottom: parameter configuration group ---
        self.param_group = QtWidgets.QGroupBox("Parameters")
        form_layout = QtWidgets.QFormLayout(self.param_group)

        # Module type
        self.module_type_combo = QtWidgets.QComboBox()
        self.module_type_combo.addItems([
            "LT geometry",
            "Convertibility",
            "Classical LT scan",
            "Custom",
        ])
        form_layout.addRow("Module type:", self.module_type_combo)

        # Variables input (free text for now: e.g., 'beta=1.0, d=2')
        self.variables_line = QtWidgets.QLineEdit()
        self.variables_line.setPlaceholderText("e.g. beta=1.0, d=2, lam=0.3")
        form_layout.addRow("Variables:", self.variables_line)

        # Custom function (multi-line, e.g. custom SDP expression or code hint)
        self.custom_func_text = QtWidgets.QTextEdit()
        self.custom_func_text.setPlaceholderText(
            "Optional: paste or describe a custom function / configuration here.\n"
            "For example, a specific family of states ρ(θ) or a custom constraint."
        )
        form_layout.addRow("Custom SDP input:", self.custom_func_text)

        main_layout.addWidget(self.param_group, stretch=2)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready.")

        # By default select the first equation
        if self.eq_tree.topLevelItemCount() > 0:
            first_item = self.eq_tree.topLevelItem(0)
            self.eq_tree.setCurrentItem(first_item)
            self._update_details_from_item(first_item)

    # ---------- Signal connections ----------

    def _connect_signals(self):
        self.eq_tree.itemSelectionChanged.connect(self.on_equation_selected)
        self.run_button.clicked.connect(self.on_run_clicked)

    # ---------- Helpers ----------

    def _find_equation_by_id(self, eq_id):
        for eq in self.equations:
            if eq["id"] == eq_id:
                return eq
        return None

    def _update_details_from_item(self, item):
        eq_id = item.data(0, QtCore.Qt.UserRole)
        eq = self._find_equation_by_id(eq_id)
        if eq is None:
            self.details_box.clear()
            return
        text = (
            f"<b>{eq['name']}</b><br>"
            f"<i>Category: {eq['category']}</i><br><br>"
            f"{eq['details']}"
        )
        self.details_box.setHtml(text)

    # ---------- Slots ----------

    def on_equation_selected(self):
        item = self.eq_tree.currentItem()
        # Only use top-level item for details
        if item is None:
            return
        if item.parent() is not None:
            # If user clicked the "Details" child, use parent
            item = item.parent()
            self.eq_tree.setCurrentItem(item)

        self._update_details_from_item(item)
        eq_id = item.data(0, QtCore.Qt.UserRole)
        eq = self._find_equation_by_id(eq_id)
        if eq:
            self.status_bar.showMessage(f"Selected: {eq['name']}")

    def on_run_clicked(self):
        # Gather selected equation
        item = self.eq_tree.currentItem()
        if item is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No selection",
                "Please select an equation to run.",
            )
            return

        if item.parent() is not None:
            item = item.parent()

        eq_id = item.data(0, QtCore.Qt.UserRole)
        eq = self._find_equation_by_id(eq_id)

        # Gather parameters
        module_type = self.module_type_combo.currentText()
        variables_str = self.variables_line.text()
        custom_func_str = self.custom_func_text.toPlainText()

        config = {
            "selected_equation_id": eq_id,
            "selected_equation_name": eq["name"] if eq else None,
            "module_type": module_type,
            "variables": variables_str,
            "custom_function": custom_func_str,
        }

        # Call backend callback if present; otherwise just print
        if self.run_callback is not None:
            try:
                self.run_callback(config)
                self.status_bar.showMessage("Run completed.", 5000)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"An error occurred while running the backend:\n{e}",
                )
                self.status_bar.showMessage("Run failed.", 5000)
        else:
            # Fallback behaviour: print to console
            print("Run button pressed with configuration:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            self.status_bar.showMessage("Run pressed (no backend callback attached).", 5000)


def launch_gui(run_callback=None):
    """
    Convenience function to launch the GUI standalone.

    Example:
        from sdp_gui import launch_gui
        launch_gui()
    """
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = LTSDPWindow(run_callback=run_callback)
    win.show()
    sys.exit(app.exec_())
