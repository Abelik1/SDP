from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Set

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .registry import GROUPS_ORDER, find_by_id, get_catalog


FAV_FILE = os.path.join(os.path.dirname(__file__), "favorites.json")


def _kv(k: str, v: Any) -> str:
    if isinstance(v, bool):
        return f"{k}={'true' if v else 'false'}"
    return f"{k}={v}"


class LTGPMainWindow(QWidget):
    """Grouped/searchable GUI.

    Contract:
      run_callback(cfg_dict) -> result_dict
    """

    def __init__(self, run_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, parent=None):
        super().__init__(parent)
        self.run_callback = run_callback
        self._favorites: Set[str] = set()
        self._last_result: Optional[Dict[str, Any]] = None

        self.setWindowTitle("LT / GP SDP Research Tool")
        self.setMinimumWidth(1050)

        self._load_favorites()
        self._build_ui()
        self._rebuild_tree()

    def _load_favorites(self):
        try:
            if os.path.exists(FAV_FILE):
                with open(FAV_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._favorites = set(data.get("favorites", []))
        except Exception:
            self._favorites = set()

    def _save_favorites(self):
        try:
            with open(FAV_FILE, "w", encoding="utf-8") as f:
                json.dump({"favorites": sorted(self._favorites)}, f, indent=2)
        except Exception:
            pass

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        root.addLayout(top)

        top.addWidget(QLabel("Search:"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter experiments (title / id / tags)")
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search, 1)

        self.btn_fav = QPushButton("☆ Favorite")
        self.btn_fav.clicked.connect(self._toggle_favorite)
        top.addWidget(self.btn_fav)

        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self._on_run)
        top.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export last result")
        self.btn_export.clicked.connect(self._export_last_result)
        self.btn_export.setEnabled(False)
        top.addWidget(self.btn_export)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        left = QWidget(); left_layout = QVBoxLayout(left)
        right = QWidget(); right_layout = QVBoxLayout(right)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        left_layout.addWidget(QLabel("Experiments"))
        left_layout.addWidget(self.tree, 1)

        controls = QGroupBox("Controls")
        form = QFormLayout(controls)
        left_layout.addWidget(controls)

        self.mode = QComboBox()
        self.mode.addItems(["Preset", "Custom"])
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self.mode)

        self.custom_json = QTextEdit()
        self.custom_json.setPlaceholderText('{"type":"...","params":{...}}')
        self.custom_json.setMinimumHeight(80)
        form.addRow("Custom JSON:", self.custom_json)

        self.dA = QSpinBox(); self.dA.setRange(2, 8); self.dA.setValue(2)
        self.dAp = QSpinBox(); self.dAp.setRange(2, 8); self.dAp.setValue(2)
        form.addRow("dA:", self.dA)
        form.addRow("dA':", self.dAp)

        self.beta = QDoubleSpinBox(); self.beta.setRange(0.01, 50.0); self.beta.setValue(1.0); self.beta.setSingleStep(0.1)
        form.addRow("β:", self.beta)

        self.eps_eq_global = QDoubleSpinBox(); self.eps_eq_global.setDecimals(12); self.eps_eq_global.setRange(0.0, 1.0); self.eps_eq_global.setValue(1e-6)
        self.eps_eq_local = QDoubleSpinBox(); self.eps_eq_local.setDecimals(12); self.eps_eq_local.setRange(0.0, 1.0); self.eps_eq_local.setValue(1e-6)
        self.eps_gibbs = QDoubleSpinBox(); self.eps_gibbs.setDecimals(12); self.eps_gibbs.setRange(0.0, 1.0); self.eps_gibbs.setValue(1e-8)
        form.addRow("ε map (global):", self.eps_eq_global)
        form.addRow("ε map (local):", self.eps_eq_local)
        form.addRow("ε Gibbs:", self.eps_gibbs)

        self.num_samples = QSpinBox(); self.num_samples.setRange(1, 5000); self.num_samples.setValue(50)
        form.addRow("num_samples:", self.num_samples)

        self.seed = QSpinBox(); self.seed.setRange(-1, 10_000_000); self.seed.setValue(0)
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

        self.solver = QComboBox(); self.solver.addItems(["SCS", "CVXOPT", "MOSEK", "AUTO"])
        form.addRow("Solver hint:", self.solver)

        self.extra_vars = QLineEdit()
        self.extra_vars.setPlaceholderText("Extra vars (optional): k=v, k2=v2 ...")
        form.addRow("Extra vars:", self.extra_vars)

        self.vars_preview = QLineEdit(); self.vars_preview.setReadOnly(True)
        left_layout.addWidget(QLabel("Variables string (sent to backend):"))
        left_layout.addWidget(self.vars_preview)

        btn_preview = QPushButton("Preview variables")
        btn_preview.clicked.connect(self._preview_variables)
        left_layout.addWidget(btn_preview)

        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMinimumHeight(200)
        right_layout.addWidget(QLabel("Experiment details:"))
        right_layout.addWidget(self.details, 1)

        self.results = QTextEdit(); self.results.setReadOnly(True); self.results.setMinimumHeight(260)
        right_layout.addWidget(QLabel("Results:"))
        right_layout.addWidget(self.results, 1)

        self._on_mode_changed()
        self._preview_variables()

    def _rebuild_tree(self):
        self.tree.clear()
        catalog = get_catalog()

        fav_root = QTreeWidgetItem(["★ Favorites"])
        self.tree.addTopLevelItem(fav_root)
        for gid in sorted(self._favorites):
            spec = find_by_id(gid)
            if spec is None:
                continue
            it = QTreeWidgetItem([f"{spec.title}  ({spec.eq_id})"])
            it.setData(0, Qt.UserRole, spec.eq_id)
            fav_root.addChild(it)
        fav_root.setExpanded(True)

        for group in GROUPS_ORDER:
            gitem = QTreeWidgetItem([group])
            self.tree.addTopLevelItem(gitem)
            for spec in catalog.get(group, []):
                it = QTreeWidgetItem([f"{spec.title}  ({spec.eq_id})"])
                it.setData(0, Qt.UserRole, spec.eq_id)
                gitem.addChild(it)
            gitem.setExpanded(False)

        leaf = None
        if fav_root.childCount() > 0:
            leaf = fav_root.child(0)
        else:
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                if top is fav_root:
                    continue
                if top.childCount() > 0:
                    leaf = top.child(0)
                    top.setExpanded(True)
                    break
        if leaf is not None:
            self.tree.setCurrentItem(leaf)

    def _apply_filter(self):
        q = self.search.text().strip().lower()
        if not q:
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                top.setHidden(False)
                for k in range(top.childCount()):
                    top.child(k).setHidden(False)
            return

        catalog = get_catalog()
        tag_map = {}
        for _, specs in catalog.items():
            for s in specs:
                tag_map[s.eq_id] = " ".join([s.eq_id, s.title, " ".join(s.tags)]).lower()

        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            any_visible = False
            for k in range(top.childCount()):
                leaf = top.child(k)
                eq_id = leaf.data(0, Qt.UserRole)
                hay = tag_map.get(eq_id, leaf.text(0).lower())
                ok = q in hay
                leaf.setHidden(not ok)
                any_visible = any_visible or ok
            top.setHidden(not any_visible)
            if any_visible:
                top.setExpanded(True)

    def _selected_eq_id(self) -> Optional[str]:
        it = self.tree.currentItem()
        if it is None:
            return None
        eq_id = it.data(0, Qt.UserRole)
        if not eq_id:
            return None
        return str(eq_id)

    def _on_tree_selection(self):
        eq_id = self._selected_eq_id()
        if not eq_id:
            self.details.setPlainText("Select an experiment.")
            self.btn_fav.setEnabled(False)
            return

        self.btn_fav.setEnabled(True)
        self.btn_fav.setText("★ Unfavorite" if eq_id in self._favorites else "☆ Favorite")

        spec = find_by_id(eq_id)
        if spec is None:
            self.details.setPlainText(f"Unknown experiment id: {eq_id}")
        else:
            self.details.setPlainText(f"{spec.title}\n\n{spec.description}")

    def _toggle_favorite(self):
        eq_id = self._selected_eq_id()
        if not eq_id:
            return
        if eq_id in self._favorites:
            self._favorites.remove(eq_id)
        else:
            self._favorites.add(eq_id)
        self._save_favorites()
        self._rebuild_tree()
        self._apply_filter()

    def _on_mode_changed(self):
        custom = (self.mode.currentIndex() == 1)
        self.custom_json.setEnabled(custom)

    def _build_variables_string(self) -> str:
        parts = [
            _kv("dA", int(self.dA.value())),
            _kv("dAp", int(self.dAp.value())),
            _kv("beta", float(self.beta.value())),
            _kv("eps_eq_global", float(self.eps_eq_global.value())),
            _kv("eps_eq_local", float(self.eps_eq_local.value())),
            _kv("eps_gibbs", float(self.eps_gibbs.value())),
            _kv("num_samples", int(self.num_samples.value())),
            _kv("seed", int(self.seed.value())),
            _kv("symmetric", bool(self.symmetric.isChecked())),
            _kv("classical", bool(self.classical.isChecked())),
            _kv("reset_system", bool(self.reset_system.isChecked())),
            _kv("solver", str(self.solver.currentText())),
        ]
        extra = self.extra_vars.text().strip()
        if extra:
            parts.append(extra)
        return ", ".join(parts)

    def _preview_variables(self):
        self.vars_preview.setText(self._build_variables_string())

    def _on_run(self):
        if self.run_callback is None:
            QMessageBox.warning(self, "No backend", "No run_callback configured.")
            return

        eq_id = self._selected_eq_id()
        if not eq_id:
            QMessageBox.warning(self, "No experiment", "Select an experiment first.")
            return

        spec = find_by_id(eq_id)
        eq_name = spec.title if spec else eq_id

        cfg: Dict[str, Any] = {
            "selected_equation_id": eq_id,
            "selected_equation_name": eq_name,
            "variables_str": self._build_variables_string(),
            "custom_function": self.custom_json.toPlainText() if self.mode.currentIndex() == 1 else "",
        }

        try:
            res = self.run_callback(cfg)
            self._last_result = res
            self.btn_export.setEnabled(isinstance(res, dict))

            if isinstance(res, dict):
                text = res.get("summary", "(no summary)")
                run_dir = res.get("run_dir")
                if run_dir:
                    text = f"Run dir: {run_dir}\n\n" + text
                self.results.setPlainText(text)
            else:
                self.results.setPlainText(str(res))
        except Exception as e:
            QMessageBox.critical(self, "Backend error", str(e))

    def _export_last_result(self):
        if not isinstance(self._last_result, dict):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result JSON", "result.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_result, f, indent=2, sort_keys=True, default=str)
            QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))