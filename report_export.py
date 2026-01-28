"""report_export.py

Generate a LaTeX report + compiled PDF summarizing an LT/GP SDP run.

Outputs
- <output_dir>/<stem>.tex
- <output_dir>/<stem>.pdf

The report includes:
- Run metadata (date/time, selected experiment, parameters)
- A narrative explanation section (TFD, dephasing, LT convexity, correlation operator X_AB)
- Results text (whatever your backend prints/returns)
- Optional appendix: extracted text from key project PDFs

Usage (backend integration)
--------------------------

    from report_export import generate_pdf_report

    pdf_path = generate_pdf_report(
        config=config,
        results_text=results_text,
        output_dir="reports",
        stem="run_001",
        include_project_pdfs=True,
        project_pdf_paths=[
            "Capstone_Notes.pdf",
            "Capstone_Questions.pdf",
            "PYU44TP1_Project_Guidelines_FINAL_2025-26.pdf",
        ],
    )

Notes
-----
- This module calls `pdflatex`. Ensure TeX Live / MiKTeX is installed on your machine.
- In this sandbox environment pdflatex exists; on your laptop it should as well if you've installed LaTeX.
- Extracted PDF text is appended as an Appendix (it is not perfect formatting; it's meant as a reference).

"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


# -----------------------------
# LaTeX helpers
# -----------------------------

_LATEX_SPECIALS = {
    "\\": r"\\textbackslash{}",
    "{": r"\\{",
    "}": r"\\}",
    "#": r"\\#",
    "$": r"\\$",
    "%": r"\\%",
    "&": r"\\&",
    "_": r"\\_",
    "^": r"\\textasciicircum{}",
    "~": r"\\textasciitilde{}",
}


def latex_escape(text: str) -> str:
    """Escape a string for safe inclusion in LaTeX."""
    if text is None:
        return ""
    out = []
    for ch in text:
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # keep paragraphs but collapse excessive internal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _latex_section(title: str, body: str) -> str:
    return f"\\section{{{latex_escape(title)}}}\n{body}\n"


def _latex_subsection(title: str, body: str) -> str:
    return f"\\subsection{{{latex_escape(title)}}}\n{body}\n"


def _latex_verbatim_block(title: str, body: str) -> str:
    body = body.replace("\t", "    ")
    return (
        f"\\subsection{{{latex_escape(title)}}}\n"
        "\\begin{verbatim}\n"
        f"{body}\n"
        "\\end{verbatim}\n"
    )


# -----------------------------
# Project narrative block
# -----------------------------

_DEFAULT_NARRATIVE = r"""
This report summarizes a run of an SDP-based toolkit for studying \emph{locally thermal} (LT) state spaces
and state transformations under \emph{Gibbs-preserving operations} (GPOs) in symmetric thermodynamic settings.

\paragraph{Local vs global thermal structure.}
For a bipartite system $AB$ with a non-interacting Hamiltonian
$H_{AB}=H_A\otimes I_B + I_A\otimes H_B$, the global Gibbs state factorizes as
$\gamma_{AB}=\gamma_A\otimes\gamma_B$.
The individual reduced states $\gamma_A$ and $\gamma_B$ are $2\times 2$ for qubits, while $\gamma_{AB}$ is $4\times 4$.

\paragraph{Locally thermal states.}
A state $\sigma_{AB}$ is \emph{locally thermal} if $\mathrm{Tr}_B\,\sigma_{AB}=\gamma_A$ and
$\mathrm{Tr}_A\,\sigma_{AB}=\gamma_B$. The set LT is convex.

\paragraph{TFD, dephasing, and classicalization.}
A thermo-field-double-like correlated state (TFD) is a canonical example of a state that is globally
athermal yet locally thermal. Dephasing in an energy basis removes off-diagonal coherences, producing a
\emph{dephased TFD}. Measuring (or fully decohering) in the energy basis yields a purely classically correlated LT state.
Convex mixtures such as $(1-p)\,\rho^M_{AB}+p\,\rho^X_{AB}$ remain LT for $p\in[0,1]$.

\paragraph{Correlation operator.}
It is often convenient to isolate correlations relative to the product Gibbs baseline via
$X_{AB}:=\rho_{AB}-\gamma_A\otimes\gamma_B$.
For LT states, $\mathrm{Tr}_A X_{AB}=\mathrm{Tr}_B X_{AB}=0$.

\paragraph{SDP viewpoint.}
The convertibility tests implemented in the codebase are feasibility SDPs over Choi matrices, enforcing
complete positivity and trace preservation, plus the Gibbs-preserving constraint $\mathcal{G}(\gamma_{AB})=\gamma_{AB}$.
Distances to LT (or classical LT) are computed via the standard trace-norm SDP.
"""


# -----------------------------
# PDF extraction
# -----------------------------

@dataclass
class ExtractedPDF:
    title: str
    path: str
    text: str


def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """Extract text from a PDF. Falls back to empty string if pypdf unavailable."""
    if PdfReader is None:
        return ""

    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    if max_pages is not None:
        pages = pages[:max_pages]

    chunks: List[str] = []
    for p in pages:
        t = p.extract_text() or ""
        chunks.append(t)
    return collapse_whitespace("\n\n".join(chunks))


def build_pdf_appendix(pdf_items: List[ExtractedPDF]) -> str:
    if not pdf_items:
        return ""

    parts = ["\\appendix\n", "\\section{Project documents (extracted text)}\n"]
    parts.append(
        "The following appendix contains automatically extracted text from your project PDFs. "
        "Formatting may be imperfect; treat this as a searchable reference.\n\n"
    )

    for item in pdf_items:
        parts.append(f"\\subsection{{{latex_escape(item.title)}}}\n")
        parts.append(f"\\textit{{Source: {latex_escape(item.path)}}}\\\\\n")
        parts.append("\\begin{verbatim}\n")
        parts.append(item.text or "(No extractable text found.)")
        parts.append("\n\\end{verbatim}\n\n")

    return "".join(parts)


# -----------------------------
# LaTeX report builder
# -----------------------------

def build_report_tex(
    *,
    config: Dict[str, Any],
    results_text: str,
    narrative_tex: str = _DEFAULT_NARRATIVE,
    extracted_pdfs: Optional[List[ExtractedPDF]] = None,
) -> str:
    extracted_pdfs = extracted_pdfs or []

    now = datetime.now()
    title = config.get("report_title") or "LT / GP SDP Run Report"

    # Pretty-print config as JSON for the report
    cfg_json = json.dumps(config, indent=2, sort_keys=True, default=str)

    body_parts: List[str] = []

    body_parts.append(_latex_section("Run metadata", ""))
    body_parts.append("\\begin{itemize}\n")
    body_parts.append(f"\\item Generated: {latex_escape(now.strftime('%Y-%m-%d %H:%M:%S'))}\\n")
    body_parts.append(f"\\item Experiment ID: {latex_escape(str(config.get('selected_equation_id', ''))) }\\n")
    body_parts.append(f"\\item Mode: {latex_escape(str(config.get('module_type', ''))) }\\n")
    body_parts.append("\\end{itemize}\n\n")

    body_parts.append(_latex_verbatim_block("Configuration (raw)", cfg_json))

    body_parts.append(_latex_section("Project narrative summary", narrative_tex))

    results_text = results_text or "(No results_text provided.)"
    body_parts.append(_latex_verbatim_block("Run output / results", results_text))

    appendix = build_pdf_appendix(extracted_pdfs)
    if appendix:
        body_parts.append(appendix)

    body = "\n".join(body_parts)

    tex = rf"""
\\documentclass[11pt,a4paper]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{lmodern}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{hyperref}}
\\usepackage{{parskip}}

\\title{{{latex_escape(title)}}}
\\author{{Auto-generated by the SDP toolkit}}
\\date{{{latex_escape(now.strftime('%Y-%m-%d'))}}}

\\begin{{document}}
\\maketitle

{body}

\\end{{document}}
"""

    return tex.strip() + "\n"


# -----------------------------
# LaTeX compilation
# -----------------------------

class LaTeXCompileError(RuntimeError):
    pass


def compile_latex(tex_path: Path, *, workdir: Path, runs: int = 2) -> Path:
    """Compile a .tex file to PDF using pdflatex. Returns path to the PDF."""
    if shutil.which("pdflatex") is None:
        raise LaTeXCompileError("pdflatex not found on PATH. Install TeX Live or MiKTeX.")

    tex_path = tex_path.resolve()
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]

    for _ in range(max(1, runs)):
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            # Write log for debugging
            log_path = workdir / (tex_path.stem + ".pdflatex.log")
            log_path.write_text(proc.stdout, encoding="utf-8")
            raise LaTeXCompileError(
                f"LaTeX compilation failed (see {log_path}).\n\nLast output:\n" + proc.stdout[-2000:]
            )

    pdf_path = workdir / (tex_path.stem + ".pdf")
    if not pdf_path.exists():
        raise LaTeXCompileError("pdflatex finished but PDF not found.")
    return pdf_path


# -----------------------------
# Public API
# -----------------------------

_DEFAULT_PROJECT_PDFS = [
    "Capstone_Notes.pdf",
    "Capstone_Questions.pdf",
    "Capstone_Cheat_Sheet.pdf",
    "Questions_for_Alex's Capstone (3).pdf",
    "projectproposal.pdf",
    "PYU44TP1_Project_Guidelines_FINAL_2025-26.pdf",
]


def generate_pdf_report(
    *,
    config: Dict[str, Any],
    results_text: str,
    output_dir: str | Path = "reports",
    stem: Optional[str] = None,
    include_project_pdfs: bool = True,
    project_pdf_paths: Optional[Iterable[str | Path]] = None,
    max_pages_per_pdf: Optional[int] = 10,
) -> Path:
    """Create <stem>.tex and compile it to <stem>.pdf.

    - output_dir: directory where the .tex/.pdf are produced.
    - stem: filename stem (defaults to time-based).
    - include_project_pdfs: if True, append extracted text from PDFs.
    - project_pdf_paths: which PDFs to include; defaults to a curated list.
    - max_pages_per_pdf: avoid gigantic PDFs by limiting extraction per document.

    Returns the PDF path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = datetime.now().strftime("LTSDP_report_%Y%m%d_%H%M%S")

    extracted: List[ExtractedPDF] = []
    if include_project_pdfs:
        pdf_list = list(project_pdf_paths) if project_pdf_paths is not None else _DEFAULT_PROJECT_PDFS

        # Resolve PDF paths relative to the caller's working directory and also /mnt/data.
        candidates_root = [Path.cwd(), Path("/mnt/data"), output_dir]

        for p in pdf_list:
            p = Path(p)
            resolved = None
            if p.is_absolute() and p.exists():
                resolved = p
            else:
                for root in candidates_root:
                    q = (root / p)
                    if q.exists():
                        resolved = q
                        break

            if resolved is None:
                continue

            text = extract_pdf_text(resolved, max_pages=max_pages_per_pdf)
            extracted.append(ExtractedPDF(title=resolved.stem, path=str(resolved), text=text))

    tex = build_report_tex(config=config, results_text=results_text, extracted_pdfs=extracted)

    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path, workdir=output_dir)
    return pdf_path
