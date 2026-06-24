"""Output-directory helpers for experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunOutputDirs:
    """Standard artifact directories for one experiment run."""

    root: Path
    tables: Path
    figures: Path
    models: Path


def run_output_root(output_dir: str | Path, run_name: str | None = None) -> Path:
    """Resolve the root artifact directory for an optional named run."""

    root = Path(output_dir)
    if run_name is None or run_name == "":
        return root
    cleaned = run_name.strip()
    if cleaned != run_name or cleaned in {".", ".."} or Path(cleaned).name != cleaned:
        raise ValueError("run_name must be a single directory name without path separators.")
    return root / "runs" / cleaned


def make_run_output_dirs(output_dir: str | Path, run_name: str | None = None) -> RunOutputDirs:
    """Create standard artifact directories and return their paths."""

    root = run_output_root(output_dir, run_name)
    dirs = RunOutputDirs(
        root=root,
        tables=root / "tables",
        figures=root / "figures",
        models=root / "models",
    )
    dirs.tables.mkdir(parents=True, exist_ok=True)
    dirs.figures.mkdir(parents=True, exist_ok=True)
    dirs.models.mkdir(parents=True, exist_ok=True)
    return dirs


def display_path(path: str | Path, root: str | Path) -> Path:
    """Return a root-relative path when possible."""

    path = Path(path)
    try:
        return path.relative_to(Path(root))
    except ValueError:
        return path
