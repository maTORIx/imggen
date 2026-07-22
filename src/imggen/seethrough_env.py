"""Provisioning for the isolated See-through environment.

``see-through --method decompose`` cannot run in imggen's own interpreter: the
See-through stack (Lin et al., SIGGRAPH 2026) pins ``diffusers==0.37`` /
``transformers==5.0`` and deep-subclasses their internals, while imggen tracks
current diffusers. So it gets its own checkout and its own venv, driven as a
subprocess.

That is an implementation detail the user should never have to build by hand —
imggen already downloads multi-gigabyte model weights on first use, and this is
the same kind of first-run cost. :func:`ensure` does the whole thing (clone,
venv, dependencies) and is called automatically by the decompose backend;
``imggen setup see-through`` is the same code path, exposed so it can be warmed
up ahead of time or re-run after a failure.

Layout (under ``$IMGGEN_CACHE`` / ``~/.cache/imggen``)::

    seethrough/repo    the See-through checkout
    seethrough/venv    its interpreter + pinned deps

``$IMGGEN_SEETHROUGH_REPO`` / ``$IMGGEN_SEETHROUGH_PYTHON`` override either half,
for people who already have a checkout or want to share one between hosts.

Stdlib only, and not imported from :mod:`imggen.cli` at module level — the setup
command imports it lazily, same rule as the rest of the light command path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Upstream checkout. Apache-2.0.
REPO_URL = "https://github.com/shitagaki-lab/see-through"

#: Wheel index for torch. The See-through venv needs a build matching the host
#: GPU exactly as imggen's own does (see ``pyproject.toml``), so it defaults to
#: the same CUDA 13.0 index; point ``$IMGGEN_TORCH_INDEX`` elsewhere (e.g.
#: ``.../whl/cpu``, ``.../whl/cu132``) on a different stack.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

#: See-through's pins (transformers 5 / numpy 2.2) resolve cleanly here.
PYTHON_VERSION = "3.12"


class SetupError(RuntimeError):
    """The isolated environment could not be provisioned."""


# --- paths --------------------------------------------------------------

def cache_home() -> Path:
    root = os.environ.get("IMGGEN_CACHE")
    return (Path(root) if root else Path.home() / ".cache" / "imggen") / "seethrough"


def repo_dir() -> Path:
    return Path(os.environ.get("IMGGEN_SEETHROUGH_REPO") or cache_home() / "repo")


def python_exe() -> Path:
    return Path(os.environ.get("IMGGEN_SEETHROUGH_PYTHON") or cache_home() / "venv" / "bin" / "python")


def entry_script(repo: Path | None = None) -> Path:
    return (repo or repo_dir()) / "inference" / "scripts" / "inference_psd.py"


def is_ready() -> bool:
    """True when both halves of the isolated install are present."""
    return entry_script().exists() and python_exe().exists()


def requirements_file() -> Path:
    """The pinned inference requirements shipped inside the package."""
    return Path(__file__).parent / "data" / "seethrough-requirements.txt"


# --- process helpers ----------------------------------------------------

def _echo(msg: str, color: str | None = None) -> None:
    import typer

    typer.secho(msg, fg=color)


def _run(cmd: list[str], step: str, cwd: Path | None = None) -> None:
    """Run *cmd*, letting its output through, and raise :class:`SetupError` if it fails."""
    _echo(f"  $ {' '.join(cmd)}", "bright_black")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SetupError(f"{step} failed (exit {proc.returncode}); see the output above")


def _uv() -> str | None:
    """Path to ``uv``, if it is on PATH."""
    return shutil.which("uv")


# --- steps --------------------------------------------------------------

def _ensure_repo(repo: Path, force: bool) -> None:
    if entry_script(repo).exists() and not force:
        _echo(f"  checkout ok: {repo}", "bright_black")
    else:
        if not shutil.which("git"):
            raise SetupError("git is required to fetch the See-through checkout, but is not on PATH")
        if repo.exists() and force:
            shutil.rmtree(repo)
        repo.parent.mkdir(parents=True, exist_ok=True)
        _echo(f"fetching See-through into {repo} ...", "cyan")
        _run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], "git clone")
        if not entry_script(repo).exists():
            raise SetupError(f"clone succeeded but {entry_script(repo)} is missing")
    # The inference code resolves its bundled configs/weights as ``assets/...``
    # relative to the repo root, but they live in ``common/assets``. Upstream's
    # README tells you to make this link by hand; do it here instead.
    link = repo / "assets"
    if not link.exists():
        link.symlink_to(Path("common") / "assets")


def _ensure_venv(py: Path, force: bool) -> None:
    if py.exists() and not force:
        _echo(f"  interpreter ok: {py}", "bright_black")
        return
    venv = py.parent.parent
    if venv.exists() and force:
        shutil.rmtree(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    _echo(f"creating the See-through venv at {venv} ...", "cyan")
    uv = _uv()
    if uv:
        _run([uv, "venv", "--python", PYTHON_VERSION, str(venv)], "uv venv")
    else:
        # No uv: fall back to imggen's own interpreter. Works as long as it is
        # new enough for See-through's pins.
        _run([sys.executable, "-m", "venv", str(venv)], "python -m venv")
    if not py.exists():
        raise SetupError(f"venv created but {py} is missing")


def _install(py: Path, repo: Path) -> None:
    uv = _uv()
    torch_index = os.environ.get("IMGGEN_TORCH_INDEX", DEFAULT_TORCH_INDEX)
    reqs = requirements_file()
    if not reqs.is_file():  # a broken install of imggen itself
        raise SetupError(f"packaged requirements missing: {reqs}")

    if uv:
        base = [uv, "pip", "install", "--python", str(py)]
    else:
        base = [str(py), "-m", "pip", "install"]
    _echo("installing torch (CUDA wheels; cached after the first host) ...", "cyan")
    _run([*base, "--index-url", torch_index, "torch", "torchvision"], "torch install")
    _echo("installing See-through inference dependencies ...", "cyan")
    _run([*base, "-r", str(reqs)], "dependency install")
    # The checkout's own package (``live2d-common``), editable so the inference
    # scripts import the code we just cloned rather than a copy.
    _echo("installing the See-through package ...", "cyan")
    _run([*base, "-e", str(repo / "common")], "See-through package install")


# --- entry point --------------------------------------------------------

def ensure(force: bool = False) -> tuple[Path, Path]:
    """Make the isolated See-through environment usable; return ``(repo, python)``.

    Idempotent: each step is skipped when already satisfied, so calling this on
    every decompose run costs two ``stat`` calls once provisioned. With *force*
    the checkout and venv are removed and rebuilt.

    Raises :class:`SetupError` if any step fails.
    """
    repo, py = repo_dir(), python_exe()
    if not force and is_ready():
        return repo, py

    explicit_py = os.environ.get("IMGGEN_SEETHROUGH_PYTHON")
    if explicit_py and not py.exists():
        # Don't build a venv at a path the user pointed somewhere deliberately —
        # they meant "use this interpreter", and it isn't there.
        raise SetupError(
            f"$IMGGEN_SEETHROUGH_PYTHON points at {py}, which does not exist.\n"
            "Unset it to let imggen provision its own environment."
        )

    _echo("setting up the isolated See-through environment (first run only)", "cyan")
    _echo(f"  repo:   {repo}", "bright_black")
    _echo(f"  python: {py}", "bright_black")
    _ensure_repo(repo, force)
    _ensure_venv(py, force)
    _install(py, repo)
    if not is_ready():
        raise SetupError("setup finished but the environment is still incomplete")
    _echo("See-through environment ready", "green")
    return repo, py


def describe() -> dict:
    """Current state of the environment, for ``imggen setup see-through --status``."""
    repo, py = repo_dir(), python_exe()
    return {
        "repo": str(repo),
        "repo_ok": entry_script(repo).exists(),
        "python": str(py),
        "python_ok": py.exists(),
        "ready": is_ready(),
    }
