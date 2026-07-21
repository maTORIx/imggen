"""Device and dtype selection.

``torch`` is imported lazily inside each function (not at module load) so that
the lightweight commands — ``version``, ``models``, ``pull``, ``samplers`` and
shell completion — stay fast and don't pay torch's ~0.6s import cost.
"""

from __future__ import annotations


def resolve_device(pref: str | None = None) -> str:
    """Pick a device, preferring CUDA > MPS > CPU unless ``pref`` is given."""
    if pref:
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, pref: str | None = None) -> "torch.dtype":
    """Pick a compute dtype appropriate for the device."""
    import torch

    if pref:
        return {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[pref.lower()]
    if device == "cuda":
        return torch.bfloat16  # Blackwell has strong bf16 support
    if device == "mps":
        return torch.float16
    return torch.float32


def describe() -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return f"cuda ({name}, sm_{cap[0]}{cap[1]}, torch {torch.__version__})"
    return f"cpu (torch {torch.__version__})"
