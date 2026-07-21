"""Sampler / scheduler aliases for ``--sampler``.

Maps a short, ComfyUI/A1111-flavoured sampler name to a diffusers scheduler
class (by name) plus any config overrides. The chosen scheduler is applied with
``Scheduler.from_config(pipe.scheduler.config, **extra)`` so it inherits the
model's own scheduler configuration (see :func:`imggen.pipelines.common.set_scheduler`).

ComfyUI splits the choice into two dropdowns — the *sampler* (``euler``,
``dpmpp_2m``, ``uni_pc`` ...) and the *scheduler* / sigma schedule (``normal``,
``karras``, ``exponential``, ``beta`` ...). diffusers folds both into a single
scheduler class: the sampler picks the class + algorithm, and the sigma schedule
is a boolean flag (``use_karras_sigmas`` / ``use_exponential_sigmas`` /
``use_beta_sigmas``) on the classes that support it. We mirror that here by
generating ``<sampler>_karras`` / ``_exponential`` / ``_beta`` variants for every
sigma-capable base sampler, so e.g. ``dpm++_2m_karras`` and ``euler_beta`` both
resolve.

Note on flow-matching models: Qwen-Image and SD3.5 use a *flow-matching*
scheduler (``FlowMatchEulerDiscreteScheduler``). Classic samplers below
(``euler``, ``ddim``, ``dpm++_2m`` ...) are for the epsilon/v-prediction models
(SD1.5 / SDXL) and generally do **not** transfer to flow models — use
``flowmatch`` there. ``set_scheduler`` degrades gracefully with a warning when a
sampler is incompatible with the loaded pipeline.
"""

from __future__ import annotations

# Base sampler name -> (diffusers scheduler class name, algorithm kwargs).
# One entry per ComfyUI "sampler" that has a diffusers equivalent. The
# karras/exponential/beta sigma variants are generated below (``_build``).
_BASE: dict[str, tuple[str, dict]] = {
    # --- epsilon / v-prediction samplers (SD1.5 / SDXL) ---
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "dpm_2": ("KDPM2DiscreteScheduler", {}),
    "dpm_2_a": ("KDPM2AncestralDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "dpm++_2s_a": ("DPMSolverSinglestepScheduler", {}),
    "dpm++_sde": ("DPMSolverSDEScheduler", {}),
    "dpm++_2m": ("DPMSolverMultistepScheduler", {}),
    "dpm++_2m_sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "dpm++_3m_sde": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
    ),
    "unipc": ("UniPCMultistepScheduler", {}),
    "unipc_bh2": ("UniPCMultistepScheduler", {"solver_type": "bh2"}),
    "deis": ("DEISMultistepScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "ddpm": ("DDPMScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
    "ipndm": ("IPNDMScheduler", {}),
    # --- distilled / few-step samplers ---
    "lcm": ("LCMScheduler", {}),
    "tcd": ("TCDScheduler", {}),
    # --- flow-matching samplers (Qwen-Image / SD3.5) ---
    "flowmatch": ("FlowMatchEulerDiscreteScheduler", {}),
    "flowmatch_heun": ("FlowMatchHeunDiscreteScheduler", {}),
}

# Sigma-schedule suffix -> the diffusers from_config flag it sets. ComfyUI's
# separate "scheduler" dropdown (karras / exponential / beta) maps onto these.
_SIGMA_SUFFIX: dict[str, str] = {
    "karras": "use_karras_sigmas",
    "exponential": "use_exponential_sigmas",
    "beta": "use_beta_sigmas",
}

# Base scheduler classes that accept the sigma-schedule flags above.
_SIGMA_CAPABLE: set[str] = {
    "EulerDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "UniPCMultistepScheduler",
}


def _build() -> dict[str, tuple[str, dict]]:
    """Expand the base table with the karras/exponential/beta sigma variants."""
    table = dict(_BASE)
    for name, (cls, extra) in _BASE.items():
        if cls not in _SIGMA_CAPABLE:
            continue
        for suffix, kw in _SIGMA_SUFFIX.items():
            table[f"{name}_{suffix}"] = (cls, {**extra, kw: True})
    # DPM++ SDE Karras is a staple; the SDE class takes the flag via **kwargs.
    table["dpm++_sde_karras"] = ("DPMSolverSDEScheduler", {"use_karras_sigmas": True})
    return table


# name -> (diffusers scheduler class name, extra from_config kwargs)
_ALIASES: dict[str, tuple[str, dict]] = _build()

# Common spellings folded onto the canonical alias (applied after ``dpmpp`` ->
# ``dpm++``, ``-`` -> ``_`` and ``uni_pc`` -> ``unipc`` normalisation).
_SYNONYMS = {
    "eulera": "euler_a",
    "euler_ancestral": "euler_a",
    "k_euler": "euler",
    "k_euler_a": "euler_a",
    "k_euler_ancestral": "euler_a",
    "dpm2": "dpm_2",
    "dpm_2_ancestral": "dpm_2_a",
    "dpm++_2s_ancestral": "dpm++_2s_a",
    "dpm++_2m_gpu": "dpm++_2m",
    "dpm++_2m_sde_gpu": "dpm++_2m_sde",
    "dpm++_3m_sde_gpu": "dpm++_3m_sde",
    "dpm++_sde_gpu": "dpm++_sde",
    "flow": "flowmatch",
    "flow_match": "flowmatch",
    "flow_match_euler": "flowmatch",
    "flow_match_heun": "flowmatch_heun",
}

# Canonical names offered in ``--sampler`` help / ``imggen samplers``.
NAMES: list[str] = list(_ALIASES.keys())


def normalize(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = key.replace("dpmpp", "dpm++")
    key = key.replace("uni_pc", "unipc")
    return _SYNONYMS.get(key, key)


def resolve(name: str) -> tuple[str, dict] | None:
    """Return ``(scheduler_class_name, extra_kwargs)`` for ``name`` or ``None``."""
    return _ALIASES.get(normalize(name))
