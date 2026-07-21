"""Vendored subset of sd_embed (https://github.com/xhinker/sd_embed, Apache-2.0).

Provides long-prompt / weighted-embedding helpers for the CLIP-based Stable
Diffusion families (SD1.5, SDXL, SD3). Only ``embedding_funcs`` and the
``parse_prompt_attention`` parser it needs are vendored; the upstream package's
heavier extras (Flux/Cascade paths, ``lark``-based prompt scheduling,
``optimum-quanto``) are intentionally left out. See ``LICENSE`` for terms.
"""
