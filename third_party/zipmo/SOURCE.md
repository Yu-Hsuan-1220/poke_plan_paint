# ZipMo (vendored)

Copied unchanged from https://github.com/CompVis/long-term-motion (local checkout `/project/YuHsuanLee/long-term-motion`, 2026-11-17).
License: MIT, Copyright 2025 Nick Stracke et al., CompVis @ LMU Munich (SPDX headers in each file).

Only the model code (`zipmo/`, `hubconf.py`) is used. Weights come from HuggingFace `CompVis/ZipMo`,
loaded by `src/common.py` (`load_planner`, `load_zipmo_vae`) from `$HF_CACHE`.
