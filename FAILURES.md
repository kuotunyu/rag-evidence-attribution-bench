# FAILURES

Observed failures, compatibility dead-ends, and the fallbacks adopted.

**Entry rules:** every entry must include an observed reproduction (exact command + error
output or run id). Anticipated-but-unobserved risks do not belong here — no speculation.

Format:

```
## YYYY-MM-DD <short title>
- Command: <exact command>
- Observed: <error / behavior, trimmed real output>
- Time spent: <hours> (stop-loss: <hours>)
- Fallback adopted: <what we did instead>
```

---

## 2026-07-18 torch ≥2.7 CPU builds fail to import on the Win10 dev machine

- Command: `uv run python -c "import torch"` (torch 2.13.0+cpu and 2.9.1+cpu, Python 3.11.7,
  Windows 10 Home 19045, i7-8750H, VC++ runtime 14.51)
- Observed: `OSError: [WinError 1114] 動態連結程式庫 (DLL) 初始化例行程序失敗 Error loading
  ".venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.` Reproduced with
  minimal PATH (`C:\Windows\System32;C:\Windows`), so not a PATH/DLL conflict; AVX2 present,
  32 GB RAM, current VC++ redistributable — root cause not identified. torch 2.6.0+cpu
  imports and runs fine on the same machine.
- Time spent: ~0.5 h (stop-loss: kept short by design — version ladder instead of root-causing)
- Fallback adopted: platform-scoped pin in pyproject `ml` extra —
  `torch>=2.4,<2.7; sys_platform == 'win32'` / `torch>=2.4; sys_platform != 'win32'`.
  Linux (CI/Docker/Colab) is unaffected and uses current torch; Colab keeps its
  preinstalled CUDA torch. Local Windows runs use torch 2.6.0+cpu for CPU-only work
  (dense retrieval, tests). GPU benchmark numbers are produced on Colab regardless.
