"""macro_gpu_lab — GPU research framework over the HQ macro brain.

Research / advisory only. Every model stays SHADOW until it clears the overfit
gate (PBO < 0.5 AND Deflated Sharpe > 0 on every horizon). Output is a fail-safe
flag-file; nothing here touches a broker or a live hot path.
"""

__version__ = "0.1.0"
