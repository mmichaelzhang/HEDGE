"""Legacy HEDGE-DTI demonstration API; see reference/ for the later core.

The sampler and the data interface are pure NumPy and import without torch;
the encoder and losses require torch (see requirements.txt).
"""
from .sampling import build_typed_graph, sample_walks
from .data import load_fold, check_no_leakage

__all__ = ["build_typed_graph", "sample_walks", "load_fold", "check_no_leakage"]
__version__ = "0.1.0"

try:  # torch-dependent components
    from .model import HedgeDTI
    from . import losses
    __all__ += ["HedgeDTI", "losses"]
except ImportError:  # pragma: no cover - torch not installed
    pass
