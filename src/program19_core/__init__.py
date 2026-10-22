"""Program19.3 2-D core: generated platelet-fibrin clot and its mechanics."""

from .config import Program19Config
from .mechanical_runner import run_pipeline
from .profiles import make_profiled_config

__all__ = ["Program19Config", "make_profiled_config", "run_pipeline"]
__version__ = "0.4.0"
