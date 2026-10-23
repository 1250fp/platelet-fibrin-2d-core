"""Program19.3 2-D core: generated platelet-fibrin clot and its mechanics."""

from .config import Program19Config
from .mechanical_runner import run_pipeline
from .state import SimulationState

__all__ = ["Program19Config", "SimulationState", "run_pipeline"]
__version__ = "0.6.0"
