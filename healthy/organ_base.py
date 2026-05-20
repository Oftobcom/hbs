# organ_base.py
from abc import ABC, abstractmethod
import numpy as np

class OrganModel(ABC):
    """Base interface for all organ models."""

    @abstractmethod
    def get_state_size(self) -> int:
        """Number of state variables of the organ."""
        pass

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Initial state vector."""
        pass

    @abstractmethod
    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: dict) -> np.ndarray:
        """Compute derivatives of the organ's state."""
        pass

    @abstractmethod
    def get_outputs(self, state_slice: np.ndarray) -> dict:
        """Compute output variables for coupling."""
        pass