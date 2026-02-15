# organ_base.py
from abc import ABC, abstractmethod
import numpy as np

class OrganModel(ABC):
    """Базовый интерфейс для всех моделей органов."""

    @abstractmethod
    def get_state_size(self) -> int:
        """Возвращает количество переменных состояния органа."""
        pass

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Начальное состояние органа."""
        pass

    @abstractmethod
    def get_derivatives(self, t: float, state_slice: np.ndarray,
                        inputs: dict) -> np.ndarray:
        """
        Вычисляет производные состояния органа.
        :param t: текущее время
        :param state_slice: часть общего вектора, относящаяся к данному органу
        :param inputs: словарь с внешними воздействиями (давления, потоки)
        :return: производные state_slice
        """
        pass

    @abstractmethod
    def get_outputs(self, state_slice: np.ndarray) -> dict:
        """
        Возвращает выходные переменные органа (потоки, давления и т.д.)
        для использования другими органами.
        """
        pass