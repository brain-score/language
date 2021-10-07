from abc import ABC, abstractmethod
import typing

import numpy as np


class Encoder(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod()
    def encode(self, X: np.array) -> np.array:
        raise NotImplementedError()


class BrainEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass


class SilicoEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass
