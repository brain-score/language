from abc import ABC, abstractmethod

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


class CompEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass


class Pereira18Encoder(BrainEncoder):
    def __init__(self, bmid) -> None:
        super().__init__(self)
        self._bmid = bmid

    def encode(self, X: np.array) -> np.array:
        pass


class HFEncoder(CompEncoder):
    def __init__(self, hfid) -> None:
        super().__init__(self)
        self._hfid = hfid

    def encode(self, X: np.array) -> np.array:
        pass


class PTEncoder(CompEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid

    def encode(self, X: np.array) -> np.array:
        pass
