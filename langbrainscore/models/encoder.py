from abc import ABC

import numpy as np

class Encoder(ABC):

    def __init__(self) -> None:
        pass

    def encode(self, X: np.array) -> np.array:
        NotImplemented