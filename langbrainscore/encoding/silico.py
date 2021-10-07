
from langbrainscore.interface.encoder import SilicoEncoder



class HFEncoder(SilicoEncoder):
    def __init__(self, model_name_or_path) -> None:
        super().__init__(self)
        self.model_name_or_path = model_name_or_path

        from transformers import AutoModel, AutoConfig, AutoTokenizer


    def encode(self, X: np.array) -> np.array:
        pass


class PTEncoder(SilicoEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid

    def encode(self, X: np.array) -> np.array:
        pass
