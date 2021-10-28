
import numpy as np
from langbrainscore.interface.encoder import ANNEncoder

class HuggingFaceEncoder(ANNEncoder):
    _model_name_or_path = None

    def __init__(self, pretrained_model_name_or_path) -> None:
        super().__init__(self)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path

        from transformers import AutoModel, AutoConfig, AutoTokenizer
        self.config = AutoConfig.from_pretrained(self._pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(self._pretrained_model_name_or_path, config=self.config)

    def encode(self, X: np.array) -> np.array:
        pass




class PTEncoder(ANNEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid

    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> pd.DataFrame:
        pass
