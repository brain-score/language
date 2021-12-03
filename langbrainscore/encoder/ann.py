
import typing
import IPython
import numpy as np
from langbrainscore.interface.encoder import _ANNEncoder
from collections import defaultdict
import string

from torch._C import Value


## Functions (probably to migrate later on)

def flatten_activations(activations):
    """
    Convert activations into dataframe format
    Input: dict, key = layer, item = 2D array of stimuli x units
    Output: pd dataframe, stimulus ID x MultiIndex (layer, unit)
    """
    labels = []
    arr_flat = []
    for layer, arr in activations.items():
        arr = np.array(arr) # for each layer
        arr_flat.append(arr)
        for i in range(arr.shape[1]): # across units
            labels.append((layer, i))
    arr_flat = np.concatenate(arr_flat, axis=1) # concatenated activations across layers
    df = pd.DataFrame(arr_flat)
    df.columns = pd.MultiIndex.from_tuples(labels) # rows: stimuli, columns: units
    return df

def get_activations(model, tokenizer, stimuli, emb_method='last-tok',
                    case='lower', punc='strip-all', punc_exceptions=[],
                    norm=None, outlier_removal=None,
                    verbose=True):
    """
    Obtain activations from (HF) models.
    
    :param model: model object, HF
    :paral tokenizer: tokenizer object, HF
    :param stimuli: list/array, [to do, edit] containing strings
    :param emb_method: str, denoting how to obtain sentence embedding
    :param case: str, denoting which casing to use
    :param punc: str, denoting which operations to perform regarding punctuation
    :param punc_exceptions: list, denoting which punctuation to NOT strip if punc == 'strip-all'
    :param norm: str, denoting how to normalize the embeddings 
    :param outlier_removel: str, denoting whether to remove 'outliers' from the embedding
        
    Returns dict with key = layer, item = 2D array of stimuli x units
    """

            
    return states_sentences


class ANNEmbeddingConfig:
    def __init__(self, aggregation: typing.Union['first', 'last', 'mean', 'median', None, typing.Callable] = 'last',
                 norm=None, outlier_removal=None) -> None:
        self.aggregation = aggregation
        self.norm = norm 
        self.outlier_removal = outlier_removal


class HuggingFaceEncoder(_ANNEncoder):
    _pretrained_model_name_or_path = None

    def __init__(self, pretrained_model_name_or_path) -> None:
        # super().__init__(self)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path

        from transformers import AutoModel, AutoConfig, AutoTokenizer
        self.config = AutoConfig.from_pretrained(self._pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(self._pretrained_model_name_or_path, config=self.config)

    def encode(self, dataset: 'langbrainscore.dataset.DataSet',
               word_level_args: ANNEmbeddingConfig = ANNEmbeddingConfig(aggregation=None),
               sentence_level_args: ANNEmbeddingConfig = ANNEmbeddingConfig(aggregation='last'),
               case: str = 'lower', punct: typing.Union[str, None, typing.List[str]] = None):

        self.model.eval() 
        n_layer = self.model.config.n_layer
        states_sentences = defaultdict(list)

        #### word-level
        for count, sentence in enumerate(dataset.stimuli.values):

            if case == 'lower':
                sentence = sentence.lower()
            elif case == 'upper':
                sentence = sentence.upper()

            tokenized: 'tokenizers.BatchEncoding' = self.tokenizer(sentence)
            word_ids = tokenized.word_ids()           
            input_ids = tokenized['']

            result_model = model(input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = result_model['hidden_states'] # 3D tensor of dims: [batch, tokens, emb size]
            
            for i in range(n_layer+1): # for each layer
                if emb_method == 'last-tok':
                    state = hidden_states[i].squeeze()[-1, :].detach().numpy() # get last token
                elif emb_method == 'mean-tok':
                    state = torch.mean(hidden_states[i].squeeze(), dim=0).detach().numpy() # mean over tokens
                elif emb_method == 'median-tok':
                    state = torch.median(hidden_states[i].squeeze(), dim=0).detach().numpy() # median over tokens
                elif emb_method == 'sum-tok':
                    state = torch.sum(hidden_states[i].squeeze(), dim=0).detach().numpy() # sum over tokens
                elif emb_method == 'all-tok':
                    raise NotImplementedError()
                else:
                    print('Sentence embedding method not implemented')
                    raise NotImplementedError()

                states_sentences[i].append(state)




class PTEncoder(_ANNEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid

    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> None:
        pass
