import pickle
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from brainio.assemblies import DataAssembly
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from brainscore_language import (load_benchmark, load_dataset, load_metric,
                                 load_model)
from brainscore_language.artificial_subject import ArtificialSubject


class HuggingfaceToolbox:
    """
    A toolbox for lower-level interaction with Huggingface models and the Brainscore Language interface.
    """

    load_metric = load_metric
    load_benchmark = load_benchmark

    def load_model(self, identifier: str) -> ArtificialSubject:
        """
        Load a model given its identifier.

        :param identifier: The model's identifier.
        :return: The loaded model.
        """
        self.model = load_model(identifier)
        return self.model

    def load_dataset(
        self, identifier: str, **selection_kwargs
    ) -> Union[DataAssembly, Any]:
        """
        Load a dataset given its identifier and select specific parts if provided.

        :param identifier: The dataset's identifier.
        :param selection_kwargs: Keyword arguments for selection.
        :return: The loaded dataset.
        """
        self.dataset = load_dataset(identifier)
        if selection_kwargs:
            self.dataset = self.dataset.sel(**selection_kwargs)
        self.dataset = self.dataset.dropna('neuroid')
        return self.dataset

    def _aggregate_embeddings_and_logits(self, method, logits, hidden_states):
        # Post processing to aggregate embeddings
        if method == "l2r":
            logits = logits[-1]
            hidden_states = [hs[-1] for hs in hidden_states]
        else:
            agg_fn = torch.mean if method == "mean" else torch.sum
            hidden_states = torch.stack(
                [
                    agg_fn(
                        torch.stack(
                            [
                                hidden_states[t][hs_idx].squeeze()
                                for t in range(len(hidden_states))
                            ]
                        ),
                        axis=0,
                    )
                    for hs_idx in range(len(hidden_states[0]))
                ],
                axis=0,
            )
            logits = agg_fn(torch.cat(logits), axis=0)
        return logits, hidden_states
        
    def embed(
        self,
        text: Union[str, Iterable[str]],
        layers: Union[str, List[int]] = "all",
        by: str = "token",
        method: str = "l2r",
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Embed the given text using the loaded model.

        :param text: The text to embed.
        :param layers: Which layers to extract embeddings from. Defaults to "all".
        :param by: Granularity of embedding ("token" or "sentence"). Defaults to "token".
        :param method: Embedding method ("l2r", "sum", or "mean"). Defaults to "l2r".
        :return: A tuple of logits and embeddings by layer.
        """

        # Parameter validation
        assert isinstance(text, (str, Iterable)), "Cannot score a non-string by tokens"
        assert by in {"token", "sentence"}, "Can only embed by `sentence` or `token`"
        assert method in {
            "l2r",
            "sum",
            "mean",
        }, "Can only aggregate using `l2r`, `mean`, or `sum`"

        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        # Tokenize the text
        text_tokens = [
            self.model._tokenize(context, 0)[0]["input_ids"].squeeze()
            for context in text
        ]
        if by == "token":
            text_tokens = torch.cat(text_tokens)
            
        # Prepare for embedding
        self.encoding = []
        hidden_states, logits = [], []
        text_iterator = tqdm(range(len(text_tokens)), desc="embedding text")
        context_window = model.basemodel.config.max_position_embeddings
        context_size = 0
        sentence_l2r_last_idx = 0

        context_window = model.basemodel.config.max_position_embeddings
        context_size = 0
        sentence_l2r_last_idx = 0
        for part_number in text_iterator:
            with torch.no_grad():
                if method == "l2r":
                    if by == "token":
                        text_part = text_tokens[
                            part_number - context_size : part_number + 1
                        ]
                        context_size = min(context_size + 1, context_window - 1)
                    elif by == "sentence":
                        while (context_size >= context_window) and (
                            sentence_l2r_last_idx < part_number
                        ):
                            context_size = sum(
                                [
                                    len(sent)
                                    for sent in text_tokens[
                                        sentence_l2r_last_idx : part_number + 1
                                    ]
                                ]
                            )
                            sentence_l2r_last_idx += 1
                        text_part = torch.cat(
                            text_tokens[sentence_l2r_last_idx : part_number + 1]
                        )[-context_window:]
                else:
                    text_part = text_tokens[part_number : part_number + 1]

                base_output = self.model.basemodel(text_part)

                if method == "l2r":
                    hidden_states, logits = (
                        base_output.hidden_states,
                        base_output.logits,
                    )
                else:
                    hidden_states.append(base_output.hidden_states)
                    logits.append(base_output.logits)
            
            agg_logits, agg_embs = self._aggregate_embeddings_and_logits(method, logits, hidden_states)
            self.encoding.append(
                {
                    "stimulus_tokens": text_tokens[part_number].cpu().numpy(),
                    "context_tokens": text_part.cpu().numpy(),
                    "part_number": part_number,
                    "hidden_states": [layer_agg_embeddings.cpu().numpy() for layer_agg_embeddings in agg_embs],
                    "logits": agg_logits.cpu().numpy(),
                }
            )
        
        # Get embeddings from required layers
        logits, hidden_states = self._aggregate_embeddings_and_logits(method, logits, hidden_states)
        self.logits = logits.cpu().numpy()
        
        if layers == "all":
            layers = np.arange(len(hidden_states))
        self.embeddings = {
            layer: hidden_states[layer].cpu().numpy() for layer in layers
        }

        self.encoding = pd.DataFrame.from_dict(self.encoding)
        return self.logits, self.embeddings

    def score(self, text, *args, **kwargs):
        """
        Score the given text using the loaded model.

        :param text: The text to score.
        :param args: Additional arguments for embedding.
        :param kwargs: Additional keyword arguments for embedding.
        :return: logits.
        """
        logits, _ = self.embed(text, layers=[], *args, **kwargs)
        return logits

    def preprocess_embeddings(
        self, preprocessor: Callable, layers: Optional[List[int]] = None
    ) -> None:
        """
        Preprocess embeddings using a given preprocessor function.

        :param preprocessor: The preprocessing function.
        :param layers: Layers to preprocess. Defaults to all layers.
        """
        if layers is None:
            layers = np.arange(len(self.embeddings))

        if isinstance(layers, int):
            layers = [layers]

        for layer in layers:
            embeddings_layer = self.embeddings[layer]
            batched_preprocessor = lambda x: [preprocessor(i) for i in x]
            self.encoding["hidden_states"].apply(batched_preprocessor)
            self.embeddings[layer] = preprocessor(embeddings_layer)

    def map_embeddings(self, mapping: Callable, layer: int) -> np.ndarray:
        dataset_values = self.dataset.values
        n_samples = len(dataset_values)
        embedding_values = np.stack([self.encoding.hidden_states[i][layer] for i in range(n_samples)])
        mapping = mapping.fit(dataset_values, embedding_values)
        preds = mapping.predict(dataset_values)
        return preds

    def evaluate_metric(self, metric: Callable, preds: np.ndarray, reduce: Optional[Callable] = None):
        dataset_values = self.dataset.values
        score = metric(dataset_values, preds)
        return reduce(score) if reduce else score

    @staticmethod
    def load_precomputed_embeddings(path: str) -> Dict[int, np.ndarray]:
        """
        Load precomputed embeddings from a file.

        :param path: Path to the embeddings file.
        :return: Loaded embeddings by layer.
        """
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    @staticmethod
    def save_embeddings(path: str, embeddings: Dict[int, np.ndarray]) -> None:
        """
        Save embeddings to a file.

        :param path: Path to save the embeddings.
        :param embeddings: Embeddings by layer to save.
        """
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    tbx = HuggingfaceToolbox()
    model = tbx.load_model("distilgpt2")
    dataset = tbx.load_dataset("Pereira2018.language", experiment="243sentences")

    logprobs, embeddings = tbx.embed(dataset.stimulus.values, by="sentence", method="l2r")

    # Preprocess the embeddings into a common space
    from scipy.stats import zscore
    tbx.preprocess_embeddings(preprocessor=zscore, layers=[6])

    # Map model predictions into dataset space
    from sklearn.linear_model import RidgeCV
    mapping = RidgeCV()
    preds = tbx.map_embeddings(mapping=mapping, layer=6)
    
    # Run metric on dataset and regression predictions
    from scipy.stats import spearmanr
    reduce = lambda score: np.mean(score.statistic)
    score = tbx.evaluate_metric(metric=spearmanr, preds=preds, reduce=reduce)
