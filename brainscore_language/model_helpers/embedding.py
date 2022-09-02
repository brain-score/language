
class KeyedVectorModel(BrainModel, TaskModel):
    """
    Lookup-table-like models where each word has an embedding.
    To retrieve the sentence activation, we take the mean of the word embeddings.
    """

    available_layers = ['projection']
    default_layers = available_layers

    def __init__(self, identifier, weights_file, random_embeddings=False, random_std=1, binary=False):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        from gensim.models.keyedvectors import KeyedVectors
        self._model = KeyedVectors.load_word2vec_format(weights_file, binary=binary)
        self._vocab = self._model.vocab
        self._index2word_set = set(self._model.index2word)
        if random_embeddings:
            self._logger.debug(f"Replacing embeddings with random N(0, {random_std})")
            random_embedding = RandomState(0).randn(len(self._index2word_set), len(self._model['the'])) * random_std
            self._model = {word: random_embedding[i] for i, word in enumerate(sorted(self._index2word_set))}
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)

    def __call__(self, stimuli, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(stimuli, *args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            stimuli = " ".join(self._model.index2word[index] for index in stimuli)
            return self._encode_sentence(stimuli, *args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['projection'])
        encoding = [np.array(self._encode_sentence(sentence)) for sentence in sentences]
        # expand "batch" dimension for compatibility with transformers (for sentence-word-aggregation)
        encoding = [np.expand_dims(sentence_encodings, 0) for sentence_encodings in encoding]
        return {'projection': encoding}

    def _encode_sentence(self, sentence):
        words = sentence.split()
        feature_vectors = []
        for word in words:
            if word in self._index2word_set:
                feature_vectors.append(self._model[word])
            else:
                self._logger.warning(f"Word {word} not present in model")
                feature_vectors.append(np.zeros((300,)))
        return feature_vectors

    def tokenize(self, text, vocab_size=None):
        vocab_size = vocab_size or self.vocab_size
        tokens = [self._vocab[word].index for word in text.split() if word in self._vocab
                  and self._vocab[word].index < vocab_size]  # only top-k vocab words
        return np.array(tokens)

    def _sent_mean(self, sentence_features):
        sent_mean = np.mean(sentence_features, axis=0)  # average across words within a sentence
        return sent_mean

    def glue_dataset(self, examples, label_list, output_mode):
        import torch
        from torch.utils.data import TensorDataset
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        if examples[0].text_b is not None:
            text_a = [example.text_a for example in examples]
            text_b = [example.text_b for example in examples]
            sents1 = [self._sent_mean(self._encode_sentence(sent)) for sent in tqdm(text_a)]
            sents2 = [self._sent_mean(self._encode_sentence(sent)) for sent in tqdm(text_b)]
            for sent1, sent2 in zip(sents1, sents2):
                sent1 = torch.tensor(sent1, dtype=torch.float64)
                sent2 = torch.tensor(sent2, dtype=torch.float64)
                if np.isnan(sent1).all():
                    sent1 = torch.ones(sent2.shape, dtype=sent1.dtype)
                if np.isnan(sent2).all():
                    sent2 = torch.ones(sent1.shape, dtype=sent2.dtype)
                f = torch.cat([sent1, sent2, torch.abs(sent1 - sent2), sent1 * sent2], -1)
                features.append(PytorchWrapper._tensor_to_numpy(f))
            all_features = torch.tensor(features).float()
        else:
            text_a = [example.text_a for example in examples]
            sents = [self._sent_mean(self._encode_sentence(sent)) for sent in tqdm(text_a)]
            for sent in sents:
                sent = torch.tensor(sent)
                features.append(PytorchWrapper._tensor_to_numpy(sent))
            all_features = torch.tensor(features).float()

        if output_mode == "classification":
            all_labels = torch.tensor([label_map[example.label] for example in examples], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([float(example.label) for example in examples], dtype=torch.float)

        dataset = TensorDataset(all_features, all_labels)
        return dataset

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def features_size(self):
        return 300
