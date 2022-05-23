from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

preprocessor_classes = {
    'demean': StandardScaler(with_std=False),
    'demean_std': StandardScaler(with_std=True),
    'minmax': MinMaxScaler,
    'discretize': KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'),
    'robust_scaler': RobustScaler(),
    'pca': PCA(n_components=10),
    'gaussian_random_projection': GaussianRandomProjection(n_components=10)
}

model_classes = ['gpt', 'bert']  # continuously update based on new model classes supported
config_name_mappings = {
                        "gpt": {
                            "n_layers": "n_layer",
                            "n_attention_heads": "n_head",
                            "n_context_len": "n_ctx",
                            "vocab_size": "vocab_size",
                            "hidden_emb_dim": "n_embd",
                            "hidden_activation_function": "activation_function"},
                        
                        "bert": {
                            "n_layers": "num_hidden_layers",
                            "n_attention_heads": "num_attention_heads",
                            "n_context_len": "max_position_embeddings",
                            "vocab_size": "vocab_size",
                            "hidden_emb_dim": "hidden_size",
                            "hidden_activation_function": "hidden_act"}
                        }
                        