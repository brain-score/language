import IPython
from pathlib import Path

import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import langbrainscore as lbs
from langbrainscore.utils.logging import log
from langbrainscore.utils.xarray import collapse_multidim_coord
from langbrainscore.benchmarks import pereira2018

# import pereira2018_mean_froi_nat_stories

os.environ["VERBOSE"] = "1"


def main():
    mpf_xr = pereira2018.pereira2018_mean_froi_nat_stories()
    mpf_dataset = lbs.dataset.Dataset(
        mpf_xr.isel(neuroid=mpf_xr.roi.str.contains("Lang")),
        dataset_name="Pereira2018LangfROIs",
    )

    log(f"stimuli: {mpf_dataset.stimuli.values}")
    mpf_dataset.to_cache("test_mpf_dataset_cache", cache_dir="./cache")
    mpf_dataset = lbs.dataset.Dataset.from_cache(
        "test_mpf_dataset_cache", cache_dir="./cache"
    )
    log(f"stimuli: {mpf_dataset.stimuli.values}")

    # Initialize brain and ANN encoders
    brain_enc = lbs.encoder.BrainEncoder()
    ann_enc = lbs.encoder.HuggingFaceEncoder(
        model_id="bert-base-uncased",
        emb_preproc=tuple(),
        context_dimension="passage",
        bidirectional=True,
        emb_aggregation="first",
        # model_id="distilgpt2", emb_preproc=tuple(), context_dimension="passage"
    )

    # Encode
    brain_enc_mpf = brain_enc.encode(mpf_dataset)
    ann_enc_mpf = ann_enc.encode(mpf_dataset).representations
    log(f"created brain-encoded data of shape: {brain_enc_mpf.shape}")
    log(f"created ann-encoded data of shape: {ann_enc_mpf.shape}")

    # ANN encoder checks
    ann_enc_check = lbs.encoder.EncoderCheck()
    ann_enc_check.similiarity_metric_across_layers(
        sim_metric="tol", enc1=ann_enc_mpf, enc2=ann_enc_mpf
    )

    # Model card
    ann_modelcard = ann_enc.get_modelcard()
    ann_enc.get_layer_sparsity(ann_encoded_dataset=ann_enc_mpf)
    ann_enc.get_explainable_variance(ann_encoded_dataset=ann_enc_mpf)

    # Initialize mapping and metric
    ann_enc_mpf = ann_enc_mpf.isel(
        neuroid=(ann_enc_mpf.layer == 4)
    )  # Select a layer # TODO: loop over layers unless it is a brain model with commitment

    rdg_cv_kfold = lbs.mapping.LearnedMap("linridge_cv", k_fold=5)
    fisher = lbs.metrics.Metric(lbs.metrics.FisherCorr)
    brsc_rdg_corr = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, rdg_cv_kfold, fisher)
    brsc_rdg_corr.run(sample_split_coord="experiment")
    log(f"brainscore (rdg, fisher) = {brsc_rdg_corr.scores.mean()}")
    log(f"ceiling (rdg, fisher) = {brsc_rdg_corr.ceilings.mean()}")

    i_map = lbs.mapping.IdentityMap(nan_strategy="drop")
    cka = lbs.metrics.Metric(lbs.metrics.CKA)
    brsc_cka = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, i_map, cka)
    brsc_cka.score(sample_split_coord="experiment", neuroid_split_coord="subject")
    log(f"brainscore (cka) = {brsc_cka}")
    IPython.embed()


if __name__ == "__main__":
    main()
