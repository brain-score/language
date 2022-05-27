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

# from langbrainscore.benchmarks import pereira2018


def package_mean_froi_pereira2018_firstsess():
    mpf = pd.read_csv(
        f"{Path(__file__).resolve().parents[1] / 'data/Pereira_FirstSession_TrialEffectSizes_20220223.csv'}"
    )
    # mpf = pd.read_csv(f"{'../data/Pereira_FirstSession_TrialEffectSizes_20220223.csv'}")
    mpf = mpf.sort_values(by=["UID", "Session", "Experiment", "Stim"])
    subj_xrs = []
    neuroidx = 0
    for uid in tqdm(mpf.UID.unique()):
        mpf_subj = mpf[mpf.UID == uid]
        sess_xrs = []
        for sess in mpf_subj.Session.unique():
            mpf_sess = mpf_subj[mpf_subj.Session == sess]
            roi_filt = [any(n in c for n in ["Lang", "MD"]) for c in mpf_sess.columns]
            mpf_rois = mpf_sess.iloc[:, roi_filt]
            data_array = np.expand_dims(mpf_rois.values, 2)
            sess_xr = xr.DataArray(
                data_array,
                dims=("sampleid", "neuroid", "timeid"),
                coords={
                    "sampleid": (
                        np.arange(0, 384)
                        if data_array.shape[0] == 384
                        else np.arange(384, 384 + 243)
                    ),
                    "neuroid": np.arange(neuroidx, neuroidx + data_array.shape[1]),
                    "timeid": np.arange(data_array.shape[2]),
                    "stimulus": ("sampleid", mpf_sess.Sentence.str.strip('"')),
                    "passage": (
                        "sampleid",
                        list(map(lambda p_s: p_s.split("_")[0], mpf_sess.Stim)),
                    ),
                    "experiment": ("sampleid", mpf_sess.Experiment),
                    "session": (
                        "neuroid",
                        np.array(
                            [mpf_sess.Session.values[0]] * data_array.shape[1],
                            dtype=object,
                        ),
                    ),
                    "subject": (
                        "neuroid",
                        [mpf_sess.UID.values[0]] * data_array.shape[1],
                    ),
                    "roi": ("neuroid", mpf_rois.columns),
                },
            )
            sess_xrs.append(sess_xr)
        neuroidx += data_array.shape[1]
        subj_xr = xr.concat(sess_xrs, dim="sampleid")
        subj_xrs.append(subj_xr)
    mpf_xr = xr.concat(subj_xrs, dim="neuroid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "stimulus", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "passage", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "experiment", "sampleid")
    mpf_xr = collapse_multidim_coord(mpf_xr, "session", "neuroid")
    return mpf_xr


def main():
    # mpf_xr = pereira2018.pereira2018_mean_froi_nat_stories()
    mpf_xr = package_mean_froi_pereira2018_firstsess()

    # try:
    #     mpf_dataset = lbs.dataset.Dataset(
    #         xr.DataArray(),
    #         dataset_name="Pereira2018LangfROIs",
    #         _skip_checks=True,
    #     )
    #     mpf_dataset.load_cache()
    # except FileNotFoundError:
    #     mpf_dataset = lbs.dataset.Dataset(
    #         mpf_xr.isel(neuroid=mpf_xr.roi.str.contains("Lang")),
    #         dataset_name="Pereira2018LangfROIs",
    #     )
    mpf_dataset = lbs.dataset.Dataset(
        mpf_xr.isel(neuroid=mpf_xr.roi.str.contains("Lang")),
        dataset_name="Pereira2018LangfROIs",
    )

    # mpf_dataset.to_cache("test_mpf_dataset_cache", cache_dir="./cache")
    # mpf_dataset = lbs.dataset.Dataset.from_cache(
    #     "test_mpf_dataset_cache", cache_dir="./cache"
    # )
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
    modelcard_layers_variance = lbs.utils.encoder.get_layer_sparsity(
        ann_encoded_dataset=ann_enc_mpf
    )
    modelcard_layers_sparsity = lbs.utils.encoder.get_explainable_variance(
        ann_encoded_dataset=ann_enc_mpf
    )

    # Initialize mapping and metric
    ann_enc_mpf = ann_enc_mpf.isel(
        neuroid=(ann_enc_mpf.layer == 4)
    )  # Select a layer # TODO: loop over layers unless it is a brain model with commitment

    rdg_cv_kfold = lbs.mapping.LearnedMap("linridge_cv", k_fold=5)
    fisher = lbs.metrics.Metric(lbs.metrics.FisherCorr)
    brsc_rdg_corr = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, rdg_cv_kfold, fisher)
    brsc_rdg_corr.run(sample_split_coord="experiment", calc_nulls=True, iters=5)
    log(f"brainscore (rdg, fisher) = {brsc_rdg_corr.scores.mean()}")
    log(f"ceiling (rdg, fisher) = {brsc_rdg_corr.ceilings.mean()}")
    log(f"null (rdg, fisher) = {brsc_rdg_corr.nulls.mean()}")

    i_map = lbs.mapping.IdentityMap(nan_strategy="drop")
    cka = lbs.metrics.Metric(lbs.metrics.CKA)
    brsc_cka = lbs.BrainScore(ann_enc_mpf, brain_enc_mpf, i_map, cka)
    brsc_cka.score(sample_split_coord="experiment", neuroid_split_coord="subject")
    log(f"brainscore (cka) = {brsc_cka}")
    IPython.embed()


if __name__ == "__main__":
    main()
