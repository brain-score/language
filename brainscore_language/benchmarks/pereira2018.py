
#  COMMENTED because a lot of the imports don't work as is
def _pereira2018_mean_froi() -> xr.DataArray:
    """ """
    import pandas as pd
    import numpy as np
    from tqdm.auto import tqdm
    import xarray as xr
    from pathlib import Path
    from brainscore_language.utils.logging import log
    from brainscore_language.utils.xarray import collapse_multidim_coord
    from brainscore_language.dataset import Dataset
    source = (
        Path(__file__).parents[2]
        / "data/Pereira_FirstSession_TrialEffectSizes_20220223.csv"
    )
    mpf = pd.read_csv(source)
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

    mpf_xr.attrs["source"] = str(source)
    mpf_xr.attrs["measurement"] = "fmri"
    mpf_xr.attrs["modality"] = "text"
    # mpf_xr.attrs["name"] = f"pereira2018_mean_froi"

    return mpf_xr


def pereira2018_mean_froi(network="Lang", load_cache=True) -> Dataset:
    """ """

    import pandas as pd
    import numpy as np
    from tqdm.auto import tqdm
    import xarray as xr
    from pathlib import Path
    from brainscore_language.utils.logging import log
    from brainscore_language.utils.xarray import collapse_multidim_coord
    from brainscore_language.dataset import Dataset

    dataset_name = (
        f"pereira2018_mean_froi_{network}" if network else "pereira2018_mean_froi"
    )

    def package() -> Dataset:
        mpf_xr = _pereira2018_mean_froi()
        if network:
            mpf_xr = mpf_xr.isel(neuroid=mpf_xr.roi.str.contains(network))
        mpf_dataset = Dataset(
            mpf_xr,
            dataset_name=dataset_name,
            # modality="text"
        )
        return mpf_dataset

    if load_cache:
        try:
            mpf_dataset = Dataset(
                xr.DataArray(),
                dataset_name=dataset_name,
                # modality="text",
                _skip_checks=True,
            )
            mpf_dataset.load_cache()
        except FileNotFoundError:
            mpf_dataset = package()
    else:
        mpf_dataset = package()
        # mpf_dataset.to_cache()

    return mpf_dataset
