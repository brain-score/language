import logging
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from tqdm import tqdm

from brainio import fetch
from brainio.packaging import write_netcdf, upload_to_s3
from brainscore_core import NeuroidAssembly

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""


def package_Pereira2018ROI():
    """
    Load Pereira2018ROI benchmark csv file and package it into an xarray benchmark.
    """
    data_path = Path(__file__).parent
    data_file = (
        data_path / "Pereira2018_Lang_fROI.csv"
    )  # contains both neural data and associated stimuli.
    _logger.info(f"Data file: {data_file}.")

    # get data
    data = pd.read_csv(data_file)
    data["Neuroid"] = data.UID.astype(str) + "_" + data.ROI

    neuroid_id_map = dict(zip(set(data.Neuroid), np.arange(len(set(data.Neuroid)))))
    presentation_id_map = dict(
        zip(set(data["Sentence"]), np.arange(len(set(data["Sentence"])))),
    )

    presentation_arr = [None] * len(presentation_id_map)
    experiment_arr = [None] * len(presentation_id_map)
    neuroid_arr = [None] * len(neuroid_id_map)
    subject_arr = [None] * len(neuroid_id_map)

    # inv_neuroid_map = {v: k for k, v in neuroid_id_map}
    # inv_presentation_map = {v: k for k, v in presentation_id_map}

    effect_sizes = np.array(data["EffectSize"])
    effect_sizes_placeholder = np.array(
        [np.nan for _ in neuroid_id_map for _ in presentation_id_map]
    ).reshape(len(presentation_id_map), len(neuroid_id_map))

    for ix, row in tqdm(
        data.iterrows(), total=len(data), desc=f"iterating through {data_file}"
    ):
        neuroid = row["Neuroid"]
        presentation = row["Sentence"]
        neuroid_id = neuroid_id_map[neuroid]
        presentation_id = presentation_id_map[presentation]

        effect_sizes_placeholder[presentation_id, neuroid_id] = row["EffectSize"]

        presentation_arr[presentation_id] = presentation
        experiment_arr[presentation_id] = row["Experiment"]
        subject_arr[neuroid_id] = row["UID"]
        neuroid_arr[neuroid_id] = neuroid

    assembly = NeuroidAssembly(
        effect_sizes_placeholder.reshape(*effect_sizes_placeholder.shape, 1),
        dims=("presentation", "neuroid", "time"),  # ? added time
        coords={
            "sentence": ("presentation", presentation_arr),
            "experiment": ("presentation", experiment_arr),
            "subject": ("neuroid", subject_arr),
            "roi": ("neuroid", neuroid_arr),
            "time": ("time", [0]),
        },
    )

    # upload
    upload_data_assembly(
        assembly,
        assembly_identifier="Pereira2018ROI",
        bucket_name="brainscore-language",
    )


def upload_data_assembly(assembly, assembly_identifier, bucket_name):
    # adapted from
    # https://github.com/mschrimpf/brainio/blob/8a40a3558d0b86072b9e221808f19005c7cb8c17/brainio/packaging.py#L217

    _logger.debug(f"Uploading {assembly_identifier} to S3")

    # identifiers
    assembly_store_identifier = "assy_" + assembly_identifier.replace(".", "_")
    netcdf_file_name = assembly_store_identifier + ".nc"
    target_netcdf_path = (
        Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
    )
    s3_key = netcdf_file_name

    # write to disk and upload
    netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
    _logger.debug(f"Wrote file to {target_netcdf_path}.")
    # upload_to_s3(target_netcdf_path, bucket_name, s3_key)
    # _logger.debug(
    #     f"Uploaded assembly {assembly_identifier} to S3: {s3_key} (SHA1 hash {netcdf_kf_sha1})"
    # )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    package_Pereira2018ROI()
