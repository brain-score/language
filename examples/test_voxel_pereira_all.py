from pathlib import Path

import langbrainscore as lbs
from langbrainscore.utils.logging import log


if __name__ == "__main__":
    dataset = lbs.dataset.Dataset.from_file_or_url(
        file_path_or_url=f"{Path(__file__).parents[1] / 'data/Pereira.parquet.gzip'}",
        data_column="EffectSize",
        sampleid_index="Stim",
        neuroid_index="Voxel",
        stimuli_index="Sentence",
        subject_index="UID",
        sampleid_metadata=["Experiment"],
        neuroid_metadata=["Network", "ROI"],
    )
    dataset.to_cache("pereira_all", cache_dir="./cache")
    log(f"stimuli: {dataset.stimuli.values}")
