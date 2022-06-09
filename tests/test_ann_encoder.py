
"""
this script runs some tests to check the consistency of representations produced by the ANNencoder class,
minimally, in simple cases (no context, single sentence, etc). the motivation of this group of tests is
some inconsistency in control-neural encode method
"""

import brainscore_language as lbs
from brainscore_language.utils.logging import log
from pathlib import Path

# pereira_xr = lbs.benchmarks.pereira2018_nat_stories()
# dataset = lbs.dataset.Dataset(pereira_xr)

def test_simple_stimulus():


    ann_enc = lbs.encoder.HuggingFaceEncoder(model_id="distilgpt2",
                                            emb_case="lower",
                                            emb_preproc=tuple())

    dataset = load_simple_dataset_from_file()
    # Encode
    # brain_enc_pereira = brain_enc.encode(mpf_dataset)
    ann_enc_pereira = ann_enc.encode(dataset).representations
    # log(f"created brain-encoded data of shape: {brain_enc_pereira.shape}")
    log(f"created ann-encoded data of shape: {ann_enc_pereira.shape}")

    # ANN encoder checks
    ann_enc_check = lbs.encoder.EncoderCheck()
    all_good_tol, bad_stims_tol = ann_enc_check.similiarity_metric_across_layers(
        sim_metric="diff",
        enc1=ann_enc_pereira,
        enc2=ann_enc_pereira)

    all_good_sim, bad_stims_sim = ann_enc_check.similiarity_metric_across_layers(
        sim_metric="cos",
        enc1=ann_enc_pereira,
        enc2=ann_enc_pereira)



def load_simple_dataset_from_file() -> lbs.dataset.Dataset:

    filepath = Path(__file__).parents[1] / 'data/molten_Pereira_FirstSession_TrialEffectSizes_20220223-copy.csv'
    log(f'constructing a Dataset using {filepath}')
    dataset = lbs.dataset.Dataset.from_file_or_url(filepath, 
                                       data_column='effect_size',
                                       sampleid_index='Stim', 
                                       neuroid_index='ROI',
                                       subject_index='UID',
                                       stimuli_index='Sentence',
                                       
                                       sampleid_metadata={'Stim': 'stim_identifier',
                                                        #   'Sentence': 'sentence',
                                                         },
                                       neuroid_metadata={'ROI': 'roi',
                                                         'Session': 'session',
                                                         'DurationTR': 'tr',
                                                         'Experiment': 'experiment',
                                                        #  'UID': 'subject',
                                                         },
                                       timeid_metadata=None,

                                       multidim_metadata=None, # if we had, 
                                                               # for example, a stimulus unfolding with time
                                       sort_by=['UID', 'Session', 'Experiment'],
                                    )
    return dataset


if __name__ == '__main__':
    test_simple_stimulus()