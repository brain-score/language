# Source Data

Per-subject Pearson r values from Hadidi et al. (2025) "Illusions of Alignment Between Large Language Models and Brains Emerge From Fragile Methods and Overlooked Confounds."

Downloaded from: https://github.com/ebrahimfeghhi/beyond-brainscore/tree/main/source_data/figure1

## Files

All CSVs have columns: `subjects, Network, Model, perf`

Models: GPT2-XL, GPT2-XL-mp (mean-pooled), GPT2-XL-sp (sum-pooled), OASM

### figure1/

| File | Dataset | CV Split | Subjects |
|------|---------|----------|----------|
| `pereira_pearson_r_shuffled.csv` | Pereira2018 | Shuffled | 10 |
| `pereira_pearson_r_contig.csv` | Pereira2018 | Contiguous (passage-level) | 10 |
| `fedorenko_pearson_r_shuffled.csv` | Fedorenko2016 | Shuffled | 5 |
| `fedorenko_pearson_r_contig.csv` | Fedorenko2016 | Contiguous (passage-level) | 5 |
| `blank_pearson_r_shuffled.csv` | Blank2014 | Shuffled | 5 |
| `blank_pearson_r_contig.csv` | Blank2014 | Contiguous (passage-level) | 5 |

These are the L2-regularized (ridge regression) results used in the paper's Figure 1.
