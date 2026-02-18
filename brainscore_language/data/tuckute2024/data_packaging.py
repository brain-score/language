import os 
import sys
import logging
import numpy as np
import pandas as pd

from brainio.assemblies import NeuroidAssembly
from pathlib import Path
from brainscore_language.utils.s3 import upload_data_assembly

def upload_tuckute2024():
    assembly = load_tuckute2024_5subj(source=os.path.join(Path(__file__).parent, 'brain-lang-data_participant_20230728.csv'), roi="lang_LH_netw")
    upload_data_assembly(assembly,
        assembly_identifier="Tuckute2024.language")
    
def groupby_coord(df: pd.DataFrame,
                  coord_col: str = 'item_id',
                  aggfunc: str = 'mean',
                  ) -> pd.DataFrame:
    """
	Group a pandas dataframe by the coordinates specified in coord_cols.
	Most common use case: group by item_id (across several UIDs).

	This function by default groups the numeric columns according to the aggfunc (mean by default).
	For the string columns, the first value is kept. Importantly, we check whether all string columns are the same for
	each item_id (or other coordinate). Then, we only keep the coords that are shared across all item_ids
	(such that we avoid the scenario where we think we can use a string column as a coordinate, but in reality it was
	different for each occurence of that item_id).

	Args:
		df (pd.DataFrame): dataframe to group
		coord_col (str): column to group by (currently only supports one coord col)
		aggfunc: aggregation function for numeric columns. If std/sem, always use n-1 for ddof

	Returns:
		df_grouped (pd.DataFrame): grouped dataframe
	"""

    df_to_return = df.copy(deep=True)

    # Create df that is grouped by col coord using aggfunc
    # Keep string columns as is (first)
    if aggfunc == 'mean':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'median':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'std':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.std(ddof=1) if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'sem':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.sem(ddof=1) if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    else:
        raise ValueError(f'aggfunc {aggfunc} not supported')

    # Check whether the string arguments were all the same (to ensure correct grouping and metadata for str cols) ###
    shared_str_cols = []
    not_shared_str_cols = []
    for coord_val in df_grouped.index:
        # Get the values of the coordinate of interest in df_to_return
        df_test = df_to_return.loc[df_to_return[coord_col] == coord_val, :]
        df_test_object = df_test.loc[:, df_test.dtypes == object]  # Get the string cols

        # If not nan, check that each col has unique values
        if not df_test_object.isna().all().all():
            for col in df_test_object.columns:
                if len(df_test_object[col].unique()) > 1:
                    # print(f'Column {col} has multiple values for item_id {item_id}: {df_test_object[col].unique()}')
                    not_shared_str_cols.append(col)
                else:
                    # print(f'Column {col} for item_id {coord_col} have the same value, i.e. we can retain the metadata when grouping by item_id')
                    shared_str_cols.append(col)

    # Check that all shared_str_cols are the same for all item_ids
    shared_str_cols_unique = np.unique(shared_str_cols)
    not_shared_str_cols_unique = np.unique(not_shared_str_cols)

    # Drop the not_shared_str_cols from df_item_id
    df_grouped_final = df_grouped.drop(columns=not_shared_str_cols_unique)

    return df_grouped_final


def _tuckute2024_5subj(source: str,
                       roi: str = "lang_LH_netw"):
    """
    Package the data from Tuckute et al. 2024 into an xarray DataArray.
    The benchmark consists of 5 train subjects; 1,000 sentences each (one rep).
    The data are averaged across the 5 subjects for the language network (LH regions: IFGorb, IFG, MFG, AntTemp, PostTemp).
    """

    df = pd.read_csv(source)

    # Filter out the train subjects
    train_subjects = [848, 853, 865, 875, 876]

    df = df[df['target_UID'].isin(train_subjects)]

    # Filter out the ROIs of interest
    df = df[df['roi'].isin([roi])]
    # Assert just one ROI
    assert len(df['roi'].unique()) == 1

    # Average over item_id
    df_mean = groupby_coord(df=df, coord_col='item_id', aggfunc='mean')

    assembly = NeuroidAssembly(np.expand_dims(df_mean.response_target.values, axis=1),
        # This is the z-scored data. Use response_target_non_norm for non-z-scored data
        dims=('presentation', 'neuroid'),
        coords={'stimulus_id': ('presentation', df_mean.index.values),
                'stimulus': ('presentation', df_mean.sentence.values),
                'cond': ('presentation', df_mean.cond.values),
                'cond_approach': ('presentation', df_mean.cond_approach.values),
                'neuroid_id': ('neuroid', [1]),  # Only one neuroid (ROI)
                'subject_UID': ('neuroid', ['-'.join([str(s) for s in train_subjects])]),
                'roi': ('neuroid', df_mean.roi.unique()),
    })


    return assembly


def load_tuckute2024_5subj(source: str = 'brain-lang-data_participant_20230728.csv',
                      roi: str = "lang_LH_netw",
                      cache: bool = False):
    """ """
    assembly = _tuckute2024_5subj(source, roi=roi)

    return assembly


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_tuckute2024()
