from random import sample
import typing

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, RidgeCV
from langbrainscore.utils import logging

from functools import partial

# TODO: verify behavior of LeavePOut and alternatives LeavePGroupsOut, etc.
from sklearn.model_selection import (
    KFold, # KFold without regard to any balancing coord (strat_coord) or grouping coord (split_coord)
    StratifiedKFold, # KFold balancing strat_coord across train/test splits 
    GroupKFold, # KFold keeping grouping coord (split_coord) entirely in one of train/test splits (no leakage)
    StratifiedGroupKFold, # KFold doing the group thing but also the strat thing on different coords 
)
#KFold, StratifiedShuffleSplit, LeavePOut




class Mapping:
    model = None

    def __init__(self,
                 X: xr.DataArray, Y: xr.DataArray,

                 mapping_class: typing.Union[str, typing.Any] = None,
                 random_seed: int = 42, 

                 k_fold: int = 5,
                 strat_coord: str = None,

                 num_split_groups_out: int = None, # (p, the # of groups in the test split)
                 split_coord: str = None, # (grouping coord)

                 #TODO
                 # handle predict held-out subject # but then we have to do mean over ROIs
                 # because individual neuroids do not correspond
                 # we kind of already have this along the `sampleid` coordinate, but we
                 # need to implement this in the neuroid coordinate

                 **kwargs) -> None:
        """Initializes a Mapping object that establishes a mapping between two encoder representations.
           The mapping is initialized with certain parameters baked in, accepted as arguments to
           the init function, listed below.

        Args:
            mapping_class (typing.Union[str, typing.Any], required): [description]. 
                This Class will be instatiated to get a mapping model. E.g. LinearRegression, Ridge,
                from the sklearn family. Must implement <?classifier> interface
            random_seed (int, optional): [description]. Defaults to 42.
            k_fold (int, optional): [description]. Defaults to 5.
            strat_coord (str, optional): [description]. Defaults to None.
            num_split_groups_out (int, optional): [description]. Defaults to None.
            split_coord (str, optional): [description]. Defaults to None.
        """
        self.random_seed = random_seed
        mapping_classes = {
            'ridge': (Ridge, {'alpha': 1.0}),
            'ridge_cv': (RidgeCV, {'alphas': np.logspace(-3, 3, 13), 'alpha_per_target': True}),
            'linear': (LinearRegression, {}),
            None: None,
        }

        self.k_fold = k_fold or 1
        self.strat_coord = strat_coord

        self.num_split_groups_out = num_split_groups_out
        self.split_coord = split_coord

        self.mapping_class = mapping_class

        if strat_coord:
            try:
                assert (X[strat_coord].values == Y[strat_coord].values).all()
            except AssertionError as e:
                raise ValueError(f'{strat_coord} coordinate does not align across X and Y')
        if split_coord:
            try:
                assert (X[split_coord].values == Y[split_coord].values).all() 
            except AssertionError as e:
                raise ValueError(f'{split_coord} coordinate does not align across X and Y')
        
        # TODO:
        # make sure there are no stimuli that have NaNs in all places along the neuroid dimension

        self.X, self.Y = X, Y
        
        
        
        self.X_nan_mask = X.isnull()
        self.Y_nan_mask = Y.isnull()
        self.X_nan_removed = X.dropna('neuroid')
        self.Y_nan_removed = Y.dropna('neuroid')

        logging.log(f'X shape: {X.data.shape}, NaNs: {self.X_nan_mask.sum()}; after NaN removal: {self.X_nan_removed.data.shape}')
        logging.log(f'Y shape: {Y.data.shape}, NaNs: {self.Y_nan_mask.sum()}; after NaN removal: {self.Y_nan_removed.data.shape}')

        if type(mapping_class) == str:
            mapping_class, _kwargs = mapping_classes[mapping_class]
            kwargs.update(_kwargs)
        
        # to save (this model uses the entire data rather than constructing splits)
        self.full_model = mapping_class(**kwargs)
        # placeholder model with the right params that we'll reuse across splits
        self.model = mapping_class(**kwargs)
        
        logging.log(f'initialized Mapping with {mapping_class}, {type(self.model)}!')


    @staticmethod
    def _extract_dense(A = None):
        '''
        returns a list of several xarrays each of which is dense (has no NaNs). 
        each will have a subset of the sampleids

        Args:
            A (xr.DataArray): 
        '''

    def extract_dense(self):
        dense_X = self._extract_dense_arrays(self.X)


    @staticmethod
    def _construct_splits(xr_dataset: xr.Dataset, # Y: xr.Dataset, 
                          strat_coord: str = None, k_folds: int = 5,
                          split_coord: str = None, num_split_groups_out: int = None,
                          random_seed: int = 42
                         ):

        sampleid = xr_dataset.sampleid.values

        if strat_coord and split_coord:
            kf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, sampleid, y=xr_dataset[split_coord].values, groups=xr_dataset[strat_coord].values)
        elif split_coord:
            kf = GroupKFold(n_splits=k_folds)
            split = partial(kf.split, sampleid, groups=xr_dataset[split_coord].values)
        elif strat_coord:
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, sampleid, y=xr_dataset[strat_coord].values)
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, sampleid)

        logging.log(f'running {type(kf)}!')
        return split()


    def construct_splits(self, A):
        return self._construct_splits(A,
                                      self.strat_coord, self.k_fold, 
                                      self.split_coord, self.num_split_groups_out,
                                      random_seed=self.random_seed)

        
    def fit_full(self, X, Y):
        # TODO
        self.fit(X, Y, k_folds=1)
        raise NotImplemented

    def fit(self, 
            #X: xr.Dataset, Y: xr.Dataset
           ) -> None:
        """creates a mapping model using k-fold cross-validation
            -> uses params from the class initialization, uses strat_coord
               and split_coord to stratify and split across group boundaries

        Returns:
            [type]: [description]
        """        

        for neuroid in self.Y.neuroid.values:

            # limit data to current neuroid, and then drop the samples that are missing data for this neuroid
            Y_slice = self.Y.sel(neuroid=neuroid).dropna(dim='sampleid')
            Y_filtered_sampleids = Y_slice.sampleid
            assert set(Y_filtered_sampleids.values).issubset(set(self.X.sampleid.values))
            logging.log(f'for neuroid {neuroid}, we used {(num_retained := len(Y_filtered_sampleids))} samples; dropped {len(self.Y.sampleid) - num_retained}') 

            X_slice = self.X.sel(sampleid=Y_filtered_sampleids.values)

            # these collections store each split for our records later
            alpha_across_splits = [] # only used in case of ridge_cv # TODO
            # TODO we aren't saving this to the object instance yet
            train_indices = []
            test_indices = []

            splits = self.construct_splits(Y_slice)

            # X_test_collection = []
            Y_test_collection = []
            Y_pred_collection = []

            for train_index, test_index in splits:
                
                train_indices.append(train_index)
                test_indices.append(test_index)

                # !! NOTE the _nan_removed variants instead of X and Y
                X_train, X_test = (
                    X_slice.sel(sampleid=Y_slice.sampleid.values[train_index]),
                    X_slice.sel(sampleid=Y_slice.sampleid.values[test_index]),
                )
                y_train, y_test = (
                    Y_slice.sel(sampleid=Y_slice.sampleid.values[train_index]),
                    Y_slice.sel(sampleid=Y_slice.sampleid.values[test_index]),
                )

                y_pred_over_time = []
                for timeid in y_train.timeid:

                    # TODO: change this code for models that also have a non-singleton timeid
                    # i.e., output evolves in time (RNN?)
                    self.model.fit(X_train.sel(timeid=0), y_train.sel(timeid=timeid).values.reshape(-1, 1))

                    # deepcopy
                    y_pred = y_test.sel(timeid=timeid).copy(deep=True).expand_dims('timeid', 1)
                    y_pred.assign_coords(timeid=('timeid', [timeid]))
                    y_pred.data = self.model.predict(X_test.sel(timeid=0)) #y_pred
                    y_pred_over_time.append(y_pred)

                y_pred_over_time = xr.concat(y_pred_over_time, dim='timeid')
                Y_pred_collection.append(y_pred_over_time)

                Y_test_collection.append(y_test)

            # ACTUALLY TODO the below is no longer true:
            #   now the Y_test_collection members are xarrays with a timeid dimension/coord
            #   but Y_pred_collection is a list of numpy arrays per discrete timeid

            # the return value is a dictionary of test/pred;
            # each of test/pred is a list of lists with two levels of
            # nesting as below:
            #   first level: CV folds
            #       second level: timeids
            yield dict(test=Y_test_collection, 
                       pred=Y_pred_collection)

    
    # def map(self, source, target) -> None:
    #     '''
    #     the works: constructs splits, fits models for each split, then evaluates the fit 
    #             of each split and returns the result (also for each split)
    #     '''
    #     pass

        
    def save_model(self) -> None:
        '''TODO: stuff that needs to be saved eventually

        - model weights
        - CV stuff (if using CV); but all arguments needed for initializing, in general
            - n_splits
            - random_state
            - split indices (based on random seed)
            - params per split (alpha, lambda)
            - validation score, etc. for each CV split?
        '''
        pass

    def predict(self, source) -> None:
        pass


class IdentityMap(Mapping):
    """Identity mapping for running RSA-type analyses that don't need splits into cv folds and don't need affine maps"""
    
    def fit(self,
            # X: xr.Dataset, Y: xr.Dataset
            ) -> None:
        """creates a mapping model using k-fold cross-validation
            depending on the class initialization, uses strat_coord
            and split_coord to stratify and split across group boundaries

        Args:, groups=None, k_folds: int = 5
            X ([type]): [description]
            Y ([type]): [description]
            k_folds (int, optional): [description]. Defaults to 5.

        Returns:
            [type]: [description]
        """
        
        return dict(test=[self.Y.data],
                    pred=[[self.X.data.sel(timeid=i) for i in self.X.timeid.values]])
    
    
