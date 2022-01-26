import typing

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from langbrainscore.mapping.rsa import RSA, RDM

from functools import partial

# TODO: verify behavior of LeavePOut and alternatives LeavePGroupsOut, etc.
from sklearn.model_selection import (
    KFold, # KFold without regard to any balancing coord (strat_coord) or grouping coord (split_coord)
    StratifiedKFold, # KFold balancing strat_coord across train/test splits 
    GroupKFold, # KFold keeping grouping coord (split_coord) entirely in one of train/test splits (no leakage)
    StratifiedGroupKFold, # KFold doing the group thing but also the strat thing on different coords 
)
#KFold, StratifiedShuffleSplit, LeavePOut


mapping_classes = {
    'ridge': (Ridge, {'alpha': 1.0}),
    'linear': (LinearRegression, {}),
    'rsa': (RSA, {}),
    'rdm': (RDM, {}),
}

class Mapping:
    model = None

    def __init__(self,
                 X: xr.Dataset, Y: xr.Dataset,

                 mapping_class: typing.Union[str, typing.Any],
                 random_seed: int = 42, 

                 k_fold: int = 5,
                 strat_coord: str = None,

                 num_split_groups_out: int = None, # (p, the # of groups in the test split)
                 split_coord: str = None, # (grouping coord)


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

        self.k_fold = k_fold or 1
        self.strat_coord = strat_coord

        self.num_split_groups_out = num_split_groups_out
        self.split_coord = split_coord

        self.mapping_class = mapping_class

        assert(X.sel(strat_coord.values) == Y.sel(strat_coord.values).all()) 
        assert(X.sel(split_coord.values) == Y.sel(split_coord.values).all()) 
        self.X, self.Y = X, Y

        if type(mapping_class) == str:
            mapping_class, _kwargs = mapping_classes[mapping_class]
            kwargs.update(_kwargs)
        
        # to save (this model uses the entire data rather than constructing splits)
        self.full_model = mapping_class(**kwargs, random_state=random_seed)
        # placeholder model with the right params that we'll reuse across splits
        self.model = mapping_class(**kwargs, random_state=random_seed)

    @staticmethod
    def construct_splits_(xr_dataset: xr.Dataset, # Y: xr.Dataset, 
                          strat_coord: str = None, k_folds: int = 5,
                          split_coord: str = None, num_split_groups_out: int = None,
                          random_seed: int = 42
                          ):

        if strat_coord and split_coord:
            kf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, xr_dataset, y=xr_dataset.sel(split_coord), group=xr_dataset.sel(strat_coord))
        elif split_coord:
            kf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, xr_dataset, group=xr_dataset.sel(split_coord))
        elif strat_coord:
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, xr_dataset, y=xr_dataset.sel(strat_coord))
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, xr_dataset)

        return split()


    def construct_splits(self, xr_dataset: xr.Dataset):
        return self.construct_splits_(xr_dataset,
                                      self.strat_coord, self.k_fold, 
                                      self.split_coord, self.num_split_groups_out,
                                      random_seed=self.random_seed)

        
    def fit_full(self, X, Y):
        # TODO
        self.fit(X, Y, k_folds=1)
        raise NotImplemented

    def fit(self, X: xr.Dataset, Y: xr.Dataset) -> None:
        """creates a mapping model using k-fold cross-validation stratified by groups

        Args:, groups=None, k_folds: int = 5
            X ([type]): [description]
            Y ([type]): [description]
            groups ([type], optional): [description]. Defaults to None.
            k_folds (int, optional): [description]. Defaults to 5.

        Returns:
            [type]: [description]
        """        

        alpha_across_splits = []
        train_indices = []
        test_indices = []

        # if no groups are provided, use simple k splits
        if groups is None:
            # k-fold CV for an array 'Y'
            groups = np.tile(np.arange(k_folds), (Y.shape[0] // k_folds + 1))[: Y.shape[0]]  
            np.random.shuffle(groups)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        X_test_collection = []
        Y_test_collection = []
        Y_pred_collection = []

        for train_index, test_index in kf.split(groups):
            
            train_indices.append(train_index)
            test_indices.append(test_index)

            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = Y[train_index, :], Y[test_index, :]
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            y_pred = np.squeeze(y_pred)

            Y_test_collection.append(y_test)
            Y_pred_collection.append(y_pred)

        return Y_pred_collection, Y_test_collection


    def map(self, source, target) -> None:
        '''
        the works: constructs splits, fits models for each split, then evaluates the fit 
                of each split and returns the result (also for each split)
        '''
        pass

        
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
    ...
