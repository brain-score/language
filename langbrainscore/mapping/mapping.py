import typing
from functools import partial
from random import sample

import numpy as np
import xarray as xr
from langbrainscore.utils import logging
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge,
                                  RidgeCV)
# TODO: verify behavior of LeavePOut and alternatives LeavePGroupsOut, etc.
from sklearn.model_selection import \
    GroupKFold  # KFold keeping grouping coord (split_coord) entirely in one of train/test splits (no leakage)
from sklearn.model_selection import \
    KFold  # KFold without regard to any balancing coord (strat_coord) or grouping coord (split_coord)
from sklearn.model_selection import \
    StratifiedGroupKFold  # KFold doing the group thing but also the strat thing on different coords
from sklearn.model_selection import \
    StratifiedKFold  # KFold balancing strat_coord across train/test splits

# KFold, StratifiedShuffleSplit, LeavePOut


class Mapping:
    model = None

    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        mapping_class: typing.Union[str, typing.Any] = None,
        random_seed: int = 42,
        k_fold: int = 5,
        strat_coord: str = None,
        num_split_groups_out: int = None,  # (p, the # of groups in the test split)
        split_coord: str = None,  # (grouping coord)
        # TODO
        # handle predict held-out subject # but then we have to do mean over ROIs
        # because individual neuroids do not correspond
        # we kind of already have this along the `sampleid` coordinate, but we
        # need to implement this in the neuroid coordinate
        **kwargs,
    ) -> None:
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
            "ridge": (Ridge, {"alpha": 1.0}),
            "ridge_cv": (
                RidgeCV,
                {"alphas": np.logspace(-3, 3, 13), "alpha_per_target": True},
            ),
            "linear": (LinearRegression, {}),
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
                raise ValueError(
                    f"{strat_coord} coordinate does not align across X and Y"
                )
        if split_coord:
            try:
                assert (X[split_coord].values == Y[split_coord].values).all()
            except AssertionError as e:
                raise ValueError(
                    f"{split_coord} coordinate does not align across X and Y"
                )

        self.X, self.Y = X, Y

        logging.log(f"X shape: {X.data.shape}")
        logging.log(f"Y shape: {Y.data.shape}")

        if type(mapping_class) == str:
            mapping_class, _kwargs = mapping_classes[mapping_class]
            kwargs.update(_kwargs)

        # to save (this model uses the entire data rather than constructing splits)
        if mapping_class:
            self.full_model = mapping_class(**kwargs)
            # placeholder model with the right params that we'll reuse across splits
            self.model = mapping_class(**kwargs)

            logging.log(
                f"initialized Mapping with {mapping_class}, {type(self.model)}!"
            )

    @staticmethod
    def _extract_dense(A=None):
        """
        returns a list of several xarrays each of which is dense (has no NaNs).
        each will have a subset of the sampleids

        Args:
            A (xr.DataArray):
        """

    def extract_dense(self):
        dense_X = self._extract_dense_arrays(self.X)

    @staticmethod
    def _construct_splits(
        xr_dataset: xr.Dataset,  # Y: xr.Dataset,
        strat_coord: str = None,
        k_folds: int = 5,
        split_coord: str = None,
        num_split_groups_out: int = None,
        random_seed: int = 42,
    ):

        sampleid = xr_dataset.sampleid.values

        if strat_coord and split_coord:
            kf = StratifiedGroupKFold(
                n_splits=k_folds, shuffle=True, random_state=random_seed
            )
            split = partial(
                kf.split,
                sampleid,
                y=xr_dataset[split_coord].values,
                groups=xr_dataset[strat_coord].values,
            )
        elif split_coord:
            kf = GroupKFold(n_splits=k_folds)
            split = partial(kf.split, sampleid, groups=xr_dataset[split_coord].values)
        elif strat_coord:
            kf = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=random_seed
            )
            split = partial(kf.split, sampleid, y=xr_dataset[strat_coord].values)
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            split = partial(kf.split, sampleid)

        logging.log(f"running {type(kf)}!")
        return split()

    def construct_splits(self, A):
        return self._construct_splits(
            A,
            self.strat_coord,
            self.k_fold,
            self.split_coord,
            self.num_split_groups_out,
            random_seed=self.random_seed,
        )

    def fit_full(self, X, Y):
        # TODO
        self.fit(X, Y, k_folds=1)
        raise NotImplemented

    def _check_sampleids(
        self, X: xr.DataArray, Y: xr.DataArray,
    ):
        """
        checks that the sampleids in X and Y are the same
        """

        if X.sampleid.values.shape != Y.sampleid.values.shape:
            raise ValueError("X and Y sampleid shapes do not match!")
        if not np.all(X.sampleid.values == Y.sampleid.values):
            raise ValueError("X and Y sampleids do not match!")

        logging.log(f"Passed sampleid check for neuroid {Y.neuroid.values}")

    def _drop_na(
        self, X: xr.DataArray, Y: xr.DataArray, dim: str = "sampleid", **kwargs
    ):
        """
        drop samples with missing values (based on Y) in X or Y along specified dimension
        Make sure that X and Y now have the same sampleids
        """
        # limit data to current neuroid, and then drop the samples that are missing data for this neuroid
        Y_slice = Y.dropna(dim=dim, **kwargs)
        Y_filtered_ids = Y_slice[dim].values

        assert set(Y_filtered_ids).issubset(set(X[dim].values))

        logging.log(
            f"for neuroid {Y_slice.neuroid.values}, we used {(num_retained := len(Y_filtered_ids))} samples; dropped {len(Y[dim]) - num_retained}"
        )

        # use only the samples that are in Y
        X_slice = self.X.sel(sampleid=Y_filtered_ids)

        return X_slice, Y_slice

    def _permute_X(
        self, X: xr.DataArray, method: str = "shuffle_X_rows", random_state: int = 42,
    ):
        """Permute the features of X.

		Parameters
		----------
		X : xr.DataArray
			The embeddings to be permuted
		method : str
			The method to use for permutation.
			'shuffle_X_rows' : Shuffle the rows of X (=shuffle the sentences and create a mismatch between the sentence embeddings and target)
			'shuffle_each_X_col': For each column (=feature/unit) of X, permute that feature's values across all sentences.
								  Retains the statistics of the original features (e.g., mean per feature) but the values of the features are shuffled for each sentence.
		random_state : int
			The seed for the random number generator.

		Returns
		-------
		xr.DataArray
			The permuted dataarray
		"""

        X_orig = X.copy(deep=True)

        logging.log(f"OBS: permuting X with method {method}")

        if method == "shuffle_X_rows":
            X = X.sample(
                n=X.shape[1], random_state=random_state
            )  # check whether X_shape is correct

        elif method == "shuffle_each_X_col":
            np.random.seed(random_state)
            for feat in X.data.shape[0]:  # per neuroid
                np.random.shuffle(X.data[feat, :])

        else:
            raise ValueError(f"Invalid method: {method}")

        assert X.shape == X_orig.shape
        assert np.all(X.data != X_orig.data)

        return X

    def fit(self, permute_X: typing.Union[bool, str] = False,) -> typing.Dict:
        """creates a mapping model using k-fold cross-validation
            -> uses params from the class initialization, uses strat_coord
               and split_coord to stratify and split across group boundaries

        Returns:
            [type]: [description]
        """
        # Loop across each Y neuroid (target)
        for neuroid in self.Y.neuroid.values:

            Y_neuroid = self.Y.sel(neuroid=neuroid)

            # limit data to current neuroid, and then drop the samples that are missing data for this neuroid
            X_slice, Y_slice = self._drop_na(self.X, Y_neuroid, dim="sampleid")

            # Assert that X and Y have the same sampleids
            self._check_sampleids(X_slice, Y_slice)

            # We can perform various checks by 'permuting' the source, X
            # TODO this is a test! do not use under normal workflow!
            if permute_X:
                X_slice = self._permute_X(X_slice, method=permute_X)

            # these collections store each split for our records later
            # TODO we aren't saving this to the object instance yet
            train_indices = []
            test_indices = []
            # only used in case of ridge_cv or any duck type that uses an alpha hparam
            alpha_across_splits = []

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

                # empty list to house the y_predictions per timeid
                y_pred_over_time = []
                # empty dictionary to house the hparam alpha per timeid
                alpha_over_time = {}

                for timeid in y_train.timeid:

                    # TODO: change this code for models that also have a non-singleton timeid
                    # i.e., output evolves in time (RNN?)
                    self.model.fit(
                        X_train.sel(timeid=0),
                        y_train.sel(timeid=timeid).values.reshape(-1, 1),
                    )

                    # store the hparam values related to the fitted models
                    alpha_over_time[timeid.item()] = getattr(
                        self.model, "alpha_", np.nan
                    )

                    # deepcopy `y_test` as `y_pred` to inherit some of the metadata and dims
                    # and then populate it with our new predicted values
                    y_pred = (
                        y_test.sel(timeid=timeid)
                        .copy(deep=True)
                        .expand_dims("timeid", 1)
                    )
                    y_pred.assign_coords(timeid=("timeid", [timeid]))
                    y_pred.data = self.model.predict(X_test.sel(timeid=0))  # y_pred
                    y_pred_over_time.append(y_pred)

                y_pred_over_time = xr.concat(y_pred_over_time, dim="timeid")
                Y_pred_collection.append(y_pred_over_time)
                Y_test_collection.append(y_test)
                alpha_across_splits.append(alpha_over_time)

            # the return value is a dictionary of test/pred;
            # each of test/pred is a list of lists with two levels of
            # nesting as below:
            #   first level: CV folds
            #       second level: timeids
            yield dict(
                test=Y_test_collection,
                pred=Y_pred_collection,
                alphas=alpha_across_splits,
            )

    # def map(self, source, target) -> None:
    #     '''
    #     the works: constructs splits, fits models for each split, then evaluates the fit
    #             of each split and returns the result (also for each split)
    #     '''
    #     pass

    def save_model(self) -> None:
        """TODO: stuff that needs to be saved eventually

        - model weights
        - CV stuff (if using CV); but all arguments needed for initializing, in general
            - n_splits
            - random_state
            - split indices (based on random seed)
            - params per split (alpha, lambda)
            - validation score, etc. for each CV split?
        """
        pass

    def predict(self, source) -> None:
        pass


class IdentityMap(Mapping):
    """Identity mapping for running RSA-type analyses that don't need splits into cv folds and don't need affine maps"""

    def fit(self):
        return dict(
            test=[self.Y], pred=[[self.X.sel(timeid=i) for i in self.X.timeid.values]]
        )