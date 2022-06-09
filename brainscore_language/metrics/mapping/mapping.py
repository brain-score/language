import typing
from functools import partial

from tqdm.auto import tqdm
import numpy as np
from joblib import Parallel, delayed
import xarray as xr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, RidgeCV

from brainscore_language.interface import _Mapping
from brainscore_language.utils import logging
from brainscore_language.utils.xarray import collapse_multidim_coord

mapping_classes_params = {
    "linreg": (LinearRegression, {}),
    "linridge_cv": (RidgeCV, {"alphas": np.logspace(-3, 3, 13)}),
    "linpls": (PLSRegression, {"n_components": 20}),
}


class IdentityMap(_Mapping):
    """
    Identity mapping for use with metrics that operate
    on non column-aligned matrices, e.g., RSA, CKA

    Imputes NaNs for downstream metrics.
    """

    def __init__(self, nan_strategy: str = "drop") -> "IdentityMap":
        self._nan_strategy = nan_strategy

    def fit_transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        ceiling: bool = False,
    ):
        if ceiling:
            logging.log("ceiling not supported for IdentityMap yet")
        # TODO: figure out how to handle NaNs better...
        if self._nan_strategy == "drop":
            X_clean = X.copy(deep=True).dropna(dim="neuroid")
            Y_clean = Y.copy(deep=True).dropna(dim="neuroid")
        elif self._nan_strategy == "impute":
            X_clean = X.copy(deep=True).fillna(0)
            Y_clean = Y.copy(deep=True).fillna(0)
        else:
            raise NotImplementedError("unsupported nan strategy.")
        return X_clean, Y_clean


class LearnedMap(_Mapping):
    def __init__(
        self,
        mapping_class: typing.Union[
            str, typing.Tuple[typing.Callable, typing.Mapping[str, typing.Any]]
        ],
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
    ) -> "LearnedMap":
        """
        Initializes a Mapping object that describes a mapping between two encoder representations.

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
        self.k_fold = k_fold
        self.strat_coord = strat_coord
        self.num_split_groups_out = num_split_groups_out
        self.split_coord = split_coord
        self.mapping_class_name = mapping_class
        self.mapping_params = kwargs

        if type(mapping_class) == str:
            _mapping_class, _kwargs = mapping_classes_params[self.mapping_class_name]
            self.mapping_params.update(_kwargs)
        # in the spirit of duck-typing, we don't need any of these checks. we will automatically
        # fail if we're missing any of these attributes
        # else:
        #     assert callable(mapping_class)
        #     assert hasattr(mapping_class(), "fit")
        #     assert hasattr(mapping_class(), "predict")

        # TODO: what is the difference between these two (model; full_model)? let's make this less
        # confusing
        self.full_model = _mapping_class(**self.mapping_params)
        self.model = _mapping_class(**self.mapping_params)
        logging.log(f"initialized Mapping with {type(self.model)}!")

    @staticmethod
    def _construct_splits(
        xr_dataset: xr.Dataset,
        strat_coord: str,
        k_folds: int,
        split_coord: str,
        num_split_groups_out: int,
        random_seed: int,
    ):
        from sklearn.model_selection import (
            GroupKFold,
            KFold,
            StratifiedGroupKFold,
            StratifiedKFold,
        )

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

        logging.log(f"running {type(kf)}!", verbosity_check=True)
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
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
    ):
        """
        checks that the sampleids in X and Y are the same
        """

        if X.sampleid.values.shape != Y.sampleid.values.shape:
            raise ValueError("X and Y sampleid shapes do not match!")
        if not np.all(X.sampleid.values == Y.sampleid.values):
            raise ValueError("X and Y sampleids do not match!")

        logging.log(
            f"Passed sampleid check for neuroid {Y.neuroid.values}",
            verbosity_check=True,
        )

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
            f"for neuroid {Y_slice.neuroid.values}, we used {(num_retained := len(Y_filtered_ids))}"
            f" samples; dropped {len(Y[dim]) - num_retained}",
            verbosity_check=True,
        )

        # use only the samples that are in Y
        X_slice = X.sel(sampleid=Y_filtered_ids)

        return X_slice, Y_slice

    # def _permute_X(
    #     self,
    #     X: xr.DataArray,
    #     method: str = "shuffle_X_rows",
    #     random_state: int = 42,
    # ):
    #     """Permute the features of X.
    #
    #     Parameters
    #     ----------
    #     X : xr.DataArray
    #             The embeddings to be permuted
    #     method : str
    #             The method to use for permutation.
    #             'shuffle_X_rows' : Shuffle the rows of X (=shuffle the sentences and create a mismatch between the sentence embeddings and target)
    #             'shuffle_each_X_col': For each column (=feature/unit) of X, permute that feature's values across all sentences.
    #                                                       Retains the statistics of the original features (e.g., mean per feature) but the values of the features are shuffled for each sentence.
    #     random_state : int
    #             The seed for the random number generator.
    #
    #     Returns
    #     -------
    #     xr.DataArray
    #             The permuted dataarray
    #     """
    #
    #     X_orig = X.copy(deep=True)
    #
    #     if logging.get_verbosity():
    #         logging.log(f"OBS: permuting X with method {method}")
    #
    #     if method == "shuffle_X_rows":
    #         X = X.sample(
    #             n=X.shape[1], random_state=random_state
    #         )  # check whether X_shape is correct
    #
    #     elif method == "shuffle_each_X_col":
    #         np.random.seed(random_state)
    #         for feat in X.data.shape[0]:  # per neuroid
    #             np.random.shuffle(X.data[feat, :])
    #
    #     else:
    #         raise ValueError(f"Invalid method: {method}")
    #
    #     assert X.shape == X_orig.shape
    #     assert np.all(X.data != X_orig.data)
    #
    #     return X

    def fit_transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        # permute_X: typing.Union[bool, str] = False,
        ceiling: bool = False,
        ceiling_coord: str = "subject",
    ) -> typing.Tuple[xr.DataArray, xr.DataArray]:
        """creates a mapping model using k-fold cross-validation
            -> uses params from the class initialization, uses strat_coord
               and split_coord to stratify and split across group boundaries

        Returns:
            [type]: [description]
        """
        from sklearn.random_projection import GaussianRandomProjection

        if ceiling:
            n_neuroids = X.neuroid.values.size
            X = Y.copy()

        logging.log(f"X shape: {X.data.shape}", verbosity_check=True)
        logging.log(f"Y shape: {Y.data.shape}", verbosity_check=True)

        if self.strat_coord:
            try:
                assert (X[self.strat_coord].values == Y[self.strat_coord].values).all()
            except AssertionError as e:
                raise ValueError(
                    f"{self.strat_coord} coordinate does not align across X and Y"
                )
        if self.split_coord:
            try:
                assert (X[self.split_coord].values == Y[self.split_coord].values).all()
            except AssertionError as e:
                raise ValueError(
                    f"{self.split_coord} coordinate does not align across X and Y"
                )

        def fit_per_neuroid(neuroid):
            Y_neuroid = Y.sel(neuroid=neuroid)

            # limit data to current neuroid, and then drop the samples that are missing data for this neuroid
            X_slice, Y_slice = self._drop_na(X, Y_neuroid, dim="sampleid")

            # Assert that X and Y have the same sampleids
            self._check_sampleids(X_slice, Y_slice)

            # select relevant ceiling split
            if ceiling:
                X_slice = X_slice.isel(
                    neuroid=X_slice[ceiling_coord] != Y_slice[ceiling_coord]
                ).dropna(dim="neuroid")

            # We can perform various sanity checks by 'permuting' the source, X
            # NOTE this is a test! do not use under normal workflow!
            # if permute_X:
            #     logging.log(
            #         f"`permute_X` flag is enabled. only do this in an adversarial setting.",
            #         cmap="WARN",
            #         type="WARN",
            #         verbosity_check=True,
            #     )
            #     X_slice = self._permute_X(X_slice, method=permute_X)

            # these collections store each split for our records later
            # TODO we aren't saving this to the object instance yet
            train_indices = []
            test_indices = []
            # only used in case of ridge_cv or any duck type that uses an alpha hparam

            splits = self.construct_splits(Y_slice)

            # X_test_collection = []
            Y_test_collection = []
            Y_pred_collection = []

            for cvfoldid, (train_index, test_index) in enumerate(splits):

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

                for timeid in y_train.timeid:

                    # TODO: change this code for models that also have a non-singleton timeid
                    # i.e., output evolves in time (RNN?)

                    x_model_train = X_train.sel(timeid=0).values
                    y_model_train = y_train.sel(timeid=timeid).values.reshape(-1, 1)

                    if ceiling and x_model_train.shape[1] > n_neuroids:
                        projection = GaussianRandomProjection(
                            n_components=n_neuroids, random_state=0
                        )
                        x_model_train = projection.fit_transform(x_model_train)

                    self.model.fit(
                        x_model_train,
                        y_model_train,
                    )

                    # store the hparam values related to the fitted models
                    alpha = getattr(self.model, "alpha_", np.nan)

                    # deepcopy `y_test` as `y_pred` to inherit some of the metadata and dims
                    # and then populate it with our new predicted values
                    y_pred = (
                        y_test.sel(timeid=timeid)
                        .copy(deep=True)
                        .expand_dims("timeid", 1)
                    )
                    x_model_test = X_test.sel(timeid=0)
                    if ceiling and x_model_train.shape[1] > n_neuroids:
                        x_model_test = projection.transform(x_model_test)
                    y_pred.data = self.model.predict(x_model_test)  # y_pred
                    y_pred = y_pred.assign_coords(timeid=("timeid", [timeid]))
                    y_pred = y_pred.assign_coords(alpha=("timeid", [alpha]))
                    y_pred = y_pred.assign_coords(cvfoldid=("timeid", [cvfoldid]))
                    y_pred_over_time.append(y_pred)

                y_pred_over_time = xr.concat(y_pred_over_time, dim="timeid")
                Y_pred_collection.append(y_pred_over_time)
                Y_test_collection.append(y_test)

            Y_test = xr.concat(Y_test_collection, dim="sampleid").sortby("sampleid")
            Y_pred = xr.concat(Y_pred_collection, dim="sampleid").sortby("sampleid")

            # test.append(Y_test)
            # pred.append(Y_pred)
            return Y_test, Y_pred

        # Loop across each Y neuroid (target)
        test = []
        pred = []
        # TODO: parallelize using order-preserving joblib-mapping
        # for neuroid in tqdm(Y.neuroid.values, desc="fitting a model per neuroid"):
        for t, p in Parallel(n_jobs=-2)(
            delayed(fit_per_neuroid)(neuroid)
            for neuroid in tqdm(Y.neuroid.values, desc="fitting a model per neuroid")
        ):
            test += [t]
            pred += [p]

        test_xr = xr.concat(test, dim="neuroid").transpose(
            "sampleid", "neuroid", "timeid"
        )
        pred_xr = xr.concat(pred, dim="neuroid").transpose(
            "sampleid", "neuroid", "timeid"
        )

        if test_xr.stimulus.ndim > 1:
            test_xr = collapse_multidim_coord(test_xr, "stimulus", "sampleid")
        if pred_xr.stimulus.ndim > 1:
            pred_xr = collapse_multidim_coord(pred_xr, "stimulus", "sampleid")

        return pred_xr, test_xr

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
