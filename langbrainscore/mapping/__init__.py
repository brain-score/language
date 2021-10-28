# from langbrainscore.mapping.
import typing
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

mapping_classes = {
    'ridge': (Ridge, {'alpha': 1.0}),
}

class Mapping:
    model = None

    def __init__(self, mapping_class: typing.Union[str, typing.Any] = None, random_seed: int = 42, **kwargs) -> None:
        if type(mapping_class) == str:
            mapping_class, _kwargs = mapping_classes[mapping_class]
            kwargs.update(_kwargs)
        
        self.model = mapping_class(**kwargs, random_state=random_seed)

    def map_cv(self, X: np.ndarray, Y: np.ndarray, groups=None, k_folds: int = 5) -> None:
        """creates a mapping model using k-fold cross-validation stratified by groups

        Args:
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


    def map_full(self, source, target) -> None:
        '''returns a fitted model
        '''

        
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
