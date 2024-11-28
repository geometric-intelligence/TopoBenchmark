"""This module contains a transform that tunes a DecisionTreeClassifier using grid search
and applies the best model for classification."""

import torch
import torch_geometric
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class TuneDecisionTree(torch_geometric.transforms.BaseTransform):
    r"""A transform that tunes a DecisionTreeClassifier using grid search
    and evaluates the best model on the validation data.

    Parameters
    ----------
    param_grid : dict, optional
        The parameter grid for hyperparameter tuning. Keys include:
        - 'criterion': ['gini', 'entropy']
        - 'max_depth': [None, 5, 10, 20]
        - 'min_samples_split': [2, 5, 10, 20]
        - 'min_samples_leaf': [1, 2, 5, 10]
        - 'max_features': [None, 'sqrt', 'log2']
        - 'class_weight': [None, 'balanced']
    random_state : int, default=42
        The random state for reproducibility.
    """

    def __init__(self, param_grid=None, random_state=42):
        super().__init__()
        self.type = "tune_decision_tree"
        self.param_grid = (
            param_grid
            if param_grid is not None
            else {
                "criterion": ["gini", "entropy"],
                "max_depth": [
                    5,
                    10,
                    20,
                    None,
                ],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": [None, "sqrt", "log2"],
                "class_weight": [None, "balanced"],
            }
        )
        self.random_state = random_state
        self.best_tree = None
        self.best_params = None
        self.validation_accuracy = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"best_params={self.best_params!r}, "
            f"validation_accuracy={self.validation_accuracy:.4f})"
        )

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to tune the classifier and evaluate it.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with additional attributes:
            - `data.best_tree` (DecisionTreeClassifier): The best classifier.
            - `data.validation_accuracy` (float): Accuracy on validation data.
        """
        # Split training and validation data
        x_train = data.x[data.train_mask].numpy()
        y_train = data.y[data.train_mask].numpy()
        x_val = data.x[data.val_mask].numpy()
        y_val = data.y[data.val_mask].numpy()

        # Initialize GridSearchCV for hyperparameter tuning
        dtree = DecisionTreeClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=dtree,
            param_grid=self.param_grid,
            scoring="accuracy",
            cv=2,  # 5-fold cross-validation
            n_jobs=-1,  # Use all available CPU cores
            verbose=1,  # Display progress
        )
        grid_search.fit(x_train, y_train)

        # Save the best model and hyperparameters
        self.best_tree = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Evaluate on the validation data

        H_tree = torch.Tensor(self.best_tree.decision_path(data.x).todense())
        data["H_tree"] = H_tree

        # Add results to the data object
        data["tree_model"] = self.best_tree

        y_pred = self.best_tree.predict(x_train)
        print("Train score", accuracy_score(y_train, y_pred))

        y_pred = self.best_tree.predict(x_val)
        print("Validation score", accuracy_score(y_val, y_pred))

        return data

    def base_param_tree(self, data, **kwargs):
        dtree = DecisionTreeClassifier(**kwargs)

        # Split training and validation data
        x_train = data.x[data.train_mask].numpy()
        y_train = data.y[data.train_mask].numpy()
        x_val = data.x[data.val_mask].numpy()
        y_val = data.y[data.val_mask].numpy()

        dtree.fit(x_train, y_train)
        H_tree = torch.Tensor(dtree.decision_path(data.x).todense())

        data["H_tree"] = H_tree

        # Add results to the data object
        data["tree_model"] = dtree

        y_pred = dtree.predict(x_train)
        print("Train score", accuracy_score(y_train, y_pred))

        y_pred = dtree.predict(x_val)
        print("Validation score", accuracy_score(y_val, y_pred))
        return data
