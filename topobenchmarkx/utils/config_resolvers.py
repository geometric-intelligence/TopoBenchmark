def get_default_transform(data_domain, model):
    r"""Get default transform for a given data domain and model.
    
    Parameters
    ----------
    data_domain: str
        Data domain.
    model: str
        Model name. Should be in the format "model_domain/name".
        
    Returns
    -------
    str
        Default transform.
    
    Raises
    ------
    ValueError
        If the combination of data_domain and model is invalid.
    """
    model_domain = model.split("/")[0]
    if data_domain == model_domain:
        return "identity"
    elif data_domain == "graph" and model_domain != "combinatorial":
        return f"graph2{model_domain}_default"
    else:
        raise ValueError(
            f"Invalid combination of data_domain={data_domain} and model_domain={model_domain}"
        )


def get_monitor_metric(task, loss):
    r"""Get monitor metric for a given task and loss.
    
    Parameters
    ----------
    task: str
        Task, either "classification" or "regression".
    loss: str
        Name of the loss function.
    
    Returns
    -------
    str
        Monitor metric.
    
    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == "classification":
        return "val/accuracy"
    elif task == "regression":
        return "val/" + loss
    else:
        raise ValueError(f"Invalid task {task}")


def get_monitor_mode(task):
    r"""Get monitor mode for a given task.
    
    Parameters
    ----------
    task: str
        Task, either "classification" or "regression".
    
    Returns
    -------
    str
        Monitor mode, either "max" or "min".
    
    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == "classification":
        return "max"
    elif task == "regression":
        return "min"
    else:
        raise ValueError(f"Invalid task {task}")


def infer_in_channels(dataset):
    r"""Infer the number of input channels for a given dataset.
    
    Parameters
    ----------
    dataset: torch_geometric.data.Dataset
        Input dataset.
    
    Returns
    -------
    list
        List with dimensions of the input channels.
    """
    def find_complex_lifting(dataset):
        r"""Find if there is a complex lifting in the dataset.
        
        Parameters
        ----------
        dataset: torch_geometric.data.Dataset
            Input dataset.
        
        Returns
        -------
        bool
            True if there is a complex lifting, False otherwise.
        str
            Name of the complex lifting, if it exists.
        """
        if "transforms" not in dataset:
            return False, None
        complex_transforms = [
            "graph2cell_lifting",
            "graph2simplicial_lifting",
            "graph2combinatorial_lifting",
        ]
        for t in complex_transforms:
            if t in dataset.transforms:
                return True, t
        return False, None

    def check_for_type_feature_lifting(dataset, lifting):
        r"""Check the type of feature lifting in the dataset.
        
        Parameters
        ----------
        dataset: torch_geometric.data.Dataset
            Input dataset.
        lifting: str
            Name of the complex lifting.
        
        Returns
        -------
        str
            Type of feature lifting.
        """
        lifting_params_keys = dataset.transforms[lifting].keys()
        if "feature_lifting" in lifting_params_keys:
            feature_lifting = dataset.transforms[lifting]["feature_lifting"]
        else:
            feature_lifting = "projection"

        return feature_lifting

    there_is_complex_lifting, lifting = find_complex_lifting(dataset)
    if there_is_complex_lifting:
        # Get type of feature lifting
        feature_lifting = check_for_type_feature_lifting(dataset, lifting)

        if isinstance(dataset.parameters.num_features, int):
            if feature_lifting == "projection":
                return [dataset.parameters.num_features] * dataset.transforms[
                    lifting
                ].complex_dim

            elif feature_lifting == "concatenation":
                return_value = [dataset.parameters.num_features]
                for i in range(2, dataset.transforms[lifting].complex_dim + 1):
                    return_value += [int(dataset.parameters.num_features * i)]

                return return_value

            else:
                return [dataset.parameters.num_features] * dataset.transforms[
                    lifting
                ].complex_dim
        else:
            if not dataset.transforms[lifting].preserve_edge_attr:
                if feature_lifting == "projection":
                    return [dataset.parameters.num_features[0]] * dataset.transforms[
                        lifting
                    ].complex_dim
                elif feature_lifting == "concatenation":
                    return_value = [dataset.parameters.num_features]
                    for i in range(2, dataset.transforms[lifting].complex_dim + 1):
                        return_value += [int(dataset.parameters.num_features * i)]

                    return return_value
                else:
                    return [dataset.parameters.num_features] * dataset.transforms[
                        lifting
                    ].complex_dim

            else:
                return list(dataset.parameters.num_features) + [
                    dataset.parameters.num_features[1]
                ] * (
                    dataset.transforms[lifting].complex_dim
                    + 1
                    - len(dataset.parameters.num_features)
                )
    else:
        if isinstance(dataset.parameters.num_features, int):
            return [dataset.parameters.num_features]
        else:
            return [dataset.parameters.num_features[0]]
