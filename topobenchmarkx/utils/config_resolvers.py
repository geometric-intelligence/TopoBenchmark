def get_default_transform(data_domain, model):
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
    if task == "classification":
        return "val/accuracy"
    elif task == "regression":
        return "val/" + loss
    else:
        raise ValueError(f"Invalid task {task}")


def get_monitor_mode(task):
    if task == "classification":
        return "max"
    elif task == "regression":
        return "min"
    else:
        raise ValueError(f"Invalid task {task}")


def infer_in_channels(dataset):
    def find_complex_lifting(dataset):
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

    there_is_complex_lifting, lifting = find_complex_lifting(dataset)
    if there_is_complex_lifting:
        if isinstance(dataset.parameters.num_features, int):
            return [dataset.parameters.num_features] * dataset.transforms[
                lifting
            ].complex_dim
        else:
            if not dataset.transforms[lifting].preserve_edge_attr:
                return [dataset.parameters.num_features[0]] * dataset.transforms[
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
