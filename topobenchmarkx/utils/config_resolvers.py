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
    
    def check_for_type_feature_lifting(dataset, lifting):
        lifting_params_keys = dataset.transforms[lifting].keys()
        if 'feature_lifting' in lifting_params_keys:
            feature_lifting = dataset.transforms[lifting]['feature_lifting']
        else:
            feature_lifting = 'projection'

        return feature_lifting


    there_is_complex_lifting, lifting = find_complex_lifting(dataset)
    if there_is_complex_lifting:
        # Get type of feature lifting
        feature_lifting = check_for_type_feature_lifting(dataset, lifting)

        if isinstance(dataset.parameters.num_features, int):
            if feature_lifting == 'projection':
                return [dataset.parameters.num_features] * dataset.transforms[
                    lifting
                ].complex_dim
            
            else: 
                return_value = [dataset.parameters.num_features]
                for i in range(2, dataset.transforms[lifting].complex_dim + 1):
                    return_value += [int(dataset.parameters.num_features * i)]

                return return_value
        else:
            if not dataset.transforms[lifting].preserve_edge_attr:
                if feature_lifting == 'projection':
                    return [dataset.parameters.num_features[0]] * dataset.transforms[
                        lifting
                    ].complex_dim
                else:
                    return_value = [dataset.parameters.num_features]
                    for i in range(2, dataset.transforms[lifting].complex_dim + 1):
                        return_value += [int(dataset.parameters.num_features * i)]

                    return return_value

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
