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
