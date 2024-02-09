import torch_geometric


class TransformChain:
    """Class that applies multiple transforms sequentially to a data."""

    def __init__(self, **kwargs):
        # Make kwargs to a list to preserve the order of the transforms
        self.transforms = list(kwargs.values())
        self.names = list(kwargs.keys())

        # Collect the parameters of the transforms
        self.parameters = {}
        self.repo_name = ""
        for transform_name in self.names:
            self.parameters[transform_name] = kwargs[transform_name].parameters
            self.repo_name += transform_name + "_"

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply multiple transforms sequentially to a signal.

        :param signal: Signal to transform
        :return: Modified signal
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def __call__(self, data):
        return self.forward(data)
