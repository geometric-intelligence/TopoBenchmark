from topobenchmarkx.transforms.liftings import GraphLifting


class Graph2HypergraphLifting(GraphLifting):
    r"""Abstract class for lifting graphs to hypergraphs.
    
    Args:
        kwargs (optional): Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2hypergraph"
