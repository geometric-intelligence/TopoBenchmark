"""Init file for hypergraph load module."""

from .citation_hypergraph_dataset_loader import CitationHypergraphDatasetLoader

HYPERGRAPH_LOADERS = {
    "CitationHypergraphDatasetLoader": CitationHypergraphDatasetLoader,
}

__all__ = ["HYPERGRAPH_LOADERS"]
