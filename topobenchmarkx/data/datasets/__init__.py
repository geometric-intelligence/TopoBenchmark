"""This module contains the implemented dataset classes for the topological benchmarking experiments."""

from .us_county_demos_dataset import USCountyDemosDataset

# Import the dataset that you have generated: LanguageDataset (see Tutorial "add_new_dataset.ipynb")
from .language_dataset import LanguageDataset

PLANETOID_DATASETS = [
    "Cora",
    "citeseer",
    "PubMed",
]

TU_DATASETS = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "COLLAB",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "NCI1",
    "NCI109",
]

FIXED_SPLITS_DATASETS = ["ZINC", "AQSOL"]

HETEROPHILIC_DATASETS = [
    "amazon_ratings",
    "questions",
    "minesweeper",
    "roman_empire",
    "tolokers",
]

PYG_DATASETS = (
    PLANETOID_DATASETS
    + TU_DATASETS
    + FIXED_SPLITS_DATASETS
    + HETEROPHILIC_DATASETS
)

__all__ = [
    "PYG_DATASETS",
    "PLANETOID_DATASETS",
    "TU_DATASETS",
    "FIXED_SPLITS_DATASETS",
    "HETEROPHILIC_DATASETS",
    "USCountyDemosDataset",
    
    # Add new dataset here to allow it to be imported through topobenchmarkx.data.datasets (see Tutorial "add_new_dataset.ipynb")
    "LanguageDataset",
]
