""" Test the TBEvaluator class."""
import pytest

from topobenchmark.evaluator import TBEvaluator

def test_evaluator():
    """ Setup the test."""
    evaluator_multilable = TBEvaluator(task="multilabel classification")
    evaluator_regression = TBEvaluator(task="regression")
    with pytest.raises(ValueError):
        TBEvaluator(task="wrong")
    repr = evaluator_multilable.__repr__()