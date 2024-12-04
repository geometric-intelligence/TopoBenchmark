""" Test the TBXEvaluator class."""
import pytest

from topobenchmark.evaluator import TBXEvaluator

class TestTBXEvaluator:
    """ Test the TBXEvaluator class."""
    
    def setup_method(self):
        """ Setup the test."""
        self.evaluator_multilable = TBXEvaluator(task="multilabel classification")
        self.evaluator_regression = TBXEvaluator(task="regression")
        with pytest.raises(ValueError):
            TBXEvaluator(task="wrong")
        repr = self.evaluator_multilable.__repr__()