"""
Class for automated testing of neural network modules.
"""

import torch
import copy

class NNModuleAutoTest:
    r"""Test the following cases:
    1) Assert if the module return at least one tensor
    2) Reproducibility. Assert that the module return the same output when called with the same data
    Additionally 
    3) Assert returned shape. 
        Important! If module returns multiple tensor. The shapes for assertion must be in list() not (!!!) tuple()
    """
    SEED = 0

    def __init__(self, params):
        self.params = params

    def run(self):
        for param in self.params:
            assert "module" in param and "init" in param and "forward" in param
            module = self.exec_func(param["module"], param["init"])
            cloned_inp = self.clone_input(param["forward"])
            
            result, result_2 = self.exec_twice(module, param["forward"], cloned_inp)

            if type(result) != tuple:
                result = (result, )
                result_2 = (result_2, )

            self.assert_return_tensor(result)
            self.assert_equal_output(module, result, result_2)

            if "assert_shape" in param:
                if type(param["assert_shape"]) != list:
                    param["assert_shape"] = [param["assert_shape"]]

                self.assert_shape(result, param["assert_shape"])

    def exec_twice(self, module, inp_1, inp_2):
        torch.manual_seed(self.SEED)
        result = self.exec_func(module, inp_1)
        
        torch.manual_seed(self.SEED)
        result_2 = self.exec_func(module, inp_2)

        return result, result_2

    def exec_func(self, func, args):
        if type(args) == tuple:
            return func(*args)
        elif type(args) == dict:
            return func(**args)
        else:
            raise TypeError(f"{type(args)} is not correct type for funcntion arguments.")
        
    def clone_input(self, args):
        if type(args) == tuple:
            return tuple(self.clone_object(a) for a in args)
        elif type(args) == dict:
            return {k: self.clone_object(v) for k, v in args.items()}

    def clone_object(self, obj):
        if hasattr(obj, "clone"):
            return obj.clone()
        else:
            return copy.deepcopy(obj)
        
    def assert_return_tensor(self, result):
        assert any(isinstance(r, torch.Tensor)  for r in result)

    def assert_equal_output(self, module, result, result_2):    
        assert len(result) == len(result_2)
        
        for i, r1 in enumerate(result):
            r2 = result_2[i]
            if isinstance(r1, torch.Tensor):
                assert torch.equal(r1, r2)
            else:
                assert r1 == r2

    def assert_shape(self, result, shapes):
        i = 0
        for t in result:
            if isinstance(t, torch.Tensor):
                assert t.shape == shapes[i] 
                i += 1