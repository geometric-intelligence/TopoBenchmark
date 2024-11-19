"""
Class for automated testing of neural network modules.
"""

import torch
import copy

class NNModuleAutoTest:
    r"""Test the following cases.

    1) Assert if the module return at least one tensor
    2) Reproducibility. Assert that the module return the same output when called with the same data
    Additionally 
    3) Assert returned shape. 
        Important! If module returns multiple tensor. The shapes for assertion must be in list() not (!!!) tuple().

    Parameters
    ----------
    params : list of dict
        List of dictionaries with parameters.
    """
    SEED = 0

    def __init__(self, params):
        self.params = params

    def run(self):
        r"""Run the test.
        """
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
        """ Execute the module twice with different inputs.

            Parameters
            ----------
            module : torch.nn.Module
                Module to be tested.
            inp_1 : tuple or dict
                Input for the module.
            inp_2 : tuple or dict
                Input for the module.

            Returns
            -------
            tuple
                Output of the module for the first input.
        """
        torch.manual_seed(self.SEED)
        result = self.exec_func(module, inp_1)
        
        torch.manual_seed(self.SEED)
        result_2 = self.exec_func(module, inp_2)

        return result, result_2

    def exec_func(self, func, args):
        """ Execute the function with the arguments.
            
            Parameters
            ----------
            func : function
                Function to be executed.
            args : tuple or dict
                Arguments for the function.
            
            Returns
            -------
            any
                Output of the function.
        """
        if type(args) == tuple:
            return func(*args)
        elif type(args) == dict:
            return func(**args)
        else:
            raise TypeError(f"{type(args)} is not correct type for funnction arguments.")
        
    def clone_input(self, args):
        """ Clone the input arguments.

            Parameters
            ----------
            args : tuple or dict
                Arguments to be cloned.
            
            Returns
            -------
            tuple or dict
                Cloned arguments.
        """
        if type(args) == tuple:
            return tuple(self.clone_object(a) for a in args)
        elif type(args) == dict:
            return {k: self.clone_object(v) for k, v in args.items()}

    def clone_object(self, obj):
        """ Clone the object.

            Parameters
            ----------
            obj : any
                Object to be cloned.

            Returns
            -------
            any
                Cloned object.
        """
        if hasattr(obj, "clone"):
            return obj.clone()
        else:
            return copy.deepcopy(obj)
        
    def assert_return_tensor(self, result):
        """ Assert if the module return at least one tensor.

            Parameters
            ----------
            result : any
                Output of the module.
        """
        if all(isinstance(r, tuple) for r in result):
            assert any([all([isinstance(r, torch.Tensor) for r in tup]) for tup in result])
        else:
            assert any(isinstance(r, torch.Tensor)  for r in result)

    def assert_equal_output(self, module, result, result_2):    
        """ Assert that the module return the same output when called with the same data.

            Parameters
            ----------
            module : torch.nn.Module
                Module to be tested.
            result : any
                Output of the module for the first input.
            result_2 : any
                Output of the module for the second input.
        """
        assert len(result) == len(result_2)
        
        for i, r1 in enumerate(result):
            r2 = result_2[i]
            if isinstance(r1, torch.Tensor):
                assert torch.equal(r1, r2)
            elif isinstance(r1, tuple) and isinstance(r2, tuple):
                for r1_, r2_ in zip(r1, r2):
                    if isinstance(r1_, torch.Tensor) and isinstance(r2_, torch.Tensor):
                        assert torch.equal(r1_, r2_)
                    else:
                        assert  r1_ == r2_
            else:
                assert r1 == r2

    def assert_shape(self, result, shapes):
        """ Assert returned shape.

            Parameters
            ----------
            result : any
                Output of the module.
            shapes : list
                List of shapes to be asserted.
        """
        i = 0
        for t in result:
            if isinstance(t, torch.Tensor):
                assert t.shape == shapes[i] 
                i += 1