# """This module implements the encoders for the neural networks."""

# from .all_cell_encoder import AllCellFeatureEncoder
# from .base import AbstractFeatureEncoder
# from .dgm_encoder import DGMStructureFeatureEncoder

# # ... import other encoders classes here
# # For example:
# # from topobenchmarkx.nn.encoders.other_encoder_1 import OtherEncoder1
# # from topobenchmarkx.nn.encoders.other_encoder_2 import OtherEncoder2

# __all__ = [
#     "AbstractFeatureEncoder",
#     "AllCellFeatureEncoder",
#     "DGMStructureFeatureEncoder",
#     # "OtherEncoder1",
#     # "OtherEncoder2",
#     # ... add other readout classes here
# ]

"""Data encoders module with automated exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any


class ModuleExportsManager:
    """Manages automatic discovery and registration of data encoder classes."""

    @staticmethod
    def is_encoder_class(obj: Any) -> bool:
        """Check if an object is a valid encoder class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid encoder class.

        Returns
        -------
        bool
            True if the object is a valid encoder class (non-private class
            defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_encoders(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all encoder classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        encoders = {}

        # Get the directory containing the encoder modules
        package_dir = Path(package_path).parent

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import the module
            module_name = f"{Path(package_path).stem}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all encoder classes in the module
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                    ):
                        encoders[name] = obj  # noqa: PERF403

        return encoders


# Create the exports manager
manager = ModuleExportsManager()

# Automatically discover and populate DATA_encoderS
ENCODERS = manager.discover_encoders(__file__)

# Automatically generate __all__
__all__ = [*ENCODERS.keys(), "ENCODERS"]

# For backwards compatibility, also create individual imports
locals().update(ENCODERS)
