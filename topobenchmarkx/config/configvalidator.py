"""This module has simple validation for config files."""
from pathlib import Path

default_expected_parameters = (
    # "SequentialFileProcessor",
    # "FileSampler",
    # "FileLoader"
)


class ConfigValidator:
    """ConfigValidator checks that configurations in a file respect basic
    constrains. The higher-level keys should be present and 'dataset_dir'
    should exist."""

    def __init__(self, expected_parameters: tuple = default_expected_parameters):
        self.expected_parameters = expected_parameters

    def validate(self, configs: dict):
        """Raise error if configurations aren't valid.
        :param configs: Dictionary with configurations
        """

        self._raise_parameter_missing_error_if_param_is_missing(configs)

        for dir in configs["FileSampler"]["dataset_dirs"]:
            self._raise_file_exists_error_if_dir_doesnt_exist(dir)

    def _raise_parameter_missing_error_if_param_is_missing(self, configs: dict):
        for param in self.expected_parameters:
            if param not in configs:
                raise ParameterMissingError(
                    f"Parameter '{param}' is missing from configuration"
                )

    @staticmethod
    def _raise_file_exists_error_if_dir_doesnt_exist(dir):
        if not Path(dir).is_dir():
            raise FileExistsError(
                f"Dataset dir '{dir}' provided in " "config file doesn't exist"
            )


class ParameterMissingError(Exception):
    """Error thrown when an expected parameter is missing."""
