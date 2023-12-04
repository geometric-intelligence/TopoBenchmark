"""This module has a class to load preprocessing configurations."""
import yaml

from topobenchmarkx.config.configvalidator import ConfigValidator


class ConfigLoader:
    """ConfigLoader is responsible to load preprocessing configurations.
    Attributes:
        - validator: Raise error if raw configs have error
    """

    def __init__(self, config_validator: ConfigValidator):
        self.validator = config_validator

    def load(self, config_file: str):
        with open(config_file, "r", encoding="latin-1") as file:
            configurations = yaml.full_load(file)

        self._validate(configurations)

        return configurations

    def _validate(self, configurations: dict):
        self.validator.validate(configurations)


def create_config_loader() -> ConfigLoader:
    """Instantiate a config validator object
    :return: New config loader
    """
    config_validator = ConfigValidator()
    return ConfigLoader(config_validator)
