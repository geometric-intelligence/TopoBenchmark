import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Loader(ABC):
    """Abstract class that provides an interface to load dataset"""

    def __init__(self, configs: dict):
        """

        Loader class is responsible for:
          - loading the audio
          - delete silence if needed
          - normalization or standartization input signal

        Arguments: (default)
            sample_rate: int
            min_duration: float
            remove_silence: bool
            normalize: bool
        """

        logger.info("Instantiated Loader object")

    @abstractmethod
    def load(self, file: str) -> Data:
        """Load dataset into Data object.

        Parameters:
          :file: Path to dataset file to load
          :label: Label can be any type, lately passed as a value to the dict

        :return: Data
        """

    # def _raise_file_extension_error_if_file_extension_isnt_allowed(self, file):
    #     extension = extract_extension_from_file(file)
    #     if extension not in self._audio_file_extensions:
    #         raise FileExtensionError(f"'{extension}' extension can't be loaded.")
