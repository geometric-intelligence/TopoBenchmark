"""Unit tests for rich_utils."""
import pytest
from unittest.mock import MagicMock, patch
from topobenchmarkx.utils.rich_utils import enforce_tags, print_config_tree
from omegaconf import DictConfig

@patch("topobenchmarkx.utils.rich_utils.pylogger.RankedLogger.info")
@patch("topobenchmarkx.utils.rich_utils.rich.tree.Tree")
@patch("topobenchmarkx.utils.rich_utils.rich.syntax.Syntax")
@patch("topobenchmarkx.utils.rich_utils.rich.print")
@patch("topobenchmarkx.utils.rich_utils.Path.write_text")
@patch("topobenchmarkx.utils.rich_utils.HydraConfig.get")
def test_print_config_tree(mock_hydra_config_get, mock_write_text, mock_rich_print, mock_syntax, mock_tree, mock_info):
    '''Test the print_config_tree function.
    
    Parameters
    ----------
    mock_hydra_config_get : MagicMock
        Mock of HydraConfig.get.
    mock_write_text : MagicMock
        Mock of Path.write_text.
    mock_rich_print : MagicMock
        Mock of rich.print.
    mock_syntax : MagicMock
        Mock of rich.syntax.Syntax.
    mock_tree : MagicMock
        Mock of rich.tree.Tree.
    mock_info : MagicMock
        Mock of pylogger.RankedLogger.info.
    '''
    # Mock the input DictConfig
    mock_cfg = DictConfig({
        "data": "mock_data",
        "model": "mock_model",
        "callbacks": "mock_callbacks",
        "logger": "mock_logger",
        "trainer": "mock_trainer",
        "paths": {"output_dir": "mock_output_dir"},
        "extras": "mock_extras"
    })

    # Mock the HydraConfig.get return value
    mock_hydra_config_get.return_value = {"run": {"dir": "mock_dir"}}

    # Call the function with save_to_file=False
    print_config_tree(mock_cfg, save_to_file=False)

    # Check if rich.print was called
    assert mock_rich_print.called

    # Check if the info logger was not called
    mock_info.assert_not_called()

    with pytest.raises(FileNotFoundError):
        # Call the function with save_to_file=True
        print_config_tree(mock_cfg, save_to_file=True)

if __name__ == "__main__":
    pytest.main()