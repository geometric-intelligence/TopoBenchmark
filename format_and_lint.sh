#!/bin/sh

# Run ruff to check for issues and fix them
ruff check . --fix

# Run docformatter to reformat docstrings and comments
docformatter --in-place --recursive --wrap-summaries 79 --wrap-descriptions 79 .

# Run black to format the code
black .