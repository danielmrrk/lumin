#!/bin/bash

# Run tests with coverage
coverage run -m pytest

# Show the coverage report in the terminal
coverage report

# Generate HTML coverage report
coverage html

# Open the HTML report in the default web browser
# MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    open htmlcov/index.html
# Linux
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open htmlcov/index.html
# Windows
elif [[ "$OSTYPE" == "msys" ]]; then
    start htmlcov/index.html
else
    echo "Cannot automatically open HTML report on this operating system."
fi
