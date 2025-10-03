#!/bin/bash

# run.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the Makefile exists in the current directory
if [ ! -f Makefile ]; then
    echo "Makefile not found!"
    exit 1
fi

# Run the Makefile
make all
