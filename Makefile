# Makefile

VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

.PHONY: all setup install run clean

all: setup install run

setup:
	python -m venv $(VENV_DIR)

install: setup
	$(PIP) install --upgrade pip
	$(PIP) install numpy opencv-python-headless

run:
	$(PYTHON) gaussian_filter.py

clean:
	rm -rf $(VENV_DIR)
