#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER=python3


ifeq (,$(shell which conda))
  HAS_CONDA=False
else
  HAS_CONDA=True
endif

#################################################################################
# COMMANDS             	                                                        #
#################################################################################


## Clean environment
clean:
	pre-commit run --all-files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete

## Generates a requirements file from the poetry env
env:
	poetry export -f requirements.txt --output envs/requirements.txt --without-hashes
