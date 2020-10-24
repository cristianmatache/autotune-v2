# # Make sure to set the below in your .bashrc
# PYENV=<path to your environment>  # e.g. PYENV="/d/Anaconda3/envs/ws"
# export PATH="$PYENV/Scripts":"$PYENV/bin":$PATH

# FORMAT ---------------------------------------------------------------------------------------------------------------
docformatter:
	python -m docformatter -r . --in-place --wrap-summaries=120 --wrap-descriptions=120

isort:
	python -m isort autotune/  -m 4 -l 120

fmt: docformatter isort

# LINT -----------------------------------------------------------------------------------------------------------------
docformatter-check:
	python -m docformatter -r . --check --wrap-summaries=120 --wrap-descriptions=120 || \
	python -m docformatter -r . --wrap-summaries=120 --wrap-descriptions=120 && exit 1 # Show diff if check failed

isort-check:
	python -m isort --check-only autotune/ -m 4 -l 120 || \
	python -m isort --diff --color autotune/ -m 4 -l 120 && exit 1 # Show diff if check failed


flake8:
	python -m flake8 . --config=build-support/.flake8

pylint:
	python -m pylint autotune/ --rcfile=build-support/.pylintrc

lint: flake8 isort-check docformatter-check  # pylint

# TYPE CHECK -----------------------------------------------------------------------------------------------------------
mypy:
	python -m mypy autotune/benchmarks/ autotune/core/ autotune/datasets/ autotune/util/ --config-file build-support/mypy.ini

# CLEAN ----------------------------------------------------------------------------------------------------------------
clean-pyc:
	find . -name *.pyc | xargs rm -f && find . -name *.pyo | xargs rm -f;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info


# OTHERS  --------------------------------------------------------------------------------------------------------------
pre-commit: mypy flake8 isort docformatter

check-all: mypy lint
