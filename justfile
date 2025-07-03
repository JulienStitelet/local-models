alias c := check

default:
    @just --list

# Check formatting
check:
    poetry run black --check .

clean:
    rm -rf .ipynb_checkpoints
    rm -rf **/.ipynb_checkpoints
    rm -rf .pytest_cache
    rm -rf **/.pytest_cache
    rm -rf __pycache__
    rm -rf **/__pycache__

# Run formatting and linting
format:
    poetry run isort .
    poetry run black .

install:
    poetry install

test:
    poetry run pytest --log-level=WARNING --disable-pytest-warnings

coverage:
    poetry run coverage run -m pytest --log-level=WARNING --disable-pytest-warnings
    poetry run coverage html
    # open htmlcov/index.html

preco:
    poetry run isort .
    poetry run black .
    poetry run pytest ./tests
    poetry run pylint src/
