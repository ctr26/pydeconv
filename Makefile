all: build install test

build:
	poetry build
install:
	poetry install
test:
	poetry run pytest
test.notebooks:
	poetry run pytest --nbmake **/*ipynb