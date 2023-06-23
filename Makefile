all: build install test

build:
	poetry build
install:
	poetry install
test:
	poetry run pytest
test.notebooks:
	poetry run pytest --nbmake **/*ipynb

get.fly.data:
	wget -nc https://download.fht.org/jug/n2v/BSD68_reproducibility.zip \
	https://download.fht.org/jug/n2v/RGB.zip \
	https://download.fht.org/jug/n2v/flywing-data.zip \
	https://download.fht.org/jug/n2v/SEM.zip \