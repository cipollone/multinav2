<h1 align="center">
  <b>Multinav</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/multinav">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/multinav">
  </a>
  <a href="https://pypi.org/project/multinav">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/multinav" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/multinav" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/multinav">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/multinav">
  </a>
  <a href="https://github.com/marcofavorito/multinav/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/marcofavorito/multinav">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/marcofavorito/multinav/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/marcofavorito/multinav/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/marcofavorito/multinav/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/marcofavorito/multinav">
    <img alt="codecov" src="https://codecov.io/gh/marcofavorito/multinav/branch/master/graph/badge.svg?token=FG3ATGP5P5">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="" src="https://img.shields.io/badge/flake8-checked-blueviolet">
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="" src="https://img.shields.io/badge/mypy-checked-blue">
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://www.mkdocs.org/">
    <img alt="" src="https://img.shields.io/badge/docs-mkdocs-9cf">
  </a>
</p>


## Preliminaries

Development mode install.

- Clone this repo with:
```
git clone --recurse-submodules https://github.com/cipollone/multinav
```

- Install Poetry, if you don't have it already:
```
pip install poetry
```

- Set up the virtual environment. 

```
poetry install
```

- Now you can run parts of this software. For example
```
poetry run python path/to/script.py
poetry run python -m multinav train ...
```

## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

Please look at the `tox.ini` file for the full list of supported commands. 

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

TBD

## Authors

- [Roberto Cipollone](https://github.com/cipollone)
- [Marco Favorito](https://marcofavorito.me/)
