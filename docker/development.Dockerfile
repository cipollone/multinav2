
FROM nvcr.io/nvidia/tensorflow:21.02-tf1-py3

	# python
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1
	# pip
ARG PIP_NO_CACHE_DIR=off
ARG PIP_DISABLE_PIP_VERSION_CHECK=on
ARG PIP_DEFAULT_TIMEOUT=100
	# poetry
ARG POETRY_NO_INTERACTION=1
ENV POETRY_VERSION=1.1.4
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false
	# install files
ARG PROJECT_TEMP_PATH="/opt/project-temp"

# Apt-get installs
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
	tmux htop git vim bash-completion

# Install poetry 
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Bug with non-bultin typing package
RUN pip uninstall -y typing

# Install dependencies
RUN mkdir ${PROJECT_TEMP_PATH}
WORKDIR ${PROJECT_TEMP_PATH}
COPY poetry.lock pyproject.toml ./
RUN poetry install 

# I won't copy the project as I expect a bind mount in home

# Entry point
ENTRYPOINT ["bash", "-l", "-c", "tmux"]
