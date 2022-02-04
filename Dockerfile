# syntax=docker/dockerfile:1
FROM ubuntu:20.04 as build


################################################################
#### set up environment ####
####  source: https://stackoverflow.com/a/54763270/2434875
################################################################
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.10

WORKDIR /app


################################################################
#### install system-wide dependencies and tools ####
################################################################
RUN apt update
RUN apt install -y python3.8 python3-pip
RUN apt install -y curl
# RUN apt install -y build-essential curl 

# RUN pip install "poetry==$POETRY_VERSION"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.10/get-poetry.py | python3.8 -
RUN ln -s "$HOME/.poetry/bin/poetry" /usr/bin/poetry
RUN pip config set global.cache-dir false
RUN poetry config virtualenvs.create false 

# we will have to rebuild any time the lockfile/pyproject.toml/source change
COPY poetry.lock pyproject.toml /app/
COPY langbrainscore/ /app/langbrainscore


################################################################
#### build app using poetry; install using pip ####
################################################################
RUN poetry install --no-interaction --no-ansi 
# RUN poetry build -f wheel && \
#     pip install --no-deps . dist/*.whl && \
#     rm -rf dist *.egg-info


################################################################
#### cleanup after build finishes ####
################################################################
RUN apt remove -y git
RUN apt autoremove -y
RUN apt clean && rm -rf /var/lib/apt/lists/*


################################################################
#### set up entrypoint to use as standalone app ####
################################################################
