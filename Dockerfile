FROM python:3.8-slim-buster

RUN pip install poetry

WORKDIR "/fctclassifier"

COPY ["poetry.lock", "pyproject.toml", "./"]

RUN poetry install --no-interaction --no-ansi
COPY . .

EXPOSE 9696

ENTRYPOINT  ["poetry", "run", "app" ]

