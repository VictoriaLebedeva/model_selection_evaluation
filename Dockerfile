FROM 3.9.13-windowsservercore-ltsc2022

RUN pip install poetry
COPY ["poetry.lock", "pyproject.toml", "./"]

WORKDIR /src/cover_type_classifier
COPY . . 
RUN  poetry install
EXPOSE 5432

ENTRYPOINT [ "poetry run app" ]