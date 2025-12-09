# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS build

ENV UV_PYTHON=python3.12 \
    UV_PROJECT_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY main.py .
COPY input.json ./input.json
COPY data/processed ./data/processed
COPY datasets ./datasets

FROM python:3.12-slim AS runtime

ENV UV_PROJECT_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src ./src
COPY --from=build /app/main.py .
COPY --from=build /app/input.json ./input.json
COPY --from=build /app/data ./data
COPY --from=build /app/datasets ./datasets

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
