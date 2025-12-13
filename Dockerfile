# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS build

ENV UV_PYTHON=python3.12 \
    UV_PROJECT_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv using curl with retry mechanism
RUN curl -LsSf https://astral.sh/uv/install.sh | sh || \
    (sleep 5 && curl -LsSf https://astral.sh/uv/install.sh | sh) || \
    (sleep 10 && curl -LsSf https://astral.sh/uv/install.sh | sh)

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY main.py input.json bentoml_service.py ./
COPY data ./data
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
COPY --from=build /app/main.py /app/input.json /app/bentoml_service.py ./
COPY --from=build /app/data ./data
COPY --from=build /app/datasets ./datasets

ENV PATH="/app/.venv/bin:${PATH}"

# Default to BentoML service, can be overridden
ENTRYPOINT ["bentoml", "serve"]
CMD ["bentoml_service:CarPricePrediction", "--production"]
