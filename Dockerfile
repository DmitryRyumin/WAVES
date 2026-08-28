FROM python:3.14.7-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HOME=/home/user \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /app \
    && chown user:user /app \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

USER user

COPY --chown=user:user pyproject.toml uv.lock ./

RUN uv sync --locked --no-install-project

COPY --chown=user:user . .

EXPOSE 7860

CMD ["python", "app.py"]