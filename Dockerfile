# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# Dependency layer first, so source edits do not invalidate the install cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer. uv installs the project editable, so a bind-mounted ./src
# during development is picked up without a rebuild.
COPY README.md ./
COPY src ./src

# Repo content the app reads at runtime, baked into the image:
#   portkey/  gateway configs (credential-free; the key is injected per request)
#   data/     seed CSVs and the document corpus
# Derived data - the LanceDB index and agent checkpoints - lives on the /data
# mount instead, so rebuilding the image never discards it.
COPY portkey ./portkey
COPY data ./data

RUN uv sync --frozen --no-dev

EXPOSE 8000

# No curl in the slim image - use the interpreter that is already here.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "agrirag.main:app", "--host", "0.0.0.0", "--port", "8000"]
