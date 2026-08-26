FROM python:3.12-slim

WORKDIR /app

# Pick up OS-level security patches (e.g. openssl) ahead of whatever the
# base image was last rebuilt with - caught by Trivy failing the CI
# security gate on a real, fixed-upstream CVE in libssl.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY sample_documents/ sample_documents/

RUN uv sync --frozen --no-dev

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
