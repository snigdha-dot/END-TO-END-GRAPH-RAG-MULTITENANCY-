# syntax=docker/dockerfile:1
# Multi-stage build: compilers stay in the builder, out of the runtime image.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./

# Build ARG so the ~2.5GB ML stack is opt-in:
#   docker build --build-arg INSTALL_ML=true .
ARG INSTALL_ML=false
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && if [ "$INSTALL_ML" = "true" ]; then \
         /opt/venv/bin/pip install --no-cache-dir -r requirements-ml.txt; \
       else \
         /opt/venv/bin/pip install --no-cache-dir -r requirements.txt; \
       fi


FROM python:3.11-slim AS runtime

# curl is needed by the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/

# Never run as root: a container escape should not land on uid 0.
USER appuser

EXPOSE 8000

# Readiness (not liveness): reports unhealthy when the database is unreachable,
# so an orchestrator stops routing traffic that is guaranteed to fail.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
