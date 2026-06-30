# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------
# Enterprise Digital Twin & Decision Intelligence Platform — production image.
# Multi-stage, deterministic, non-root, minimal runtime surface.
# ----------------------------------------------------------------------------

# ---- Stage 1: builder ------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build prerequisites, then remove apt lists for a smaller layer.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Runtime dependencies are installed into an isolated prefix for a clean copy
# later. Versions are constrained for reproducibility; for fully deterministic
# builds, supply a hash-pinned constraints file via --require-hashes.
RUN pip install --prefix=/install "numpy>=2.0,<3.0" "PyYAML>=6.0,<7.0"

# ---- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

# OCI metadata.
LABEL org.opencontainers.image.title="enterprise-digital-twin" \
      org.opencontainers.image.description="Enterprise Digital Twin & Decision Intelligence Platform" \
      org.opencontainers.image.licenses="Proprietary" \
      org.opencontainers.image.source="https://example.com/enterprise-digital-twin"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/scripts \
    APP_ENV=production \
    APP_PORT=8080 \
    HEALTH_ENDPOINT=http://127.0.0.1:8080/health

# Create an unprivileged user and group.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the installed dependency tree from the builder stage.
COPY --from=builder /install /usr/local

# Copy application code with deterministic ownership.
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app deployment/scripts/ ./deployment/scripts/
COPY --chown=app:app deployment/docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

# Drop privileges.
USER app:app

EXPOSE 8080

# Container-level health check uses the deterministic health checker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "/app/deployment/scripts/health_check.py", "--root", "/app", "--quiet"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]