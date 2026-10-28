FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY sibyl/ ./sibyl/

RUN python -m pip install --no-cache-dir --upgrade "pip>=26.2.1" "setuptools>=83.0.0" \
    && pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 sibyl \
    && mkdir /data && chown sibyl:sibyl /data
USER sibyl
WORKDIR /data
ENV SIBYL_OUTPUT_ROOT=/data/reports

ENTRYPOINT ["sibyl-mcp"]
