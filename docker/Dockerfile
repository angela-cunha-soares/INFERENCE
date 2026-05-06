################################################################################
# bwb -- reproducible Bayesian Water Balance image
#
# Build:  docker build -f docker/Dockerfile -t bwb:0.1.0 .
# Run:    docker run --rm -v "$PWD/output:/app/output" bwb:0.1.0 bwb info
################################################################################

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for PyMC / numpy / netcdf4
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-dev \
        libnetcdf-dev \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY pyproject.toml ./
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        "numpy>=1.21" "pandas>=2.0" "scipy>=1.10" \
        "pymc>=5.0" "arviz>=0.14" \
        "matplotlib>=3.7" "seaborn>=0.12" \
        "joblib"

# Now copy the source and install the package itself
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY data_processed ./data_processed
RUN python -m pip install -e .

ENV BWB_OUTPUT_DIR=/app/output
RUN mkdir -p /app/output /app/figures

ENTRYPOINT ["bwb"]
CMD ["info"]
