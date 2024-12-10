FROM nvcr.io/nvidia/tensorflow:24.11-tf2-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    VENV_PATH="/app/.venv" \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git-all \
    wget
    # python3-pip \
    # python3-opencv

# Ensure Python 3.12 is the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Poetry
# RUN pip install poetry