FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG INSTALL_TRAIN_DEPS=0

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
ENV WANDB_MODE=offline
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    fontconfig \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    git \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    vim \
    wget \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/JAMEL

COPY pyproject.toml README.md LICENSE ./
COPY jamel ./jamel
COPY scripts ./scripts
COPY shell ./shell
COPY configs ./configs
COPY docs ./docs
COPY tests ./tests
COPY examples ./examples
COPY third_party ./third_party
COPY env ./env

RUN python -m pip install --upgrade pip setuptools wheel uv \
    && uv pip install --system -e ".[dev]"

RUN if [[ "$INSTALL_TRAIN_DEPS" == "1" ]]; then \
      uv pip install --system -r third_party/verl-agent/requirements.txt; \
    else \
      uv pip install --system -e ".[train]"; \
    fi

RUN npm install -g monocart-coverage-reports monocart-locator

RUN python -m playwright install --with-deps chromium

ENV JAMEL_ROOT=/workspace/JAMEL
ENV VERL_AGENT_ROOT=/workspace/JAMEL/third_party/verl-agent
ENV SCALEWOB_ROOT=/workspace/JAMEL/env/browser_env/scalewob-env
ENV PYTHONPATH=/workspace/JAMEL:/workspace/JAMEL/third_party/verl-agent

CMD ["/bin/bash"]
