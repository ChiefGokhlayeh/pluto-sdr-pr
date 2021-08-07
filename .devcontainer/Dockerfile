# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.166.0/containers/python-3/.devcontainer/base.Dockerfile

ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Should be created by preceding base-image
ARG USERNAME="vscode"
ARG MAMBAFORGE_VERSION="latest"
ARG MAMBAFORGE_PREFIX="/opt/mambaforge"
RUN curl -SL "https://github.com/conda-forge/miniforge/releases/${MAMBAFORGE_VERSION}/download/Mambaforge-"$(uname)"-"$(uname -m)".sh" -o /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p "${MAMBAFORGE_PREFIX}" \
    && "${MAMBAFORGE_PREFIX}/bin/mamba" init bash zsh \
    && chown -R "${USERNAME}" "${MAMBAFORGE_PREFIX}" \
    && rm -f /tmp/mambaforge.sh

USER "${USERNAME}"
ENV PATH="${MAMBAFORGE_PREFIX}/bin:${PATH}"
RUN "${MAMBAFORGE_PREFIX}/bin/mamba" init bash zsh

USER root

ARG CUDA_VERSION="11-4"
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    ffmpeg \
    software-properties-common \
    vim \
    && . /etc/os-release \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID}/"$(uname -m)"/ /" \
    && apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID}/"$(uname -m)"/7fa2af80.pub" \
    && add-apt-repository contrib \
    && apt-get update \
    && apt-get -y  --no-install-recommends install \
    "cuda-toolkit-${CUDA_VERSION}" \
    "cuda-tools-${CUDA_VERSION}" \
    libcuda1 \
    && rm -rf /var/lib/apt/lists/*
