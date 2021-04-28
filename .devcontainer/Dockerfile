# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.166.0/containers/python-3/.devcontainer/base.Dockerfile

ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

ARG POETRY_VERSION="1.1.6"
RUN pip3 install "poetry==$POETRY_VERSION"
