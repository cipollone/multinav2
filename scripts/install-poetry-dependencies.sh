#!/usr/bin/env bash

set -e

function install_poetry_dependency(){
    current_directory="$(pwd)"
    cd "$1" && poetry build
    cd "${current_directory}"
    poetry run pip install --force-reinstall "$(find $1/dist -type f -name "*.whl")"
}

install_poetry_dependency third_party/temprl
install_poetry_dependency third_party/gym-sapientino

set +e
