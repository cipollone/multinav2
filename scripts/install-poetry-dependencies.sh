#!/usr/bin/env bash

set -e

function install_poetry_dependency(){
    current_directory="$(pwd)"
    cd "$1" && poetry build && pip install "$(find dist -type f -name "*.whl")"
    cd "${current_directory}"
}

install_poetry_dependency third_party/temprl
install_poetry_dependency third_party/gym-sapientino

set +e
