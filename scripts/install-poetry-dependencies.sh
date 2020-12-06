#!/usr/bin/env bash

set -e

function install_poetry_dependency(){
    current_directory="$(pwd)"
    cd "$1"
    poetry build
    pip install dist/*.whl
    cd "${current_directory}"
}

install_poetry_dependency third_party/temprl


set +e
