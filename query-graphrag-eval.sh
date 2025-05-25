#!/bin/bash
#
if [ $# -eq 0 ]; then
    echo "No question parameter provided."
    exit 1
fi

cd ./graphrag-eval/ || exit

graphrag query --root . --method "$2" --query "$1"
