#!/bin/bash

# index-graphrag.sh
# Script to index documents using GraphRAG
echo "Starting directory: $(pwd)"

# Change to the graphrag-10k directory
cd ./graphrag-10k/ || exit

echo "after cd directory: $(pwd)"

# Run the graphrag index command
graphrag index --root .

# Return the exit code
exit $?