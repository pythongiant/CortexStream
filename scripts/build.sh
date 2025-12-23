#!/bin/bash
# Build script for CortexStream

set -e

mkdir -p build
cd build
cmake ..
make -j$(nproc)
