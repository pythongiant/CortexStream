#!/bin/bash
# Quick start guide for building and testing CortexStream

set -e

echo "=== CortexStream Build Guide ==="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Configure
echo -e "${BLUE}[1/4] Configuring CMake...${NC}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Step 2: Build
echo -e "${BLUE}[2/4] Building project...${NC}"
make -j$(nproc)

# Step 3: Run example
echo -e "${BLUE}[3/4] Building example...${NC}"
# Note: This assumes example executables are built
# ./examples/simple_inference

# Step 4: Summary
echo -e "${GREEN}[4/4] Build complete!${NC}"
echo ""
echo "=== Files Created ==="
ls -la bin/ 2>/dev/null || echo "  (No executables yet - example compilation needs MLX)"

echo ""
echo "=== Next Steps ==="
echo "1. Install MLX: pip install mlx"
echo "2. Update CMakeLists.txt with MLX paths"
echo "3. Rebuild: cd build && cmake .. && make"
echo "4. Run example: ./examples/simple_inference"
echo ""

cd ..
