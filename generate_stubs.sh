#!/bin/bash
# set -x

if [ -f ".env" ]; then
  source .env
else
  echo "ERROR: .env file not found!"
  exit 1
fi

cd "$BUILD_DIR" || exit 1

# Generate stub
PYTHONPATH=. pybind11-stubgen mlcore_cpp --output-dir "$STUB_OUT" || exit 1

# Move stub to BUILD_DIR
mv "$STUB_OUT/mlcore_cpp.pyi" "$BUILD_DIR/mlcore_cpp.pyi" || exit 1

echo "✅ Stub generation complete and moved to build directory."
