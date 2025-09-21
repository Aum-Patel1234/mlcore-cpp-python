# mlcore

A core Machine Learning library implemented in C++ with Python bindings via pybind11.

## Overview

This repository is designed as a learning resource to understand how to implement basic ML algorithms and utilities in C++, and expose them to Python for easy experimentation. It demonstrates:

- Writing efficient C++ code for ML tasks
- Creating Python bindings with pybind11
- Using CMake to build C++ extension modules for Python

## Prerequisites

- C++17 compatible compiler
- CMake >= 4.0
- Python 3.x
- pybind11 (can be installed via pip: `pip install pybind11`)

## Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/mlcore.git
cd mlcore
```

2. **Create a build directory**:

```bash
mkdir -p cpp/build
cd cpp/build
```

3. **Build the project using CMake**:

```bash
cmake ..
make
```

This will generate the Python bindings (`mlcore_cpp`) in the build directory.

4. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

5. **Use in Python**:

```python
import sys
sys.path.append("cpp/build")  # path to the compiled module

import mlcore_cpp

# Example usage
log_reg = mlcore_cpp.LogisticRegression(alpha=0.01)
```

## Notes

- You can use the provided alias to simplify building:

```bash
alias cmk='cd cpp/build && cmake .. && make'
cmk
```

- Environment variables (like build paths) are stored in `.env` to avoid hardcoding them in scripts.

- For development, you can regenerate Python stubs for IDE auto-completion:

```bash
bash generate_stubs.sh
```

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs, suggestions, or improvements.