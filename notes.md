# xtensor notes — functions
---

## Overview / quick tips
- `xt::xarray<T>`: dynamic, N-dimensional array (used everywhere in your code).
- `xt::view(...)`: creates a **view** (no copy) into an array. Use assignment to a `xt::view(...)` to write into a target.
- `xt::eval(...)`: materializes an expression into a concrete array (forces evaluation / copy).
- Broadcasting: `xt::newaxis` (and shapes with `xt::all()`) are used to create axes so arithmetic broadcasts elementwise across the desired dimensions.
- Prefer using `xt::amax(xt::abs(a-b))` as a robust convergence check instead of relying on `operator!=` between arrays.

---

## `xt::xarray<T>`
**Purpose:** dynamic runtime-sized multi-dimensional array.  
**Key properties:** flexible shape, stores elements in contiguous memory by default.

**Example**
```cpp
xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}}; // shape (2,2)
std::cout << "shape: " << a.shape()[0] << "x" << a.shape()[1] << "\n";
