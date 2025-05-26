#pragma once

#include <cstddef>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#ifndef NDARRAY_H
#define NDARRAY_H

template <typename T> class Ndarray {
private:
  T *arr;
  std::size_t arr_size;

public:
  Ndarray(py::array_t<T> numpy_array);

  // use const as it does not change the state of the object
  std::size_t size() const;
  std::string dtype() const;
};

#endif // !NDARRAY_H
