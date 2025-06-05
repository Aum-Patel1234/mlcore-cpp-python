#pragma once

#include <cstddef>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <xtensor.hpp>
#include <xtensor/containers/xstorage.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace py = pybind11;
#ifndef NDARRAY_H
#define NDARRAY_H

template <typename T> class Ndarray {
private:
  std::vector<T> arr;
  xt::xarray<T> xarray;
  std::size_t arr_size;
  std::vector<int> shape_;

public:
  Ndarray(py::array_t<T> numpy_array);

  T *arrange(const std::vector<int> &shape);
  T *zeros(const std::vector<int> &ndim);

  // use const as it does not change the state of the object
  std::size_t size() const;
  int ndim() const;
  std::vector<int> shape() const;
  std::string dtype() const;
  py::array_t<T> get() const;
  std::vector<T> getVec() const;
  void cpp_forloop() const;
};

#endif // !NDARRAY_H
