#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#ifndef KMEANS
#define KMEANS

namespace py = pybind11;

class Kmeans {
 private:
  int k, iterations;
  xt::xarray<double> X, centroids;

  inline void minmaxScaling(xt::xarray<double>& x);

 public:
  Kmeans(int k, int iterations = 100);

  void fit(const py::array_t<double>& x_in);
  void printCentroids() const;
  py::array_t<double> predict(py::array_t<double>& x_in);
};

#endif  // !KMEANS
