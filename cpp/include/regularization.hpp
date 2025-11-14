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

namespace py = pybind11;

#ifndef REGULARIZATION
#define REGULARIZATION

enum class RegType { L1, L2, Elastic };

class Regularization {
 private:
  xt::xarray<double> X;
  xt::xarray<double> y;
  xt::xarray<double> slope;
  // xt::xarray<double> intercepts;
  RegType reg_type;
  double intercept;
  int iterations;
  double learning_rate, lambda1, lambda2;

 public:
  Regularization(py::array_t<double>& X_in, py::array_t<double>& Y_in, const int iterations, const RegType type,
                 const double lr, const double lambda1, const double lambda2);

  void fit();
  void printCost(const xt::xarray<double>& y, const xt::xarray<double>& y_pred);
  void printSlopeIntercept() const;
  py::array_t<double> predict(py::array_t<double>& X);
};

#endif  // !REGULARIZATION
