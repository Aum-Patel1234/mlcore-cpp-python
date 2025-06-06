#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace py = pybind11;

#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

class LinearRegression {
private:
  xt::xarray<double> X;
  xt::xarray<double> y;
  xt::xarray<double> slope;
  xt::xarray<double> intercepts;
  int iterations;
  double learning_rate;

public:
  explicit LinearRegression(py::array numpy_array, int iterations,
                            double learning_rate);

  void fit();
  void normalEquationFit();
  void printSlopeIntercept() const;
  py::array predict(py::array X);
};

#endif // !LINEAR_REGRESSION
