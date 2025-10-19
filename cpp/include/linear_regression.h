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
  // xt::xarray<double> intercepts;
  double intercept;
  int iterations;
  double learning_rate;

public:
  LinearRegression(py::array_t<double> x, py::array_t<double> y, int iterations,
                   double learning_rate);

  void fit();
  void normalEquationFit();
  void printSlopeIntercept() const;
  py::array_t<double> predict(py::array_t<double> X);
};

#endif // !LINEAR_REGRESSION
