#pragma once

#include "../base_model.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace py = pybind11;

#ifndef LOGISITIC_REGRESSION
#define LOGISITIC_REGRESSION

class LogisticRegression : public BaseModel {
private:
  xt::xarray<double> X;
  xt::xarray<double> y;
  double alpha;

public:
  LogisticRegression(double alpha);

  void fit(py::array_t<double> x, py::array_t<double> y);
  void printSlopeIntercept() const;
  py::array_t<double> predict(py::array_t<double> X);
};

#endif // !LOGISITIC_REGRESSION
