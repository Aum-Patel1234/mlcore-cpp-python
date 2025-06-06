// #define XTENSOR_PYTHON_USE_DYNAMIC_API 0
#define FORCE_IMPORT_ARRAY
#include "linear_regression.h"
#include <iostream>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>

namespace py = pybind11;

LinearRegression::LinearRegression(py::array numpy_array, int iterations = 1000,
                                   double lr = 0.01) {
  try {
    py::array_t<double> arr = py::cast<py::array_t<double>>(numpy_array);
    this->X = xt::pyarray<double>(arr);
    this->iterations = iterations;
    this->learning_rate = lr;
    this->slope = xt::zeros<double>({arr.shape()[0]});
    this->intercepts = xt::zeros<double>({arr.shape()[0]});
  } catch (const std::exception &e) {
    throw std::invalid_argument("Array can't be converted to double.");
  }
}

void LinearRegression::fit() {
  for (int i = 0; i < this->iterations; i++) {
    auto slope_gradient =
        -2 * this->X * (this->y - (this->slope * this->X + this->intercepts));
    auto intercept_gradient =
        -2 * (this->y - (this->slope * this->X + this->intercepts));

    this->slope -= slope_gradient * this->learning_rate;
    this->intercepts -= intercept_gradient * this->learning_rate;
  }
  std::cout << "Ran for iterations = " << this->iterations << std::endl;
}

void LinearRegression::normalEquationFit() {}

void LinearRegression::printSlopeIntercept() const {
  std::cout << "Slopes - ";
  for (const auto num : this->slope) {
    std::cout << num << "\t";
  }
  std::cout << std::endl << "Intercepts - ";
  for (const auto num : this->intercepts) {
    std::cout << num << "\t";
  }
  std::cout << std::endl;
}

py::array LinearRegression::predict(py::array test) {
  py::array prediction;
  return prediction;
}
