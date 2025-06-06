#include <pybind11/cast.h>
#include <sys/types.h>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#define FORCE_IMPORT_ARRAY

#include "linear_regression.h"
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>

namespace py = pybind11;

LinearRegression::LinearRegression(py::array_t<double> x, py::array_t<double> y,
                                   int iterations = 1000, double lr = 0.01) {
  try {
    // std::cout << "here\n";
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();

    std::vector<std::size_t> x_shape(x_buf.shape.begin(), x_buf.shape.end());
    std::vector<std::size_t> x_strides(x_buf.strides.begin(),
                                       x_buf.strides.end());

    std::vector<std::size_t> y_shape(y_buf.shape.begin(), y_buf.shape.end());
    std::vector<std::size_t> y_strides(y_buf.strides.begin(),
                                       y_buf.strides.end());

    // Correct call with 6 parameters: (ptr, size, ownership, shape, strides)
    this->X = xt::adapt(static_cast<double *>(x_buf.ptr),
                        x_buf.size, // total number of elements
                        xt::no_ownership(), x_shape, x_strides);

    this->y = xt::adapt(static_cast<double *>(y_buf.ptr), y_buf.size,
                        xt::no_ownership(), y_shape, y_strides);

    // std::cout << "here 2\n";
    if (this->X.dimension() != 2)
      throw std::invalid_argument(
          "X must be a 2D array: [n_samples, n_features]");

    if (this->y.dimension() == 2 && this->y.shape()[1] != 1)
      throw std::invalid_argument("y must be 1D or 2D column vector");

    if (this->y.shape()[0] != this->X.shape()[0])
      throw std::invalid_argument(
          "X and y must have the same number of rows (samples)");

    this->iterations = iterations;
    this->learning_rate = lr;
    size_t n_samples = this->X.shape()[0];
    size_t n_features = this->X.shape()[1];
    this->slope = xt::zeros<double>({this->X.shape()[1]});
    this->intercept = 0;
  } catch (const std::exception &e) {
    throw std::invalid_argument("Array can't be converted to double.");
  }
}

void LinearRegression::fit() {
  auto n = this->X.shape()[0];
  for (int i = 0; i < this->iterations; i++) {
    // auto slope_gradient =
    //     -2 * this->X * (this->y - (this->slope * this->X +
    //     this->intercepts));
    auto y_predicted = xt::linalg::dot(this->X, this->slope) + this->intercept;
    auto error = this->y - y_predicted;

    xt::xarray<double> slope_gradient =
        -2 * xt::linalg::dot(xt::transpose(this->X), error) / n;
    // auto slope_gradient = -2 * xt::linalg::dot(this->X, this->y -
    // y_predicted);
    double intercept_gradient = (-2.0 / n) * xt::sum(error)();

    // auto intercept_gradient =
    //     -2 * (this->y - (this->slope * this->X + this->intercepts));

    this->slope -= slope_gradient * this->learning_rate;
    this->intercept -= intercept_gradient * this->learning_rate;
    if (i % 500 == 0) { // Print every 10% of iterations
      double loss = xt::mean(xt::pow(error, 2))();
      std::cout << "Iteration " << i << ", Loss: " << loss << std::endl;
      std::cout << "intercept " << this->intercept << std::endl;
    }
  }
  std::cout << "Ran for iterations = " << this->iterations << std::endl;
}

// void LinearRegression::fit() {
//   for (int i = 0; i < this->iterations; i++) {
//     // y_pred = dot(X, slope) + intercept
//     xt::xarray<double> y_pred = xt::linalg::dot(this->X, this->slope) +
//     this->intercepts;
//
//     // error = y - y_pred
//     xt::xarray<double> error = this->y - y_pred;
//
//     // gradient w.r.t slope: -2 * X^T * error / n
//     xt::xarray<double> slope_gradient =
//         -2.0 * xt::linalg::dot(xt::transpose(this->X), error) /
//         this->X.shape()[0];
//
//     // gradient w.r.t intercept: -2 * mean(error)
//     xt::xarray<double> intercept_gradient =
//         -2.0 * xt::mean(error);
//
//     // update parameters
//     this->slope -= this->learning_rate * slope_gradient;
//     this->intercepts -= this->learning_rate * intercept_gradient;
//   }
//
//   std::cout << "Ran for iterations = " << this->iterations << std::endl;
// }

void LinearRegression::normalEquationFit() {}

void LinearRegression::printSlopeIntercept() const {
  std::cout << "Slope - " << this->slope << std::endl;
  std::cout << "Intercept - " << this->intercept;
  std::cout << std::endl;
}

py::array LinearRegression::predict(py::array test) {
  py::array prediction;
  return prediction;
}
