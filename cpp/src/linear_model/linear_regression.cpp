#include <cstddef>
#include <pybind11/cast.h>
#include <sys/types.h>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/views/xstrided_view.hpp>
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

void LinearRegression::normalEquationFit() {
  // col for intercept
  auto ones_col =
      xt::ones<double>(std::vector<std::size_t>{this->X.shape()[0], 1});
  // std::cout << ones_col.shape()[0] << "," << ones_col.shape()[1] <<
  // std::endl;

  auto x = xt::hstack(xt::xtuple(this->X, ones_col));
  // std::cout << x.shape()[0] << "," << x.shape()[1] << std::endl;

  auto xT = xt::transpose(x);
  // std::cout << "xT - " << xT.shape()[0] << "," << xT.shape()[1] << std::endl;

  auto xTx = xt::linalg::dot(xT, x);
  // std::cout << "xTx - " << xTx.shape()[0] << "," << xTx.shape()[1] <<
  // std::endl;

  auto y_col = xt::reshape_view(
      this->y, std::vector<std::size_t>{this->y.shape()[0], 1});
  auto xTy = xt::linalg::dot(xT, y_col);
  // std::cout << "xTy - " << xTy.shape()[0] << "," << xTy.shape()[1] <<
  // std::endl;

  double lambda = 1e-5; // to avoid nan error
  auto I = xt::eye<double>(xTx.shape()[0]);
  auto theta = xt::linalg::dot(xt::linalg::inv(xTx + lambda * I), xTy);
  std::cout << "final ans for Normal eq - " << theta << std::endl;
}

void LinearRegression::printSlopeIntercept() const {
  std::cout << "Slope - " << this->slope << std::endl;
  std::cout << "Intercept - " << this->intercept;
  std::cout << std::endl;
}

py::array_t<double> LinearRegression::predict(py::array_t<double> test) {
  py::array prediction;
  return prediction;
}
