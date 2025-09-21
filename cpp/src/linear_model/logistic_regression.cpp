#include "logistic_regression.h"
#include "common.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <pybind11/pytypes.h>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xmanipulation.hpp>
namespace py = pybind11;

LogisticRegression::LogisticRegression(double alpha) {
  this->alpha = alpha;
  this->bias = 0;
}

void LogisticRegression::fit(py::array_t<double> &x_arr,
                             py::array_t<double> &y_arr, int iterations) {
  // FORMULA: loss fn for logistic_regression:
  // L(a, y) = -[y.log(a) + (1-y)log(1-a)]
  // where a = sigmoid(z), z = wi.xi + b
  // dz = a-y,  dw = (a-y)*x , db = (a-y)

  helper(this->X, this->y, x_arr, y_arr);
  // this->X = xt::pyarray<double>(x_arr);
  // this->y = xt::pyarray<double>(y_arr);
  auto x_shape = X.shape();
  // auto y_shape = y.shape();
  int m = static_cast<int>(x_shape[0]), n = static_cast<int>(x_shape[1]);

  if (this->y.dimension() == 1 && this->y.shape(0) == m) {
    this->y = this->y.reshape({m, 1});
  } else if (this->y.shape(0) != m) {
    throw std::runtime_error(
        "Mismatch: number of samples in X and y does not match!");
  }
  std::cout << "Samples: " << m << ", Features: " << n << std::endl;

  this->theta = xt::zeros<double>({n, 1});
  // theta = std::make_unique<xt::xarray<double>>(xt::zeros<double>({n, 1}));

  // auto print_iter = [&](int it, const xt::xarray<double> &z,
  //                       const xt::xarray<double> &a,
  //                       const xt::xarray<double> &diff,
  //                       const xt::xarray<double> &dw, double db) {
  //   std::cout << "=== iter " << it << " ===\n";
  //   std::cout << "z: " << z << "\n";
  //   std::cout << "a: " << a << "\n";
  //   std::cout << "diff: " << diff << "\n";
  //   std::cout << "dw: " << dw << "\n";
  //   std::cout << "db: " << db << "\n";
  //   std::cout << "theta: " << this->theta << "\n";
  //   std::cout << "bias: " << this->bias << "\n";
  //   std::cout << "---------------------\n";
  // };

  for (int _ = 0; _ < iterations; _++) {
    // z: (m,1)
    auto z = xt::linalg::dot(this->X, theta) + this->bias;
    // a= (m,1)
    xt::xarray<double> a = 1.0 / (1.0 + xt::exp(-z));
    // diff= m,1
    auto diff = a - this->y;

    // dw=(n,1) -> (n,m)*(m,1)
    auto dw =
        xt::linalg::dot(xt::transpose(this->X), diff) / static_cast<double>(m);

    auto db_arr = xt::sum(diff) / static_cast<double>(m);
    double db = db_arr(0);

    if (_ % 20 == 0) {
      // print_iter(_, z, a, diff, dw, db);
      printCost(y, a, m, _);
    }

    theta -= alpha * dw;
    bias -= alpha * db;
  }
}

void LogisticRegression::printCost(xt::xarray<double> &y, xt::xarray<double> &a,
                                   int m, int i) const {
  // L(a, y) = 1/m* [y.log(a) + (1-y)log(1-a)]
  double inv_m = 1.0 / static_cast<double>(m);
  auto cost = -inv_m * xt::sum(y * xt::log(a) + (1 - y) * xt::log(1 - a));
  std::cout << "Cost (" << i << ") = " << cost(0) << std::endl;
}

py::array_t<double> LogisticRegression::predict(py::array_t<double> &X) {
  // theta - (n,1) , X - (m, n)
  if (this->theta.shape()[0] != X.shape()[1])
    throw std::runtime_error(
        "features of the input is not same as that of training data!!");

  xt::xarray<double> input;
  helperOne(input, X);
  // std::cout << "Input (shape: ";
  // for (auto s : input.shape())
  //   std::cout << s << " ";
  // std::cout << "):\n";
  //
  // std::size_t i = 0;
  // for (auto it = input.begin(); it != input.end(); ++it) {
  //   std::cout << *it << " ";
  //   i++;
  //   if (i % input.shape()[1] == 0)
  //     std::cout << "\n";  }
  // std::cout << std::endl;

  xt::xarray<double> z = xt::linalg::dot(input, this->theta) + this->bias;
  xt::xarray<double> prediction = 1.0 / (1 + xt::exp(-z));
  // xt::xarray<double> prediction = xt::where(sigmoid >= 0.5, 1, 0);

  auto shape = prediction.shape();
  std::vector<std::size_t> dims(shape.begin(), shape.end());
  py::array_t<double> out(dims);

  // copy into numpy buffer
  auto buffer = out.request();
  double *ptr = static_cast<double *>(buffer.ptr);
  std::size_t idx = 0;

  for (auto it = prediction.begin(); it != prediction.end(); it++)
    ptr[idx++] = static_cast<double>(*it);

  return out;
}

void LogisticRegression::printSlopeIntercept() const {
  std::cout << "Weights (theta): " << this->theta << std::endl;
  std::cout << "Bias (intercept): " << this->bias << std::endl;
}
