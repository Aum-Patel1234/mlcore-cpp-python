#include "logistic_regression.h"
#include "common.hpp"
#include <cmath>
#include <pybind11/pytypes.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
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

  auto print_iter = [&](int it, const xt::xarray<double> &z,
                        const xt::xarray<double> &a,
                        const xt::xarray<double> &diff,
                        const xt::xarray<double> &dw, double db) {
    std::cout << "=== iter " << it << " ===\n";
    std::cout << "z: " << z << "\n";
    std::cout << "a: " << a << "\n";
    std::cout << "diff: " << diff << "\n";
    std::cout << "dw: " << dw << "\n";
    std::cout << "db: " << db << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "bias: " << this->bias << "\n";
    std::cout << "---------------------\n";
  };

  for (int _ = 0; _ < iterations; _++) {
    // z: (m,1)
    auto z = xt::linalg::dot(this->X, theta) + this->bias;
    // a= (m,1)
    auto a = 1.0 / (1.0 + xt::exp(-z));
    // diff= m,1
    auto diff = a - this->y;

    // dw=(n,1) -> (n,m)*(m,1)
    auto dw =
        xt::linalg::dot(xt::transpose(this->X), diff) / static_cast<double>(m);

    auto db_arr = xt::sum(diff) / static_cast<double>(m);
    double db = db_arr(0);

    if (_ == 0 || _ == iterations - 1) {
      print_iter(_, z, a, diff, dw, db);
    }

    theta -= alpha * dw;
    bias -= alpha * db;
  }
}

py::array_t<double> LogisticRegression::predict(py::array_t<double> &X) {
  return py::none();
}

void LogisticRegression::printSlopeIntercept() const {
  std::cout << "Weights (theta): " << this->theta << std::endl;
  std::cout << "Bias (intercept): " << this->bias << std::endl;
}
