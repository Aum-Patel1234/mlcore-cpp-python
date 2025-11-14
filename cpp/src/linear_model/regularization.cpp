#include "../../include/regularization.hpp"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xtensor_forward.hpp>

#include "../../include/common.hpp"

Regularization::Regularization(py::array_t<double>& X_in, py::array_t<double>& Y_in, const int iterations,
                               const RegType type, const double lr, const double lambda1, const double lambda2)
    : iterations(iterations), learning_rate(lr), reg_type(type), lambda1(lambda1), lambda2(lambda2) {
  helper(this->X, this->y, X_in, Y_in);
  size_t sampleSize = this->X.shape()[0];
  if (this->y.shape()[0] != sampleSize) {
    throw std::invalid_argument("X and y must have the same number of samples");
  }
  size_t featureSize = this->X.shape()[1];

  this->slope = xt::zeros<double>({featureSize});
  this->intercept = 0;
}

void Regularization::printCost(const xt::xarray<double>& y, const xt::xarray<double>& y_pred) {
  const size_t m = this->X.shape()[0];
  double reg = 0;
  double mse = xt::sum(xt::square(y - y_pred))() / static_cast<double>(m);

  if (this->reg_type == RegType::L1)
    reg = xt::sum(this->lambda1 * xt::abs(this->slope))();  // l1 norm
  else if (this->reg_type == RegType::L2)
    reg = xt::sum(this->lambda2 * xt::square(this->slope))();  // l2 norm
  else
    reg = xt::sum(this->lambda1 * xt::abs(this->slope))() +
          xt::sum(this->lambda2 * xt::square(this->slope))();  // l1 + l2 norm

  std::cout << mse + reg << "\n";
}

void Regularization::fit() {
  const size_t n = this->X.shape()[0];

  for (int i = 0; i < this->iterations; i++) {
    xt::xarray<double> y_pred = xt::linalg::dot(this->X, this->slope) + this->intercept;
    xt::xarray<double> error = this->y - y_pred;

    xt::xarray<double> slope_grad = -2 / static_cast<double>(n) * (xt::linalg::dot(xt::transpose(this->X), error));
    double intercept_grad = -2 / static_cast<double>(n) * xt::sum(error)();
    auto sign_w = xt::where(this->slope > 0.0, 1.0, xt::where(this->slope < 0.0, -1.0, 0.0));

    if (this->reg_type == RegType::L1) {
      slope_grad += this->lambda1 * sign_w;
    } else if (this->reg_type == RegType::L2) {
      slope_grad += 2 * this->lambda2 * this->slope;
    } else {
      slope_grad += (this->lambda1 * sign_w + 2 * this->lambda2 * this->slope);
    }

    assert(this->slope.shape() == slope_grad.shape());
    this->slope -= this->learning_rate * slope_grad;
    this->intercept -= this->learning_rate * intercept_grad;

    if (i % 100 == 0) {
      std::cout << "Const of " << i << " iteration : ";
      this->printCost(this->y, y_pred);
    }
  }
}

py::array_t<double> Regularization::predict(py::array_t<double>& X) {
  xt::xarray<double> X_test;
  helperOne(X_test, X);
  xt::xarray<double> prediction = xt::linalg::dot(X_test, this->slope) + this->intercept;
  return py::array_t<double>(prediction.shape(), prediction.data());
}

void Regularization::printSlopeIntercept() const {
  std::cout << "=== Model Parameters ===\n";
  std::cout << "Slope (weights):\n" << this->slope << "\n";
  std::cout << "Intercept: " << this->intercept << "\n";
  std::cout << "========================\n\n";
}
