#include "nn/nn.hpp"
#include "../../include/common.hpp"
#include <pybind11/cast.h>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/reducers/xreducer.hpp>

NeuralNetwork2L::NeuralNetwork2L(py::array_t<double> &x, py::array_t<double> &y,
                                 int n_x, int n_y, int iterations, double lr,
                                 int n_h) {
  this->n_h = n_h;
  this->n_x = n_x;
  this->n_y = n_y;
  try {
    // std::cout << "here\n";
    helper(this->X, this->y, x, y);

    this->iterations = iterations;
    this->learning_rate = lr;
  } catch (const std::exception &e) {
    throw std::invalid_argument("Array can't be converted to double.");
  }
}

void NeuralNetwork2L::initialize_parameters(int n_x, int n_h, int n_y) {
  // NOTE:
  // xarray of shape {n_h, n_x} with random values from a normal distribution
  // 0.0 is the mean, 1.0 is the standard deviation
  this->W1 = xt::random::randn<double>({n_h, n_x}, 0, 1) * 0.01;
  this->b1 = xt::zeros<double>({n_h, 1});
  this->W2 = xt::random::randn<double>({n_y, n_h}, 0, 1) * 0.01;
  this->b2 = xt::zeros<double>({n_y, 1});
}

double NeuralNetwork2L::cost(const xt::xarray<double> &A2,
                             const xt::xarray<double> &Y) {
  double m = static_cast<double>(Y.shape()[1]);
  xt::xarray<double> logProbs = Y * xt::log(A2) + (1 - Y) * xt::log(1 - A2);
  double cost = -xt::sum(logProbs)() / m;
  return cost;
}

xt::xarray<double> NeuralNetwork2L::sigmoid(const xt::xarray<double> &z) {
  return 1.0 / (1 + xt::exp(-z));
}

void NeuralNetwork2L::forward_propagation(const xt::xarray<double> &X) {
  xt::xarray<double> &W1 = this->W1, &W2 = this->W2, &b1 = this->b1,
                     &b2 = this->b2;

  auto Z1 = xt::linalg::dot(W1, X) + b1; // n_h, n_x * n_x, m -> (n_h, m)
  auto A1 = xt::tanh(Z1);

  auto Z2 = xt::linalg::dot(W2, A1) + b2; // n_y, n_h * n_h, m -> (n_y, m)
  auto A2 = sigmoid(Z2);

  this->cache["Z1"] = Z1;
  this->cache["A1"] = A1;
  this->cache["Z2"] = Z2;
  this->cache["A2"] = A2;
}

void NeuralNetwork2L::backard_propagation() {
  auto Z1 = this->cache["Z1"]; // (n_h, m)
  auto A1 = this->cache["A1"];
  auto Z2 = this->cache["Z2"]; // (n_y, m)
  auto A2 = this->cache["A2"];
  auto &Y = this->y, &W2 = this->W2;
  const double m = static_cast<double>(Y.shape()[1]);

  auto dZ2 = A2 - Y;
  // (n_y, m) * (m, n_h) -> (n_h, n_y)
  auto dW2 = (1.0 / m) * xt::linalg::dot(dZ2, xt::transpose(A1));
  auto db2 = (1.0 / m) * xt::sum(dZ2, {1}, xt::keep_dims);
  // NOTE:
  // i think true for keep_dims is not required. refer the docs below
  // https://xtensor.readthedocs.io/en/latest/api/reducing_functions.html#_CPPv4I000_N3xtl13check_conceptI18is_reducer_optionsI3EVSEEEEN2xt3sumEDaRR1E3EVS

  // tanh derivative: (1 - A1^2) && (n_h, n_y)*(n_y, m)+ (n_h, m)
  auto dZ1 = xt::linalg::dot(xt::transpose(W2), dZ2) * (1 - xt::pow(A1, 2));
  // (n_h, m)*(m, n_x) -> (n_h, n_x)
  auto dW1 = (1.0 / m) * xt::linalg::dot(dZ1, xt::transpose(this->X));
  auto db1 = (1.0 / m) * xt::sum(dZ1, {1}, xt::keep_dims);

  this->grads["dW1"] = dW1;
  this->grads["db1"] = db1;
  this->grads["dW2"] = dW2;
  this->grads["db2"] = db2;
}

void NeuralNetwork2L::train() {
  this->initialize_parameters(this->n_x, this->n_h, this->n_y);
  for (int i = 0; i < iterations; i++) {
    this->forward_propagation(this->X);
    this->backard_propagation();
    if (i % 100 == 0) {
      std::cout << std::endl;
      std::cout << "Cost at iteration " << i << " = "
                << this->cost(this->cache["A2"], this->y) << std::endl;
    }

    this->W1 -= grads["dW1"] * learning_rate;
    this->b1 -= grads["db1"] * learning_rate;
    this->W2 -= grads["dW2"] * learning_rate;
    this->b2 -= grads["db2"] * learning_rate;
  }
}

py::array_t<double> NeuralNetwork2L::predict(py::array_t<double> &X) {
  xt::xarray<double> X_local;
  helperOne(X_local, X);
  // std::cout << X_local.shape()[0] << std::endl;
  this->forward_propagation(X_local);
  xt::xarray<double> A2 = xt::eval(this->cache["A2"]);
  auto pred_expr = xt::where(A2 > 0.5, 1, 0);
  xt::xarray<double> predictions = xt::eval(pred_expr);

  // below is just converting from xarray to pyarray
  auto shape = predictions.shape();
  std::vector<std::size_t> dims(shape.begin(), shape.end());
  py::array_t<double> out(dims);

  auto buffer = out.request();
  double *ptr = static_cast<double *>(buffer.ptr);
  std::size_t idx = 0;
  for (auto it = predictions.begin(); it != predictions.end(); ++it) {
    ptr[idx++] = static_cast<double>(*it);
  }
  return out;
}
