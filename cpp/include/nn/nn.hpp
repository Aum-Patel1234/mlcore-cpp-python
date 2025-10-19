#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xmanipulation.hpp>

namespace py = pybind11;

#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

class NeuralNetwork2L {
private:
  xt::xarray<double> X;
  xt::xarray<double> y;
  xt::xarray<double> W1, b1, W2, b2;
  std::unordered_map<std::string, xt::xarray<double>> cache, grads;
  // will maintain values of Z1, A1, Z2, A2 during forward_propagation
  int iterations;
  double learning_rate;
  int n_h, n_x, n_y; // number of neurons in the hidden layer
  // xt::xarray<double> intercepts;

  void initialize_parameters(int n_x, int n_h, int n_y);
  void forward_propagation(const xt::xarray<double> &X);
  void backard_propagation();
  xt::xarray<double> sigmoid(const xt::xarray<double> &z);
  double cost(const xt::xarray<double> &A, const xt::xarray<double> &Y);

public:
  NeuralNetwork2L(py::array_t<double> &x, py::array_t<double> &y, int n_x,
                  int n_y, int iterations = 1000, double lr = 0.01,
                  int n_h = 4);

  void train();
  py::array_t<double> predict(py::array_t<double> &X);
};

#endif // NEURAL_NETWORK
