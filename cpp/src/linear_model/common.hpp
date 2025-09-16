#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace py = pybind11;

void helper(xt::xarray<double> &X, xt::xarray<double> &Y,
            py::array_t<double> &x, py::array_t<double> &y);
