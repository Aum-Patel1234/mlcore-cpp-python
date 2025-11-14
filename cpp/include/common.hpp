#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace py = pybind11;

void helper(xt::xarray<double>& X, xt::xarray<double>& Y, const py::array_t<double>& x, const py::array_t<double>& y);

void helperOne(xt::xarray<double>& X, const py::array_t<double>& x);
