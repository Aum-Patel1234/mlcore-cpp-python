#include "common.hpp"

void helper(xt::xarray<double> &X, xt::xarray<double> &Y,
            py::array_t<double> &x, py::array_t<double> &y) {
  py::buffer_info x_buf = x.request();
  py::buffer_info y_buf = y.request();

  std::vector<std::size_t> x_shape(x_buf.shape.begin(), x_buf.shape.end());
  std::vector<std::size_t> y_shape(y_buf.shape.begin(), y_buf.shape.end());

  // convert byte strides → element strides
  std::vector<std::size_t> x_strides;
  for (auto s : x_buf.strides)
    x_strides.push_back(static_cast<std::size_t>(s) / sizeof(double));

  std::vector<std::size_t> y_strides;
  for (auto s : y_buf.strides)
    y_strides.push_back(static_cast<std::size_t>(s) / sizeof(double));

  // Correct call with strides in elements
  X = xt::adapt(static_cast<double *>(x_buf.ptr),
                x_buf.size,         // total number of elements
                xt::no_ownership(), // we don't own Python memory
                x_shape, x_strides);

  Y = xt::adapt(static_cast<double *>(y_buf.ptr), y_buf.size,
                xt::no_ownership(), y_shape, y_strides);
}
