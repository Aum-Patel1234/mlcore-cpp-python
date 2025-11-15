#include "../../include/kmeans.hpp"

#include <xtensor/core/xmath.hpp>

#include "../../include/common.hpp"
#include "kmeans.hpp"

Kmeans::Kmeans(int k, int iterations) : k(k), iterations(iterations) {}

inline void Kmeans::minmaxScaling(xt::xarray<double>& x) {
  double mn = xt::amin(x)();
  double mx = xt::amax(x)();

  x = (x - mn) / (mx - mn) * 9.0 + 1.0;  // scale of 1-10
}

void Kmeans::fit(const py::array_t<double>& x_in) {
  helperOne(this->X, x_in);

  auto x = this->X;  // making a copy in case of scaling the input
  if (xt::amax(x)() > 10 || xt::amin(x)() <= 0) this->minmaxScaling(x);
}

void Kmeans::printCentroids() const {}

py::array_t<double> Kmeans::predict(py::array_t<double>& x_in) { return py::array_t<double>(); }
