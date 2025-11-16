#include "../../include/kmeans.hpp"

#include <cassert>
#include <cstddef>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xeval.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>

#include "../../include/common.hpp"
#include "kmeans.hpp"

Kmeans::Kmeans(int k, int iterations) : k(k), iterations(iterations) {}

inline void Kmeans::minmaxScaling(xt::xarray<double>& x) {
  double mn = xt::amin(x)();
  double mx = xt::amax(x)();

  x = (x - mn) / (mx - mn) * 9.0 + 1.0;  // scale of 1-10
}

inline void Kmeans::intializeCentroids(xt::xarray<double>& x) {
  const int n_samples = x.shape()[0];
  const int n_features = x.shape()[1];

  auto idx = xt::random::randint<int>({k}, 0, n_samples);

  // IMPORTANT:
  // xt::keep(idx) - Use these exact indices — do fancy indexing.
  this->centroids = xt::eval(xt::view(x, xt::keep(idx), xt::all()));  // k, n_features

  assert(this->centroids.shape()[0] == k);
  assert(this->centroids.shape()[1] == n_features);
}

inline xt::xarray<double> Kmeans::findDistanceFromCentroid(const xt::xarray<double>& x) const {
  // FORMULA: : D[i, j] = sqrt( sum_over_f ( x[i,f] - centroids[j,f] )^2 )
  auto x_exp = xt::view(x, xt::all(), xt::newaxis(), xt::all());                // (n_samples, 1, n_features)
  auto c_exp = xt::view(this->centroids, xt::newaxis(), xt::all(), xt::all());  // (1, k, n_features)

  // Pairwise squared distances: (n_samples, k, n_features) -> sum over features -> (n_samples, k)
  auto sq = xt::square(x_exp - c_exp);
  auto sum_sq = xt::sum(sq, {2});  // sum across the features axis

  return xt::sqrt(sum_sq);
}

inline std::vector<int> Kmeans::findLabel(const xt::xarray<double>& distances) const {
  xt::xarray<std::size_t> labels = xt::eval(xt::argmin(distances, 1));
  return std::vector<int>(labels.begin(), labels.end());
}

void Kmeans::fit(const py::array_t<double>& x_in) {
  helperOne(this->X, x_in);

  auto x = xt::eval(this->X);  // making a copy in case of scaling the input
  if (xt::amax(x)() > 10 || xt::amin(x)() <= 0) this->minmaxScaling(x);
  assert(xt::amin(x)() > 0 && xt::amax(x)() <= 10);

  this->intializeCentroids(x);  // k,n_features

  xt::xarray<double> prevCentroids = xt::zeros<double>({0});
  for (int i = 0; i < this->iterations && prevCentroids != this->centroids; i++) {
    prevCentroids = xt::eval(this->centroids);

    auto distances = this->findDistanceFromCentroid(x);
    std::vector<int> predictedLabel = this->findLabel(distances);

    xt::xarray<double> new_centroids = xt::zeros<double>({(size_t)k, x.shape()[1]});

    for (int j = 0; j < k; j++) {
      std::vector<std::size_t> idx;
      idx.reserve(predictedLabel.size());

      for (std::size_t k = 0; k < predictedLabel.size(); k++) {
        if (predictedLabel[k] == j) {
          idx.push_back(k);
        }
      }

      if (!idx.empty()) {
        xt::xarray<double> rows = xt::eval(xt::view(x, xt::keep(idx), xt::all()));
        xt::xarray<double> centroid_j = xt::eval(xt::mean(rows, {0}));
        xt::view(new_centroids, j, xt::all()) = centroid_j;  // https://xtensor.readthedocs.io/en/latest/view.html
      }
    }

    this->centroids = new_centroids;

    if (xt::amax(xt::abs(prevCentroids - this->centroids))() < 1e-6) break;
  }
}

void Kmeans::printCentroids() const { std::cout << "Centroids:\n" << this->centroids << "\n"; }

py::array_t<int> Kmeans::predict(py::array_t<double>& x_in) {
  xt::xarray<double> X;
  helperOne(X, x_in);
  auto distances = this->findDistanceFromCentroid(X);
  auto labels = this->findLabel(distances);
  return py::array_t<int>(labels.size(), labels.data());
}
