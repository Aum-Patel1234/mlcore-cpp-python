#pragma once

#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
class BaseModel {
public:
  virtual ~BaseModel() = default;

  virtual void fit(const xt::xarray<double> &X,
                   const xt::xarray<double> &y) = 0;

  virtual xt::xarray<double> predict(const xt::xarray<double> &X) const = 0;

  virtual double score(const xt::xarray<double> &X,
                       const xt::xarray<double> &y) const = 0;
};

// virtual before method enables runtime polymorphism (i.e., the correct method
// gets called).
//
// = 0 after method makes it a pure virtual method — must be implemented by
// derived classes.
//
// virtual ~BaseModel() ensures destructors clean up correctly when using base
// class pointers.
