#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <xtensor/core/xtensor_forward.hpp>

#include "../include/linear_regression.h"
#include "../include/logistic_regression.h"
#include "../include/regularization.hpp"
#include "ndarray/ndarray.h"
#include "nn/nn.hpp"
#include "xtensor_python_config.h"

#define BIND_NDARRAYWRAPPER(TYPE, NAME)                                \
  py::class_<NdarrayWrapper<TYPE>>(m, NAME)                            \
      .def(py::init<const py::array_t<TYPE>&>(), py::arg("input_arr")) \
      .def("size", &NdarrayWrapper<TYPE>::size)                        \
      .def("dtype", &NdarrayWrapper<TYPE>::dtype)                      \
      .def("ndim", &NdarrayWrapper<TYPE>::ndim)                        \
      .def("get", &NdarrayWrapper<TYPE>::get)                          \
      .def("cpp_forloop", &NdarrayWrapper<TYPE>::cpp_forloop)          \
      .def("getVec", &NdarrayWrapper<TYPE>::getVec)                    \
      .def("shape", &NdarrayWrapper<TYPE>::shape);

namespace py = pybind11;

template <typename T>
class NdarrayWrapper {
 private:
  Ndarray<T> arr;

 public:
  // directly construct arr with input_arr
  NdarrayWrapper(const py::array_t<T>& input_arr) : arr(input_arr) {
    if (!py::isinstance<py::array_t<T>>(input_arr))
      throw std::invalid_argument("incompatible datatype for this NdarrayWrapper");
  }
  std::size_t size() const { return this->arr.size(); }

  std::string dtype() const { return this->arr.dtype(); }

  int ndim() const { return this->arr.ndim(); }
  std::vector<int> shape() const { return this->arr.shape(); }
  std::vector<T> getVec() const { return this->arr.getVec(); }
  py::array_t<T> get() const { return this->arr.get(); }
  void cpp_forloop() const { this->arr.cpp_forloop(); }
};

PYBIND11_MODULE(mlcore_cpp, m) {
  m.doc() =
      "ML Core C++ bindings with NdarrayWrapper for common types.\n"
      "Supported dtypes:\n"
      "  int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,\n"
      "  int64_t, uint64_t, float, double, long double";

  namespace py = pybind11;

  BIND_NDARRAYWRAPPER(std::int8_t, "NdarrayWrapperInt8");
  BIND_NDARRAYWRAPPER(std::uint8_t, "NdarrayWrapperUInt8");
  BIND_NDARRAYWRAPPER(std::int16_t, "NdarrayWrapperInt16");
  BIND_NDARRAYWRAPPER(std::uint16_t, "NdarrayWrapperUInt16");
  BIND_NDARRAYWRAPPER(std::int32_t, "NdarrayWrapperInt32");
  BIND_NDARRAYWRAPPER(std::uint32_t, "NdarrayWrapperUInt32");
  BIND_NDARRAYWRAPPER(std::int64_t, "NdarrayWrapperInt64");
  BIND_NDARRAYWRAPPER(std::uint64_t, "NdarrayWrapperUInt64");
  BIND_NDARRAYWRAPPER(float, "NdarrayWrapperFloat");
  BIND_NDARRAYWRAPPER(double, "NdarrayWrapperDouble");
  BIND_NDARRAYWRAPPER(long double, "NdarrayWrapperLongDouble");

  py::class_<LinearRegression>(m, "LinearRegression")
      .def(py::init<py::array_t<double>, py::array_t<double>, int, double>(), py::arg("x"), py::arg("y"),
           py::arg("iterations") = 1000, py::arg("lr") = 0.01)
      .def("fit", &LinearRegression::fit)
      .def("normalEquationFit", &LinearRegression::normalEquationFit)
      .def("predict", &LinearRegression::predict)
      .def("printSlopeIntercept", &LinearRegression::printSlopeIntercept);

  py::class_<LogisticRegression>(m, "LogisticRegression")
      .def(py::init<double>(), py::arg("alpha") = 0.01)
      .def("fit", &LogisticRegression::fit, py::arg("x"), py::arg("y"), py::arg("iterations") = 1000)
      .def("predict", &LogisticRegression::predict, py::arg("X"))
      .def("printSlopeIntercept", &LogisticRegression::printSlopeIntercept);

  py::class_<NeuralNetwork2L>(m, "NeuralNetwork2L")
      .def(py::init<py::array_t<double>&, py::array_t<double>&, int, int, int, double, int>(), py::arg("x"),
           py::arg("y"), py::arg("n_x"), py::arg("n_y"), py::arg("iterations") = 1000, py::arg("learning_rate") = 0.01,
           py::arg("n_h") = 4)
      .def("train", &NeuralNetwork2L::train)
      .def("predict", &NeuralNetwork2L::predict, py::arg("X"));

  py::enum_<RegType>(m, "RegType")
      .value("L1", RegType::L1)
      .value("L2", RegType::L2)
      .value("Elastic", RegType::Elastic)
      .export_values();

  py::class_<Regularization>(m, "Regularization")
      .def(py::init<py::array_t<double>&,  // X
                    py::array_t<double>&,  // y
                    int,                   // iterations
                    RegType,               // reg type
                    double,                // learning_rate
                    double,                // lambda1
                    double                 // lambda2
                    >(),
           py::arg("X"), py::arg("y"), py::arg("iterations") = 1000, py::arg("type") = RegType::L2,
           py::arg("learning_rate") = 0.01, py::arg("lambda1") = 0.0, py::arg("lambda2") = 0.0)
      .def("fit", &Regularization::fit)
      .def("printCost", &Regularization::printCost, py::arg("y"), py::arg("y_pred"))
      .def("printSlopeIntercept", &Regularization::printSlopeIntercept)
      .def("predict", &Regularization::predict, py::arg("X"));
}
