#include "ndarray.h"
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

#define BIND_NDARRAYWRAPPER(TYPE, NAME)                                        \
  py::class_<NdarrayWrapper<TYPE>>(m, NAME)                                    \
      .def(py::init<const py::array_t<TYPE> &>(), py::arg("input_arr"))        \
      .def("size", &NdarrayWrapper<TYPE>::size)                                \
      .def("dtype", &NdarrayWrapper<TYPE>::dtype)                              \
      .def("ndim", &NdarrayWrapper<TYPE>::ndim)                                \
      .def("get", &NdarrayWrapper<TYPE>::get)                                  \
      .def("shape", &NdarrayWrapper<TYPE>::shape);

namespace py = pybind11;

template <typename T> class NdarrayWrapper {
private:
  Ndarray<T> arr;

public:
  // directly construct arr with input_arr
  NdarrayWrapper(const py::array_t<T> &input_arr) : arr(input_arr) {
    if (!py::isinstance<py::array_t<T>>(input_arr))
      throw std::invalid_argument(
          "incompatible datatype for this NdarrayWrapper");
  }
  std::size_t size() const { return this->arr.size(); }

  std::string dtype() const { return this->arr.dtype(); }

  int ndim() const { return this->arr.ndim(); }
  std::vector<int> shape() const { return this->arr.shape(); }
  std::vector<T> get() const { return this->arr.get(); }
};

PYBIND11_MODULE(mlcore_cpp, m) {
  m.doc() = "ML Core C++ bindings with NdarrayWrapper for common types.\n"
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
}
