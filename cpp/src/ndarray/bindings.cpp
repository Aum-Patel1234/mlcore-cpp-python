#include "ndarray.h"
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

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
};

PYBIND11_MODULE(mlcore_cpp, m) {
  m.doc() = "ML Core C++ bindings with NdarrayWrapper for common types.\n"
            "Supported dtypes:\n"
            "  int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,\n"
            "  int64_t, uint64_t, float, double, long double";

  // NdarrayWrapper instantiations for all types
  py::class_<NdarrayWrapper<std::int8_t>>(m, "NdarrayWrapperInt8")
      .def(py::init<const py::array_t<std::int8_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::int8_t>::size)
      .def("dtype", &NdarrayWrapper<std::int8_t>::dtype);

  py::class_<NdarrayWrapper<std::uint8_t>>(m, "NdarrayWrapperUInt8")
      .def(py::init<const py::array_t<std::uint8_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::uint8_t>::size)
      .def("dtype", &NdarrayWrapper<std::uint8_t>::dtype);

  py::class_<NdarrayWrapper<std::int16_t>>(m, "NdarrayWrapperInt16")
      .def(py::init<const py::array_t<std::int16_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::int16_t>::size)
      .def("dtype", &NdarrayWrapper<std::int16_t>::dtype);

  py::class_<NdarrayWrapper<std::uint16_t>>(m, "NdarrayWrapperUInt16")
      .def(py::init<const py::array_t<std::uint16_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::uint16_t>::size)
      .def("dtype", &NdarrayWrapper<std::uint16_t>::dtype);

  py::class_<NdarrayWrapper<std::int32_t>>(m, "NdarrayWrapperInt32")
      .def(py::init<const py::array_t<std::int32_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::int32_t>::size)
      .def("dtype", &NdarrayWrapper<std::int32_t>::dtype);

  py::class_<NdarrayWrapper<std::uint32_t>>(m, "NdarrayWrapperUInt32")
      .def(py::init<const py::array_t<std::uint32_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::uint32_t>::size)
      .def("dtype", &NdarrayWrapper<std::uint32_t>::dtype);

  py::class_<NdarrayWrapper<std::int64_t>>(m, "NdarrayWrapperInt64")
      .def(py::init<const py::array_t<std::int64_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::int64_t>::size)
      .def("dtype", &NdarrayWrapper<std::int64_t>::dtype);

  py::class_<NdarrayWrapper<std::uint64_t>>(m, "NdarrayWrapperUInt64")
      .def(py::init<const py::array_t<std::uint64_t> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<std::uint64_t>::size)
      .def("dtype", &NdarrayWrapper<std::uint64_t>::dtype);

  py::class_<NdarrayWrapper<float>>(m, "NdarrayWrapperFloat")
      .def(py::init<const py::array_t<float> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<float>::size)
      .def("dtype", &NdarrayWrapper<float>::dtype);

  py::class_<NdarrayWrapper<double>>(m, "NdarrayWrapperDouble")
      .def(py::init<const py::array_t<double> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<double>::size)
      .def("dtype", &NdarrayWrapper<double>::dtype);

  py::class_<NdarrayWrapper<long double>>(m, "NdarrayWrapperLongDouble")
      .def(py::init<const py::array_t<long double> &>(), py::arg("input_arr"))
      .def("size", &NdarrayWrapper<long double>::size)
      .def("dtype", &NdarrayWrapper<long double>::dtype);
}
