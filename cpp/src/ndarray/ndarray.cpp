#include "ndarray.h"
#include <cstddef>
#include <cxxabi.h>
#include <pybind11/buffer_info.h>
#include <string>

namespace py = pybind11;

template <typename T> Ndarray<T>::Ndarray(py::array_t<T> numpy_array) {
  py::buffer_info buffer = numpy_array.request();
  this->arr = static_cast<T *>(buffer.ptr);
  this->arr_size = static_cast<std::size_t>(buffer.size);
}

template <typename T> std::size_t Ndarray<T>::size() const {
  return this->arr_size;
}

template <typename T> std::string Ndarray<T>::dtype() const {
  return typeid(T).name();
}

// IMPORTANT: as it helps in compile time
template class Ndarray<std::int8_t>;
template class Ndarray<std::uint8_t>;
template class Ndarray<std::int16_t>;
template class Ndarray<std::uint16_t>;
template class Ndarray<std::int32_t>;
template class Ndarray<std::uint32_t>;
template class Ndarray<std::int64_t>;
template class Ndarray<std::uint64_t>;
template class Ndarray<float>;
template class Ndarray<double>;
template class Ndarray<long double>;
