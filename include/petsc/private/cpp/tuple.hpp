#ifndef PETSC_CPP_TUPLE_HPP
#define PETSC_CPP_TUPLE_HPP

#if defined(__cplusplus)
  #include <tuple>

namespace Petsc
{

namespace util
{

  #if PETSC_CPP_VERSION >= 14
using std::tuple_element_t;
  #else
template <std::size_t I, class T>
using tuple_element_t = typename std::tuple_element<I, T>::type;
  #endif

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_TUPLE_HPP
