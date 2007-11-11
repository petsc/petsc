#include <boost/preprocessor/repetition/enum_params.hpp>

#ifndef TINY_MAX_SIZE
#  define TINY_MAX_SIZE 3  // default maximum size is 3
#endif

template <BOOST_PP_ENUM_PARAMS(1, class T)>
struct tiny_size
  : mpl::int_<3>
{};
