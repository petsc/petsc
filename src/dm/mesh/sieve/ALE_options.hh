#ifndef included_ALE_options_hh
#define included_ALE_options_hh
// This should be included indirectly -- only by including ALE.hh

#define ALE_OPTIONS_SIZE BOOST_MPL_LIMIT_VECTOR_SIZE

namespace ALE {
  #undef  __CLASS__
  #define __CLASS__ "ALE::Options"
  template <typename OptionsSpecifierList_>
  class Options {
    
  };// class Options

} // namespace ALE


#endif
