// 
// File:          src/BS/LinkCheckerI/Checker_Impl/openLibrary.cc
// Symbol:        BS.LinkCheckerI.Checker-v3.0
// Symbol Type:   class
// Babel Version: 0.6.1
// Description:   Server-side implementation for BS.LinkCheckerI.Checker
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "BS_LinkCheckerI_Checker_Impl.hh"

// DO-NOT-DELETE splicer.begin(BS.LinkCheckerI.Checker._includes)
#include <dlfcn.h>
// DO-NOT-DELETE splicer.end(BS.LinkCheckerI.Checker._includes)

// user defined non-static methods:
// referred to by:
//    BS.LinkError
void
BS::LinkCheckerI::Checker_impl::openLibrary (
  /*in*/ std::string fullpath ) 
throw ( 
  BS::LinkError
){
  // DO-NOT-DELETE splicer.begin(BS.LinkCheckerI.Checker.openLibrary)
  if (!dlopen(fullpath.c_str(), RTLD_LOCAL | RTLD_LAZY)) {
    std::string   errorMsg(dlerror());
    BS::LinkError error = BS::LinkError::_create();

    error.setMessage(errorMsg);
    throw error;
  }
  // DO-NOT-DELETE splicer.end(BS.LinkCheckerI.Checker.openLibrary)
}

