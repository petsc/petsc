// 
// File:          src/BS/Error_Impl/BS_Error_Impl.hh
// Symbol:        BS.Error-v3.0
// Symbol Type:   class
// Babel Version: 0.6.1
// Description:   Server-side implementation for BS.Error
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_BS_Error_Impl_hh
#define included_BS_Error_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_BS_Error_IOR_h
#include "BS_Error_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_BS_Error_hh
#include "src/BS/Error/BS_Error.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "src/SIDL/BaseInterface/SIDL_BaseInterface.hh"
#endif


// DO-NOT-DELETE splicer.begin(BS.Error._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(BS.Error._includes)

namespace BS { 

  /**
   * Symbol "BS.Error" (version 3.0)
   */
  class Error_impl {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Error self;

    // DO-NOT-DELETE splicer.begin(BS.Error._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(BS.Error._implementation)

  private:
    // private default constructor (required)
    Error_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Error_impl( struct BS_Error__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Error_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

  };  // end class Error_impl

} // end namespace BS

// DO-NOT-DELETE splicer.begin(BS.Error._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(BS.Error._misc)

#endif
