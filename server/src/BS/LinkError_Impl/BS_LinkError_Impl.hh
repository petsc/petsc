// 
// File:          src/BS/LinkError_Impl/BS_LinkError_Impl.hh
// Symbol:        BS.LinkError-v3.0
// Symbol Type:   class
// Babel Version: 0.6.1
// Description:   Server-side implementation for BS.LinkError
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_BS_LinkError_Impl_hh
#define included_BS_LinkError_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_BS_LinkError_IOR_h
#include "BS_LinkError_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_BS_LinkError_hh
#include "src/BS/LinkError/BS_LinkError.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "src/SIDL/BaseInterface/SIDL_BaseInterface.hh"
#endif


// DO-NOT-DELETE splicer.begin(BS.LinkError._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(BS.LinkError._includes)

namespace BS { 

  /**
   * Symbol "BS.LinkError" (version 3.0)
   */
  class LinkError_impl {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    LinkError self;

    // DO-NOT-DELETE splicer.begin(BS.LinkError._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(BS.LinkError._implementation)

  private:
    // private default constructor (required)
    LinkError_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    LinkError_impl( struct BS_LinkError__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~LinkError_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

  };  // end class LinkError_impl

} // end namespace BS

// DO-NOT-DELETE splicer.begin(BS.LinkError._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(BS.LinkError._misc)

#endif
