// 
// File:          TOPS_State_Impl.hh
// Symbol:        TOPS.State-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for TOPS.State
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_TOPS_State_Impl_hh
#define included_TOPS_State_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_State_IOR_h
#include "TOPS_State_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_State_hh
#include "TOPS_State.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.State._includes)
// Insert-Code-Here {TOPS.State._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(TOPS.State._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.State" (version 0.0.0)
   */
  class State_impl
  // DO-NOT-DELETE splicer.begin(TOPS.State._inherits)
  // Insert-Code-Here {TOPS.State._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.State._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    State self;

    // DO-NOT-DELETE splicer.begin(TOPS.State._implementation)
    // Insert-Code-Here {TOPS.State._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(TOPS.State._implementation)

  private:
    // private default constructor (required)
    State_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    State_impl( struct TOPS_State__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~State_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    Initialize (
      /* in */ ::sidl::array< ::std::string> args
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    Finalize() throw () 
    ;
  };  // end class State_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.State._misc)
// Insert-Code-Here {TOPS.State._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.State._misc)

#endif
