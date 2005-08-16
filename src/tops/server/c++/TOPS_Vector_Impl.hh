// 
// File:          TOPS_Vector_Impl.hh
// Symbol:        TOPS.Vector-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Vector
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 

#ifndef included_TOPS_Vector_Impl_hh
#define included_TOPS_Vector_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_Vector_IOR_h
#include "TOPS_Vector_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Vector_hh
#include "TOPS_Vector.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.Vector._includes)
// Insert-Code-Here {TOPS.Vector._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(TOPS.Vector._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.Vector" (version 0.0.0)
   */
  class Vector_impl
  // DO-NOT-DELETE splicer.begin(TOPS.Vector._inherits)
  // Insert-Code-Here {TOPS.Vector._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.Vector._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Vector self;

    // DO-NOT-DELETE splicer.begin(TOPS.Vector._implementation)
    // Insert-Code-Here {TOPS.Vector._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(TOPS.Vector._implementation)

  private:
    // private default constructor (required)
    Vector_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Vector_impl( struct TOPS_Vector__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Vector_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    ::TOPS::Vector
    clone() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    zero() throw () 
    ;
    /**
     * user defined non-static method.
     */
    double
    dot (
      /* in */ ::TOPS::Vector y
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    double
    norm2() throw () 
    ;
    /**
     * user defined non-static method.
     */
    ::sidl::array<double>
    getArray() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    restoreArray() throw () 
    ;
  };  // end class Vector_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Vector._misc)
// Insert-Code-Here {TOPS.Vector._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Vector._misc)

#endif
