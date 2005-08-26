// 
// File:          TOPS_MatrixStructured_Impl.hh
// Symbol:        TOPS.MatrixStructured-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.MatrixStructured
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 

#ifndef included_TOPS_MatrixStructured_Impl_hh
#define included_TOPS_MatrixStructured_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_MatrixStructured_IOR_h
#include "TOPS_MatrixStructured_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_MatrixStructured_hh
#include "TOPS_MatrixStructured.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._includes)
#include "TOPS.hh"
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.MatrixStructured" (version 0.0.0)
   */
  class MatrixStructured_impl
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._inherits)
  // Insert-Code-Here {TOPS.MatrixStructured._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    MatrixStructured self;

    // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._implementation)
  public: // not really public, but we make it public so TOPS::Solver::Structured can access directly
    int vlength[4],vlower[4],vdimen;
    int gghostlower[4],gghostlength[4];
    Mat mat;
    // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._implementation)

  private:
    // private default constructor (required)
    MatrixStructured_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    MatrixStructured_impl( struct TOPS_MatrixStructured__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~MatrixStructured_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    int32_t
    dimen() throw () 
    ;
    /**
     * user defined non-static method.
     */
    int32_t
    lower (
      /* in */ int32_t a
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    length (
      /* in */ int32_t a
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    set (
      /* in */ int32_t i,
      /* in */ ::sidl::array<double> values
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    set (
      /* in */ int32_t i,
      /* in */ int32_t j,
      /* in */ ::sidl::array<double> values
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    set (
      /* in */ int32_t i,
      /* in */ int32_t j,
      /* in */ int32_t k,
      /* in */ ::sidl::array<double> values
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    set (
      /* in */ int32_t i,
      /* in */ int32_t j,
      /* in */ int32_t k,
      /* in */ int32_t l,
      /* in */ ::sidl::array<double> values
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    zero() throw () 
    ;
  };  // end class MatrixStructured_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._misc)
// Insert-Code-Here {TOPS.MatrixStructured._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._misc)

#endif
