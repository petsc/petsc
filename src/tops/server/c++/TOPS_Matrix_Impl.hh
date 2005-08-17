// 
// File:          TOPS_Matrix_Impl.hh
// Symbol:        TOPS.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 

#ifndef included_TOPS_Matrix_Impl_hh
#define included_TOPS_Matrix_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_Matrix_IOR_h
#include "TOPS_Matrix_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Matrix_hh
#include "TOPS_Matrix.hh"
#endif
#ifndef included_TOPS_Vector_hh
#include "TOPS_Vector.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.Matrix._includes)
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.Matrix._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.Matrix" (version 0.0.0)
   */
  class Matrix_impl
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix._inherits)
  // Insert-Code-Here {TOPS.Matrix._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Matrix self;

    // DO-NOT-DELETE splicer.begin(TOPS.Matrix._implementation)
    Mat mat;
    // DO-NOT-DELETE splicer.end(TOPS.Matrix._implementation)

  private:
    // private default constructor (required)
    Matrix_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Matrix_impl( struct TOPS_Matrix__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Matrix_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    apply (
      /* in */ ::TOPS::Vector x,
      /* in */ ::TOPS::Vector y
    )
    throw () 
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
    void
    set (
      /* in */ int32_t i,
      /* in */ int32_t j,
      /* in */ ::sidl::array<double> values
    )
    throw () 
    ;

  };  // end class Matrix_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Matrix._misc)
// Insert-Code-Here {TOPS.Matrix._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Matrix._misc)

#endif
