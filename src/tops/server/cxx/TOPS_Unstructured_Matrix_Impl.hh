// 
// File:          TOPS_Unstructured_Matrix_Impl.hh
// Symbol:        TOPS.Unstructured.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Server-side implementation for TOPS.Unstructured.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.12
// 

#ifndef included_TOPS_Unstructured_Matrix_Impl_hh
#define included_TOPS_Unstructured_Matrix_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_Unstructured_Matrix_IOR_h
#include "TOPS_Unstructured_Matrix_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Unstructured_Matrix_hh
#include "TOPS_Unstructured_Matrix.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._includes)
#include "TOPS.hh"
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._includes)

namespace TOPS { 
  namespace Unstructured { 

    /**
     * Symbol "TOPS.Unstructured.Matrix" (version 0.0.0)
     */
    class Matrix_impl
    // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._inherits)
    // Insert-Code-Here {TOPS.Unstructured.Matrix._inherits} (optional inheritance here)
    // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._inherits)
    {

    private:
      // Pointer back to IOR.
      // Use this to dispatch back through IOR vtable.
      Matrix self;

      // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._implementation)
    public:
      Mat mat;
      // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._implementation)

    private:
      // private default constructor (required)
      Matrix_impl() 
      {} 

    public:
      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Matrix_impl( struct TOPS_Unstructured_Matrix__object * s ) : self(s,
        true) { _ctor(); }

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
      set (
        /* in */ int32_t row,
        /* in */ int32_t column,
        /* in */ ::sidl::array<double> values
      )
      throw () 
      ;

      /**
       * user defined non-static method.
       */
      void
      set (
        /* in */ int32_t row,
        /* in */ ::sidl::array<int32_t> columns,
        /* in */ ::sidl::array<double> values
      )
      throw () 
      ;

      /**
       * user defined non-static method.
       */
      void
      set (
        /* in */ ::sidl::array<int32_t> rows,
        /* in */ int32_t column,
        /* in */ ::sidl::array<double> values
      )
      throw () 
      ;

      /**
       * user defined non-static method.
       */
      void
      set (
        /* in */ ::sidl::array<int32_t> rows,
        /* in */ ::sidl::array<int32_t> columns,
        /* in */ ::sidl::array<double> values
      )
      throw () 
      ;

      /**
       * user defined non-static method.
       */
      void
      apply (
        /* in */ ::sidl::array<double> x,
        /* in */ ::sidl::array<double> y
      )
      throw () 
      ;

      /**
       * user defined non-static method.
       */
      void
      zero() throw () 
      ;
    };  // end class Matrix_impl

  } // end namespace Unstructured
} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._misc)
// Insert-Code-Here {TOPS.Unstructured.Matrix._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._misc)

#endif
