// 
// File:          TOPS_Structured_Matrix_Impl.hxx
// Symbol:        TOPS.Structured.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.Structured.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_TOPS_Structured_Matrix_Impl_hxx
#define included_TOPS_Structured_Matrix_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_TOPS_Structured_Matrix_IOR_h
#include "TOPS_Structured_Matrix_IOR.h"
#endif
#ifndef included_TOPS_Matrix_hxx
#include "TOPS_Matrix.hxx"
#endif
#ifndef included_TOPS_Structured_Matrix_hxx
#include "TOPS_Structured_Matrix.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._includes)
#if defined(HAVE_LONG_LONG)
#undef HAVE_LONG_LONG
#endif
#include "TOPS.hxx"
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._includes)

namespace TOPS { 
  namespace Structured { 

    /**
     * Symbol "TOPS.Structured.Matrix" (version 0.0.0)
     */
    class Matrix_impl : public virtual ::TOPS::Structured::Matrix 
    // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._inherits)
    // Insert-Code-Here {TOPS.Structured.Matrix._inherits} (optional inheritance here)
    // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._inherits)
    {

    // All data marked protected will be accessable by 
    // descendant Impl classes
    protected:

      bool _wrapped;

      // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._implementation)
    public: // not really public, but we make it public so TOPS::Solver::Structured can access directly
      int vlength[4],vlower[4],vdimen;
      int gghostlower[4],gghostlength[4];
      Mat mat;
      // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._implementation)

    public:
      // default constructor, used for data wrapping(required)
      Matrix_impl();
      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Matrix_impl( struct TOPS_Structured_Matrix__object * s ) : StubBase(s,
        true), _wrapped(false) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~Matrix_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // true if this object was created by a user newing the impl
      inline bool _isWrapped() {return _wrapped;}

      // static class initializer
      static void _load();

    public:

      /**
       * user defined non-static method.
       */
      int32_t
      dimen_impl() ;
      /**
       * user defined non-static method.
       */
      int32_t
      lower_impl (
        /* in */int32_t a
      )
      ;

      /**
       * user defined non-static method.
       */
      int32_t
      length_impl (
        /* in */int32_t a
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      set_impl (
        /* in */int32_t i,
        /* in array<double,2> */::sidl::array<double> values
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      set_impl (
        /* in */int32_t i,
        /* in */int32_t j,
        /* in array<double,2> */::sidl::array<double> values
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      set_impl (
        /* in */int32_t i,
        /* in */int32_t j,
        /* in */int32_t k,
        /* in array<double,2> */::sidl::array<double> values
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      set_impl (
        /* in */int32_t i,
        /* in */int32_t j,
        /* in */int32_t k,
        /* in */int32_t l,
        /* in array<double,2> */::sidl::array<double> values
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      apply_impl (
        /* in array<double> */::sidl::array<double> x,
        /* in array<double> */::sidl::array<double> y
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      zero_impl() ;
    };  // end class Matrix_impl

  } // end namespace Structured
} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._misc)
// Insert-Code-Here {TOPS.Structured.Matrix._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._misc)

#endif
