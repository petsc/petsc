// 
// File:          TOPS_StructuredMatrix_Impl.hxx
// Symbol:        TOPS.StructuredMatrix-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.StructuredMatrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_TOPS_StructuredMatrix_Impl_hxx
#define included_TOPS_StructuredMatrix_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_TOPS_StructuredMatrix_IOR_h
#include "TOPS_StructuredMatrix_IOR.h"
#endif
#ifndef included_TOPS_Structured_Matrix_hxx
#include "TOPS_Structured_Matrix.hxx"
#endif
#ifndef included_TOPS_StructuredMatrix_hxx
#include "TOPS_StructuredMatrix.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._includes)
// Insert-Code-Here {TOPS.StructuredMatrix._includes} (includes or arbitrary code)
#if defined(HAVE_LONG_LONG)
#undef HAVE_LONG_LONG
#endif
#include "TOPS.hxx"
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.StructuredMatrix" (version 0.0.0)
   */
  class StructuredMatrix_impl : public virtual ::TOPS::StructuredMatrix 
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._inherits)
  // Insert-Code-Here {TOPS.StructuredMatrix._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._implementation)
    // Insert-Code-Here {TOPS.StructuredMatrix._implementation} (additional details)
      int vlength[4],vlower[4],vdimen;
      int gghostlower[4],gghostlength[4];
      Mat mat;
      ::gov::cca::Services myServices;
    // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._implementation)

  public:
    // default constructor, used for data wrapping(required)
    StructuredMatrix_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    StructuredMatrix_impl( struct TOPS_StructuredMatrix__object * s ) : 
      StubBase(s,true), _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~StructuredMatrix_impl() { _dtor(); }

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
    getDimen_impl() ;
    /**
     * user defined non-static method.
     */
    int32_t
    getLower_impl (
      /* in */int32_t dimen
    )
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getLength_impl (
      /* in */int32_t dimen
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setDimen_impl (
      /* in */int32_t dim
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setLower_impl (
      /* in array<int,3> */::sidl::array<int32_t> values
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setLength_impl (
      /* in array<int,3> */::sidl::array<int32_t> values
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setGhostLower_impl (
      /* in array<int,3> */::sidl::array<int32_t> values
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setGhostLength_impl (
      /* in array<int,3> */::sidl::array<int32_t> values
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setMat_impl (
      /* in */void* m
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

    /**
     *  Starts up a component presence in the calling framework.
     * @param services the component instance's handle on the framework world.
     * Contracts concerning Svc and setServices:
     * 
     * The component interaction with the CCA framework
     * and Ports begins on the call to setServices by the framework.
     * 
     * This function is called exactly once for each instance created
     * by the framework.
     * 
     * The argument Svc will never be nil/null.
     * 
     * Those uses ports which are automatically connected by the framework
     * (so-called service-ports) may be obtained via getPort during
     * setServices.
     */
    void
    setServices_impl (
      /* in */::gov::cca::Services services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class StructuredMatrix_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._misc)
// Insert-Code-Here {TOPS.StructuredMatrix._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._misc)

#endif
