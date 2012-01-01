// 
// File:          TOPS_StructuredSolver_Impl.hxx
// Symbol:        TOPS.StructuredSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.StructuredSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_TOPS_StructuredSolver_Impl_hxx
#define included_TOPS_StructuredSolver_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_TOPS_StructuredSolver_IOR_h
#include "TOPS_StructuredSolver_IOR.h"
#endif
#ifndef included_TOPS_Structured_Solver_hxx
#include "TOPS_Structured_Solver.hxx"
#endif
#ifndef included_TOPS_StructuredSolver_hxx
#include "TOPS_StructuredSolver.hxx"
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
#ifndef included_gov_cca_ports_ParameterGetListener_hxx
#include "gov_cca_ports_ParameterGetListener.hxx"
#endif
#ifndef included_gov_cca_ports_ParameterSetListener_hxx
#include "gov_cca_ports_ParameterSetListener.hxx"
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


// DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._includes)
#if defined(HAVE_LONG_LONG)
#undef HAVE_LONG_LONG
#endif
#include "petscdmmg.h"
#include "TOPS.hxx"
#include "gov_cca_ports_ParameterPortFactory.hxx"
#include "gov_cca_ports_ParameterPort.hxx"
// DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.StructuredSolver" (version 0.0.0)
   */
  class StructuredSolver_impl : public virtual ::TOPS::StructuredSolver 
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._inherits)
  // Insert-Code-Here {TOPS.StructuredSolver._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._implementation)
    DMMG                                  *dmmg;
    DA                                    da;
    int                                   lengths[4],m,n,p,dim,s,levels,bs;
    DAStencilType                         stencil_type;
    DAPeriodicType                        wrap;
    int                                   startedpetsc;
    gov::cca::Services                    myServices;
    gov::cca::ports::ParameterPortFactory ppf;
    gov::cca::ports::ParameterPort        params;

    int setupParameterPort();
    // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._implementation)

  public:
    // default constructor, used for data wrapping(required)
    StructuredSolver_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    StructuredSolver_impl( struct TOPS_StructuredSolver__object * s ) : 
      StubBase(s,true), _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~StructuredSolver_impl() { _dtor(); }

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
    ::gov::cca::Services
    getServices_impl() ;
    /**
     * user defined non-static method.
     */
    int32_t
    dimen_impl() ;
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
    setDimen_impl (
      /* in */int32_t dim
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setLength_impl (
      /* in */int32_t a,
      /* in */int32_t l
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setStencilWidth_impl (
      /* in */int32_t width
    )
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getStencilWidth_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setLevels_impl (
      /* in */int32_t levels
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    Initialize_impl() ;
    /**
     * user defined non-static method.
     */
    void
    solve_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setBlockSize_impl (
      /* in */int32_t bs
    )
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<double>
    getSolution_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setSolution_impl (
      /* in array<double> */::sidl::array<double> location
    )
    ;


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


    /**
     *  Inform the listener that someone is about to fetch their 
     * typemap. The return should be true if the listener
     * has changed the ParameterPort definitions.
     */
    bool
    updateParameterPort_impl (
      /* in */const ::std::string& portName
    )
    ;


    /**
     *  The component wishing to be told after a parameter is changed
     * implements this function.
     * @param portName the name of the port (typemap) on which the
     * value was set.
     * @param fieldName the name of the value in the typemap.
     */
    void
    updatedParameterValue_impl (
      /* in */const ::std::string& portName,
      /* in */const ::std::string& fieldName
    )
    ;

  };  // end class StructuredSolver_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._misc)
// Insert-Code-Here {TOPS.StructuredSolver._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._misc)

#endif
