// 
// File:          TOPS_UnstructuredSolver_Impl.hxx
// Symbol:        TOPS.UnstructuredSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.UnstructuredSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_TOPS_UnstructuredSolver_Impl_hxx
#define included_TOPS_UnstructuredSolver_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_TOPS_UnstructuredSolver_IOR_h
#include "TOPS_UnstructuredSolver_IOR.h"
#endif
#ifndef included_TOPS_Unstructured_Solver_hxx
#include "TOPS_Unstructured_Solver.hxx"
#endif
#ifndef included_TOPS_UnstructuredSolver_hxx
#include "TOPS_UnstructuredSolver.hxx"
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


// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._includes)
#if defined(HAVE_LONG_LONG)
#undef HAVE_LONG_LONG
#endif
#include "TOPS.hxx"
#include "petscdmmg.h"
#include "gov_cca_ports_ParameterPortFactory.hxx"
#include "gov_cca_ports_ParameterPort.hxx"
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.UnstructuredSolver" (version 0.0.0)
   */
  class UnstructuredSolver_impl : public virtual ::TOPS::UnstructuredSolver 
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._inherits)
  // Insert-Code-Here {TOPS.UnstructuredSolver._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._implementation)
    DMMG                                  *dmmg;
    ::Sliced                              slice;
    int                                   startedpetsc;
    gov::cca::Services                    myServices;
    int                                   bs,n,Nghosted;
    gov::cca::ports::ParameterPortFactory ppf;
    gov::cca::ports::ParameterPort        params;

    int setupParameterPort();
    // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._implementation)

  public:
    // default constructor, used for data wrapping(required)
    UnstructuredSolver_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    UnstructuredSolver_impl( struct TOPS_UnstructuredSolver__object * s ) : 
      StubBase(s,true), _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~UnstructuredSolver_impl() { _dtor(); }

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
    void
    setLocalSize_impl (
      /* in */int32_t m
    )
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getLocalSize_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setGhostPoints_impl (
      /* in array<int> */::sidl::array<int32_t> ghosts
    )
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<int32_t>
    getGhostPoints_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setPreallocation_impl (
      /* in */int32_t d,
      /* in */int32_t od
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setPreallocation_impl (
      /* in array<int> */::sidl::array<int32_t> d,
      /* in array<int> */::sidl::array<int32_t> od
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

  };  // end class UnstructuredSolver_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._misc)
// Insert-Code-Here {TOPS.UnstructuredSolver._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._misc)

#endif
