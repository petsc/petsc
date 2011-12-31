// 
// File:          Ex4_System_Impl.hxx
// Symbol:        Ex4.System-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex4.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_Ex4_System_Impl_hxx
#define included_Ex4_System_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_Ex4_System_IOR_h
#include "Ex4_System_IOR.h"
#endif
#ifndef included_Ex4_System_hxx
#include "Ex4_System.hxx"
#endif
#ifndef included_TOPS_Matrix_hxx
#include "TOPS_Matrix.hxx"
#endif
#ifndef included_TOPS_System_Compute_Matrix_hxx
#include "TOPS_System_Compute_Matrix.hxx"
#endif
#ifndef included_TOPS_System_Compute_RightHandSide_hxx
#include "TOPS_System_Compute_RightHandSide.hxx"
#endif
#ifndef included_TOPS_System_Initialize_Once_hxx
#include "TOPS_System_Initialize_Once.hxx"
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
#ifndef included_gov_cca_ports_GoPort_hxx
#include "gov_cca_ports_GoPort.hxx"
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


// DO-NOT-DELETE splicer.begin(Ex4.System._includes)
#define MPICH_IGNORE_CXX_SEEK
#include "mpi.h"
#include "TOPS.hxx"
// DO-NOT-DELETE splicer.end(Ex4.System._includes)

namespace Ex4 { 

  /**
   * Symbol "Ex4.System" (version 0.0.0)
   */
  class System_impl : public virtual ::Ex4::System 
  // DO-NOT-DELETE splicer.begin(Ex4.System._inherits)
  // Insert-Code-Here {Ex4.System._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(Ex4.System._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(Ex4.System._implementation)
    TOPS::Unstructured::Solver solver;
    gov::cca::Services         myServices;
    int                        n;
    // DO-NOT-DELETE splicer.end(Ex4.System._implementation)

  public:
    // default constructor, used for data wrapping(required)
    System_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    System_impl( struct Ex4_System__object * s ) : StubBase(s,true), _wrapped(
      false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~System_impl() { _dtor(); }

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
    void
    computeMatrix_impl (
      /* in */::TOPS::Matrix J,
      /* in */::TOPS::Matrix B
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    initializeOnce_impl() ;
    /**
     * user defined non-static method.
     */
    void
    computeRightHandSide_impl (
      /* in array<double> */::sidl::array<double> b
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
     *  
     * Execute some encapsulated functionality on the component. 
     * Return 0 if ok, -1 if internal error but component may be 
     * used further, and -2 if error so severe that component cannot
     * be further used safely.
     */
    int32_t
    go_impl() ;
  };  // end class System_impl

} // end namespace Ex4

// DO-NOT-DELETE splicer.begin(Ex4.System._misc)
// Insert-Code-Here {Ex4.System._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(Ex4.System._misc)

#endif
