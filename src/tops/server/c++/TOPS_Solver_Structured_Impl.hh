// 
// File:          TOPS_Solver_Structured_Impl.hh
// Symbol:        TOPS.Solver_Structured-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Solver_Structured
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 

#ifndef included_TOPS_Solver_Structured_Impl_hh
#define included_TOPS_Solver_Structured_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_Solver_Structured_IOR_h
#include "TOPS_Solver_Structured_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Solver_Structured_hh
#include "TOPS_Solver_Structured.hh"
#endif
#ifndef included_TOPS_System_hh
#include "TOPS_System.hh"
#endif
#ifndef included_gov_cca_CCAException_hh
#include "gov_cca_CCAException.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._includes)
#include "petscdmmg.h"

// Includes for all uses ports
#include "TOPS.hh"
// DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.Solver_Structured" (version 0.0.0)
   */
  class Solver_Structured_impl
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._inherits)
  // Insert-Code-Here {TOPS.Solver_Structured._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Solver_Structured self;

    // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._implementation)
    DMMG               *dmmg;
    DA                 da;
    int                M,N,P,m,n,p,dim,s,levels,bs;
    DAStencilType      stencil_type;
    DAPeriodicType     wrap;
    TOPS::System       system;
    int                startedpetsc;
    gov::cca::Services myServices;
    // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._implementation)

  private:
    // private default constructor (required)
    Solver_Structured_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Solver_Structured_impl( struct TOPS_Solver_Structured__object * s ) : 
      self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Solver_Structured_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    setSystem (
      /* in */ ::TOPS::System system
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::TOPS::System
    getSystem() throw () 
    ;
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
    solve() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setBlockSize (
      /* in */ int32_t bs
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<double>
    getSolution() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setSolution (
      /* in */ ::sidl::array<double> location
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setDimension (
      /* in */ int32_t dim
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getDimension() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setDimensionX (
      /* in */ int32_t dim
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getDimensionX() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setDimensionY (
      /* in */ int32_t dim
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getDimensionY() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setDimensionZ (
      /* in */ int32_t dim
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getDimensionZ() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setStencilWidth (
      /* in */ int32_t width
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getStencilWidth() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setLevels (
      /* in */ int32_t levels
    )
    throw () 
    ;


    /**
     * Starts up a component presence in the calling framework.
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
    setServices (
      /* in */ ::gov::cca::Services services
    )
    throw ( 
      ::gov::cca::CCAException
    );

  };  // end class Solver_Structured_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._misc)
// Insert-Code-Here {TOPS.Solver_Structured._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._misc)

#endif
