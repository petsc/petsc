// 
// File:          Ex2_System_Impl.hh
// Symbol:        Ex2.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for Ex2.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_Ex2_System_Impl_hh
#define included_Ex2_System_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_Ex2_System_IOR_h
#include "Ex2_System_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_Ex2_System_hh
#include "Ex2_System.hh"
#endif
#ifndef included_TOPS_Solver_hh
#include "TOPS_Solver.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(Ex2.System._includes)
#include "TOPS.hh"
// DO-NOT-DELETE splicer.end(Ex2.System._includes)

namespace Ex2 { 

  /**
   * Symbol "Ex2.System" (version 0.0.0)
   */
  class System_impl
  // DO-NOT-DELETE splicer.begin(Ex2.System._inherits)
  // Insert-Code-Here {Ex2.System._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(Ex2.System._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    System self;

    // DO-NOT-DELETE splicer.begin(Ex2.System._implementation)
    TOPS::Solver_Structured solver;
    double grashof;
    double prandtl;
    double lid;
    // DO-NOT-DELETE splicer.end(Ex2.System._implementation)

  private:
    // private default constructor (required)
    System_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    System_impl( struct Ex2_System__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~System_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    computeResidual (
      /* in */ ::sidl::array<double> x,
      /* in */ ::sidl::array<double> f
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setSolver (
      /* in */ ::TOPS::Solver solver
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    initializeOnce() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    initializeEverySolve() throw () 
    ;
  };  // end class System_impl

} // end namespace Ex2

// DO-NOT-DELETE splicer.begin(Ex2.System._misc)
// Insert-Code-Here {Ex2.System._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(Ex2.System._misc)

#endif
