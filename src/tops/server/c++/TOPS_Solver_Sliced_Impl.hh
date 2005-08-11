// 
// File:          TOPS_Solver_Sliced_Impl.hh
// Symbol:        TOPS.Solver_Sliced-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for TOPS.Solver_Sliced
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_TOPS_Solver_Sliced_Impl_hh
#define included_TOPS_Solver_Sliced_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_Solver_Sliced_IOR_h
#include "TOPS_Solver_Sliced_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Matrix_hh
#include "TOPS_Matrix.hh"
#endif
#ifndef included_TOPS_Solver_Sliced_hh
#include "TOPS_Solver_Sliced.hh"
#endif
#ifndef included_TOPS_System_hh
#include "TOPS_System.hh"
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


// DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._includes)
// Insert-Code-Here {TOPS.Solver_Sliced._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.Solver_Sliced" (version 0.0.0)
   */
  class Solver_Sliced_impl
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._inherits)
  // Insert-Code-Here {TOPS.Solver_Sliced._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Solver_Sliced self;

    // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._implementation)
    // Insert-Code-Here {TOPS.Solver_Sliced._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._implementation)

  private:
    // private default constructor (required)
    Solver_Sliced_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Solver_Sliced_impl( struct TOPS_Solver_Sliced__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Solver_Sliced_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    setLocalRowSize (
      /* in */ int32_t m
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getLocalRowSize() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setGlobalRowSize (
      /* in */ int32_t M
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getGlobalRowSize() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setLocalColumnSize (
      /* in */ int32_t n
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getLocalColumnSize() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setGlobalColumnSize (
      /* in */ int32_t N
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getGlobalColumnSize() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setGhostPoints (
      /* in */ ::sidl::array<int32_t> ghosts
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setPreallocation (
      /* in */ int32_t d,
      /* in */ int32_t od
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setPreallocation (
      /* in */ ::sidl::array<int32_t> d,
      /* in */ ::sidl::array<int32_t> od
    )
    throw () 
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
    ::TOPS::Vector
    getRightHandSize (
      /* in */ int32_t level
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::TOPS::Vector
    getSolution (
      /* in */ int32_t Level
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    getJacobian (
      /* in */ int32_t Level,
      /* out */ ::TOPS::Matrix& J,
      /* out */ ::TOPS::Matrix& B
    )
    throw () 
    ;

  };  // end class Solver_Sliced_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._misc)
// Insert-Code-Here {TOPS.Solver_Sliced._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._misc)

#endif
