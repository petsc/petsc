// 
// File:          TOPS_UnstructuredSolver_Impl.hh
// Symbol:        TOPS.UnstructuredSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.UnstructuredSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 

#ifndef included_TOPS_UnstructuredSolver_Impl_hh
#define included_TOPS_UnstructuredSolver_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_TOPS_UnstructuredSolver_IOR_h
#include "TOPS_UnstructuredSolver_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_System_System_hh
#include "TOPS_System_System.hh"
#endif
#ifndef included_TOPS_UnstructuredSolver_hh
#include "TOPS_UnstructuredSolver.hh"
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


// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._includes)
#include "TOPS.hh"
#include "petscdmmg.h"
#include "gov_cca_ports_ParameterPortFactory.hh"
#include "gov_cca_ports_ParameterPort.hh"
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._includes)

namespace TOPS { 

  /**
   * Symbol "TOPS.UnstructuredSolver" (version 0.0.0)
   */
  class UnstructuredSolver_impl
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._inherits)
  // Insert-Code-Here {TOPS.UnstructuredSolver._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    UnstructuredSolver self;

    // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._implementation)
    DMMG                                  *dmmg;
    ::Sliced                              slice;
    TOPS::System::System                  system;
    int                                   startedpetsc;
    gov::cca::Services                    myServices;
    int                                   bs,n,Nghosted;
    gov::cca::ports::ParameterPortFactory ppf;
    gov::cca::ports::ParameterPort        params;

    int setupParameterPort();
    // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._implementation)

  private:
    // private default constructor (required)
    UnstructuredSolver_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    UnstructuredSolver_impl( struct TOPS_UnstructuredSolver__object * s ) : 
      self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~UnstructuredSolver_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    ::gov::cca::Services
    getServices() throw () 
    ;
    /**
     * user defined non-static method.
     */
    void
    setSystem (
      /* in */ ::TOPS::System::System sys
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::TOPS::System::System
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
    setValue (
      /* in */ const ::std::string& key,
      /* in */ const ::std::string& value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setValue (
      /* in */ const ::std::string& key,
      /* in */ int32_t value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setValue (
      /* in */ const ::std::string& key,
      /* in */ bool value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setValue (
      /* in */ const ::std::string& key,
      /* in */ double value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::std::string
    getValue (
      /* in */ const ::std::string& key
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getValueInt (
      /* in */ const ::std::string& key
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    bool
    getValueBool (
      /* in */ const ::std::string& key
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    double
    getValueDouble (
      /* in */ const ::std::string& key
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    setLocalSize (
      /* in */ int32_t m
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    getLocalSize() throw () 
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
    ::sidl::array<int32_t>
    getGhostPoints() throw () 
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


    /**
     * Inform the listener that someone is about to fetch their 
     * typemap. The return should be true if the listener
     * has changed the ParameterPort definitions.
     */
    bool
    updateParameterPort (
      /* in */ const ::std::string& portName
    )
    throw () 
    ;


    /**
     * The component wishing to be told after a parameter is changed
     * implements this function.
     * @param portName the name of the port (typemap) on which the
     * value was set.
     * @param fieldName the name of the value in the typemap.
     */
    void
    updatedParameterValue (
      /* in */ const ::std::string& portName,
      /* in */ const ::std::string& fieldName
    )
    throw () 
    ;

  };  // end class UnstructuredSolver_impl

} // end namespace TOPS

// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._misc)
// Insert-Code-Here {TOPS.UnstructuredSolver._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._misc)

#endif
