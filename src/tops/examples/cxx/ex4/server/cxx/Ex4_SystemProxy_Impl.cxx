// 
// File:          Ex4_SystemProxy_Impl.cxx
// Symbol:        Ex4.SystemProxy-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex4.SystemProxy
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "Ex4_SystemProxy_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_TOPS_Matrix_hxx
#include "TOPS_Matrix.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._includes)
// Insert-Code-Here {Ex4.SystemProxy._includes} (additional includes or code)
#include <iostream>
#define MPICH_IGNORE_CXX_SEEK
#include "mpi.h"
#include "TOPS.hxx"
// DO-NOT-DELETE splicer.end(Ex4.SystemProxy._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
Ex4::SystemProxy_impl::SystemProxy_impl() : StubBase(reinterpret_cast< void*>(
  ::Ex4::SystemProxy::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._ctor2)
  // Insert-Code-Here {Ex4.SystemProxy._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy._ctor2)
}

// user defined constructor
void Ex4::SystemProxy_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._ctor)
  // Insert-Code-Here {Ex4.SystemProxy._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy._ctor)
}

// user defined destructor
void Ex4::SystemProxy_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._dtor)
  // Insert-Code-Here {Ex4.SystemProxy._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy._dtor)
}

// static class initializer
void Ex4::SystemProxy_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._load)
  // Insert-Code-Here {Ex4.SystemProxy._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  computeMatrix[]
 */
void
Ex4::SystemProxy_impl::computeMatrix_impl (
  /* in */::TOPS::Matrix J,
  /* in */::TOPS::Matrix B ) 
{
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy.computeMatrix)
  // Insert-Code-Here {Ex4.SystemProxy.computeMatrix} (computeMatrix method)
#undef __FUNCT__
#define __FUNCT__ "Ex4::SystemProxy_impl::computeMatrix_impl"

  // This proxy routine simply passes the invocation through to 
  // the connected System implementation.
  
  TOPS::System::Compute::Matrix system;
  system = ::babel_cast< TOPS::System::Compute::Matrix >(
    myServices.getPort("u_proxy_TOPS.System.Compute.Matrix"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
          << ": TOPS.System.Compute.Matrix port is nil, " 
          << "possibly not connected." << std::endl;
    return;
  }

  // Use the port
  system.computeMatrix(J,B);
  std::cout << "SystemProxy: after call to system.computeMatrix" << std::endl;
 
  myServices.releasePort("TOPS.System.Compute.Matrix");
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy.computeMatrix)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex4::SystemProxy_impl::initializeOnce_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy.initializeOnce)
  // Insert-Code-Here {Ex4.SystemProxy.initializeOnce} (initializeOnce method)
#undef __FUNCT__
#define __FUNCT__ "Ex4::SystemProxy_impl::initializeOnce_impl"

  // This proxy routine simply passes the invocation through to 
  // the connected System implementation.
  
  TOPS::System::Initialize::Once system;
  system = ::babel_cast< TOPS::System::Initialize::Once >(
    myServices.getPort("u_proxy_TOPS.System.Initialize.Once"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
          << ": TOPS.System.Initialize.Once port is nil, " 
          << "possibly not connected." << std::endl;
    return;
  }

  // Use the port
  system.initializeOnce();
  std::cout << "SystemProxy: after call to system.initializeOnce" << std::endl;

  myServices.releasePort("TOPS.System.Initialize.Once");
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy.initializeOnce)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex4::SystemProxy_impl::computeRightHandSide_impl (
  /* in array<double> */::sidl::array<double> b ) 
{
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy.computeRightHandSide)
  // Insert-Code-Here {Ex4.SystemProxy.computeRightHandSide} (computeRightHandSide method)
#undef __FUNCT__
#define __FUNCT__ "Ex4::SystemProxy_impl::computeRightHandSide_impl"

  // This proxy routine simply passes the invocation through to 
  // the connected System implementation.
  
  TOPS::System::Compute::RightHandSide system;
  system = ::babel_cast< TOPS::System::Compute::RightHandSide >(
    myServices.getPort("u_proxy_TOPS.System.Compute.RightHandSide"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
          << ": TOPS.System.Compute.RightHandSide port is nil, " 
          << "possibly not connected." << std::endl;
    return;
  }

  // Use the port
  system.computeRightHandSide(b);
  std::cout << "SystemProxy: after call to system.computeRightHandSide" << std::endl;
 
  myServices.releasePort("TOPS.System.Compute.RightHandSide");
  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy.computeRightHandSide)
}

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
Ex4::SystemProxy_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(Ex4.SystemProxy.setServices)
  // Insert-Code-Here {Ex4.SystemProxy.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex4::SystemProxy_impl::setServices"

  myServices = services;

  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting (*this) to gov::cca::Port \n",
        __FILE__, __LINE__);
    return;
  }
  
  // Since this is a System proxy, it defines both uses and provides ports
  // System and Solver.
  
  // Provides ports for System
  // Initialization
  myServices.addProvidesPort(p,
               "TOPS.System.Initialize.Once",
               "TOPS.System.Initialize.Once", myServices.createTypeMap());
  // Matrix computation
  myServices.addProvidesPort(p,
               "TOPS.System.Compute.Matrix",
               "TOPS.System.Compute.Matrix", myServices.createTypeMap());
  
  // RHS computation
  myServices.addProvidesPort(p,
               "TOPS.System.Compute.RightHandSide",
               "TOPS.System.Compute.RightHandSide", myServices.createTypeMap());
                   
  // --------------------------------------------------------------------------
  // Symmetrical uses/provides ports for proxy
  // --------------------------------------------------------------------------
  // Initialization
  myServices.registerUsesPort("u_proxy_TOPS.System.Initialize.Once",
               "TOPS.System.Initialize.Once", myServices.createTypeMap());
               
  // Matrix computation
  myServices.registerUsesPort("u_proxy_TOPS.System.Compute.Matrix",
               "TOPS.System.Compute.Matrix", myServices.createTypeMap());
  
  // RHS computation
  myServices.registerUsesPort("u_proxy_TOPS.System.Compute.RightHandSide",
               "TOPS.System.Compute.RightHandSide", myServices.createTypeMap());
 

  // DO-NOT-DELETE splicer.end(Ex4.SystemProxy.setServices)
}


// DO-NOT-DELETE splicer.begin(Ex4.SystemProxy._misc)
// Insert-Code-Here {Ex4.SystemProxy._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex4.SystemProxy._misc)

