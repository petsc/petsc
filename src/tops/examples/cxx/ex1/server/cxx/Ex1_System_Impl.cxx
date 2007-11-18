// 
// File:          Ex1_System_Impl.cxx
// Symbol:        Ex1.System-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex1.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "Ex1_System_Impl.hxx"

// 
// Includes for all method dependencies.
// 
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
// DO-NOT-DELETE splicer.begin(Ex1.System._includes)
#include <iostream>

// Includes for uses ports
#include "TOPS_Structured_Solver.hxx"
// DO-NOT-DELETE splicer.end(Ex1.System._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
Ex1::System_impl::System_impl() : StubBase(reinterpret_cast< void*>(
  ::Ex1::System::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(Ex1.System._ctor2)
  // Insert-Code-Here {Ex1.System._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Ex1.System._ctor2)
}

// user defined constructor
void Ex1::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._ctor)
  // Insert-Code-Here {Ex1.System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex1.System._ctor)
}

// user defined destructor
void Ex1::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._dtor)
  // Insert-Code-Here {Ex1.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex1.System._dtor)
}

// static class initializer
void Ex1::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._load)
  // Insert-Code-Here {Ex1.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex1.System._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  computeResidual[]
 */
void
Ex1::System_impl::computeResidual_impl (
  /* in array<double> */::sidl::array<double> x,
  /* in array<double> */::sidl::array<double> f ) 
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.computeResidual)
#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::computeResidual"

  TOPS::Structured::Solver solver;
  solver = ::babel_cast< TOPS::Structured::Solver> (
  	this->myServices.getPort("TOPS.Structured.Solver") );
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << ": TOPS.Structured.Solver port is nil, " 
	      << "possibly not connected." << std::endl;
    return;
  }

  int xs = f.lower(0);      // first grid point in X and Y directions on this process
  int ys = f.lower(1);
  int xm = f.length(0);       // number of local grid points in X and Y directions on this process
  int ym = f.length(1);
  int i,j;
  int mx = solver.length(0);
  int my = solver.length(1);

  this->myServices.releasePort("TOPS.Structured.Solver");

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double sc     = hx*hy;
  double hxdhy  = hx/hy; 
  double hydhx  = hy/hx;
 
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f.set(i,j,x.get(i,j));
      } else {
        double u       = x.get(i,j);
        double uxx     = (2.0*u - x.get(i-1,j) - x.get(i+1,j))*hydhx;
        double uyy     = (2.0*u - x.get(i,j-1) - x.get(i,j+1))*hxdhy;
        f.set(i,j,uxx + uyy - sc*exp(u));
      }
    }  
  }  


  // DO-NOT-DELETE splicer.end(Ex1.System.computeResidual)
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
Ex1::System_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.setServices)
  // Insert-Code-Here {Ex1.System.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::setServices"

  myServices = services;

  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting (*this) to gov::cca::Port \n",
	    __FILE__, __LINE__);
    return;
  }
  
  // Provides ports
  // Residual computation
  myServices.addProvidesPort(p,
			   "TOPS.System.Compute.Residual",
			   "TOPS.System.Compute.Residual", myServices.createTypeMap());
  

  // GoPort (instead of main)
  myServices.addProvidesPort(p, 
			     "DoSolve",
			     "gov.cca.ports.GoPort",
			     myServices.createTypeMap());

  // Uses ports:
  myServices.registerUsesPort("TOPS.Structured.Solver",
			      "TOPS.Structured.Solver", myServices.createTypeMap());

  // DO-NOT-DELETE splicer.end(Ex1.System.setServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex1::System_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.System.go)
  // Insert-Code-Here {Ex1.System.go} (go method)

#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::go"
  
  TOPS::Solver solver = ::babel_cast< TOPS::Structured::Solver >( 
  	myServices.getPort("TOPS.Structured.Solver") );
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << ": TOPS.Structured.Solver port is nil, "
              << "possibly not connected." << std::endl;
    return 1;
  }

  solver.Initialize();
  
  solver.solve();

  myServices.releasePort("TOPS.StructuredSolver");

  return 0;
  // DO-NOT-DELETE splicer.end(Ex1.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex1.System._misc)
// Insert-Code-Here {Ex1.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex1.System._misc)

