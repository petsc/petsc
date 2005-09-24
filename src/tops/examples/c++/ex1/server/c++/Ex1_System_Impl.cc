// 
// File:          Ex1_System_Impl.cc
// Symbol:        Ex1.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for Ex1.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "Ex1_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex1.System._includes)
#include <iostream>
#include "petsc.h"
#include "petscconf.h"
#if defined(PETSC_HAVE_CCAFE)
#  define USE_PORTS 1
#endif

// Includes for uses ports
#include "TOPS_Structured_Solver.hh"
// DO-NOT-DELETE splicer.end(Ex1.System._includes)

// user-defined constructor.
void Ex1::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._ctor)
  // Insert-Code-Here {Ex1.System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex1.System._ctor)
}

// user-defined destructor.
void Ex1::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._dtor)
  // Insert-Code-Here {Ex1.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex1.System._dtor)
}

// static class initializer.
void Ex1::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex1.System._load)
  // Insert-Code-Here {Ex1.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex1.System._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSolver[]
 */
void
Ex1::System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.setSolver)
  this->solver = (TOPS::Structured::Solver)solver;
  // DO-NOT-DELETE splicer.end(Ex1.System.setSolver)
}

/**
 * Method:  computeResidual[]
 */
void
Ex1::System_impl::computeResidual (
  /* in */ ::sidl::array<double> x,
  /* in */ ::sidl::array<double> f ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.computeResidual)
#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::computeResidual"

  TOPS::Structured::Solver solver;
#ifdef USE_PORTS
  solver = this->myServices.getPort("TOPS.Structured.Solver");
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.Structured.Solver port is nil, " 
	      << "possibly not connected." << std::endl;
    return;
  }
#else
  solver = this->solver;
#endif

  int xs = f.lower(0);      // first grid point in X and Y directions on this process
  int ys = f.lower(1);
  int xm = f.length(0);       // number of local grid points in X and Y directions on this process
  int ym = f.length(1);
  int i,j;
  int mx = solver.length(0);
  int my = solver.length(1);

#ifdef USE_PORTS
  this->myServices.releasePort("TOPS.Structured.Solver");
#endif

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
        CHKMEMA;
        f.set(i,j,x.get(i,j));
        CHKMEMA;
      } else {
        double u       = x.get(i,j);
        double uxx     = (2.0*u - x.get(i-1,j) - x.get(i+1,j))*hydhx;
        double uyy     = (2.0*u - x.get(i,j-1) - x.get(i,j+1))*hxdhy;
        CHKMEMA;
        f.set(i,j,uxx + uyy - sc*exp(u));
        CHKMEMA;
      }
    }  
  }  


  // DO-NOT-DELETE splicer.end(Ex1.System.computeResidual)
}

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
Ex1::System_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(Ex1.System.setServices)
  // Insert-Code-Here {Ex1.System.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::setServices"

  myServices = services;
  gov::cca::TypeMap tm = services.createTypeMap();
  if(tm._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: gov::cca::TypeMap is nil\n",
	    __FILE__, __LINE__);
    exit(1);
  }
  gov::cca::Port p = self;      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting self to gov::cca::Port \n",
	    __FILE__, __LINE__);
    exit(1);
  }
  
  // Provides ports
  // Basic functionality
  myServices.addProvidesPort(p,
			   "TOPS.System",
			   "TOPS.System", tm);

  // Residual computation
  myServices.addProvidesPort(p,
			   "TOPS.System.Compute.Residual",
			   "TOPS.System.Compute.Residual", tm);
  

  // GoPort (instead of main)
  myServices.addProvidesPort(p, 
			     "DoSolve",
			     "gov.cca.ports.GoPort",
			     myServices.createTypeMap());

  // Uses ports:
  myServices.registerUsesPort("TOPS.Structured.Solver",
			      "TOPS.Structured.Solver", tm);

  // DO-NOT-DELETE splicer.end(Ex1.System.setServices)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex1::System_impl::go ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.System.go)
  // Insert-Code-Here {Ex1.System.go} (go method)

#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::go"
  
  // Parameter port stuff here (instead of argc, argv);
  // for now pass fake argc and argv to solver
  int argc = 1; 
  char *argv[1];
  argv[0] = (char*) malloc(10*sizeof(char));
  strcpy(argv[0],"ex1");

  TOPS::Solver solver = myServices.getPort("TOPS.Structured.Solver");
  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  
  PetscOptionsSetValue("-snes_monitor",PETSC_NULL);

  // We don't need to call setSystem since it will be obtained through
  // getPort calls

  solver.solve();

  myServices.releasePort("TOPS.StructuredSolver");

  PetscFunctionReturn(0);
  // DO-NOT-DELETE splicer.end(Ex1.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex1.System._misc)
// Insert-Code-Here {Ex1.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex1.System._misc)

