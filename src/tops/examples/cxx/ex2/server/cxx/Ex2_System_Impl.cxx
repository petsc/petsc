// 
// File:          Ex2_System_Impl.cxx
// Symbol:        Ex2.System-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex2.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "Ex2_System_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(Ex2.System._includes)
// Insert-Code-Here {Ex2.System._includes} (additional includes or code)

#include <iostream>

// DO-NOT-DELETE splicer.end(Ex2.System._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
Ex2::System_impl::System_impl() : StubBase(reinterpret_cast< void*>(
  ::Ex2::System::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(Ex2.System._ctor2)
  // Insert-Code-Here {Ex2.System._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Ex2.System._ctor2)
}

// user defined constructor
void Ex2::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex2.System._ctor)
  this->lid     = 0.0;
  this->prandtl = 1.0;
  this->grashof = 1.0;
  // DO-NOT-DELETE splicer.end(Ex2.System._ctor)
}

// user defined destructor
void Ex2::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex2.System._dtor)
  // Insert-Code-Here {Ex2.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex2.System._dtor)
}

// static class initializer
void Ex2::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex2.System._load)
  // Insert-Code-Here {Ex2.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex2.System._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initializeOnce[]
 */
void
Ex2::System_impl::initializeOnce_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex2.System.initializeOnce)
  this->solver.setBlockSize(4);
  // DO-NOT-DELETE splicer.end(Ex2.System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex2::System_impl::initializeEverySolve_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex2.System.initializeEverySolve)
  this->lid = 1.0/(this->solver.length(0)*this->solver.length(1));
  // DO-NOT-DELETE splicer.end(Ex2.System.initializeEverySolve)
}

/**
 * Method:  computeResidual[]
 */
void
Ex2::System_impl::computeResidual_impl (
  /* in array<double> */::sidl::array<double> x,
  /* in array<double> */::sidl::array<double> f ) 
{
  // DO-NOT-DELETE splicer.begin(Ex2.System.computeResidual)
  TOPS::Structured::Solver solver = this->solver;
  int xs = f.lower(1);      // first grid point in X and Y directions on this process
  int ys = f.lower(2);
  int xm = f.length(1);       // number of local grid points in X and Y directions on this process
  int ym = f.length(2);
  int i,j;
  int mx = solver.length(0);
  int my = solver.length(1);

  double hx     = 1.0/(double)(mx-1), dhx = (double)(mx-1);
  double hy     = 1.0/(double)(my-1), dhy = (double)(my-1);
  double hxdhy  = hx/hy; 
  double hydhx  = hy/hx;

  double vx,avx,vxp,vxm,vy,avy,vyp,vym,u,uxx,uyy;
  double grashof = this->grashof;  
  double prandtl = this->prandtl;
  double lid     = this->lid;

  int xints = xs, xinte = xs+xm, yints = ys, yinte = ys+ym;

#define U 0
#define V 1
#define OMEGA 2
#define TEMP  3
  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=xs; i<xs+xm; i++) {
      f.set(U,i,j,x.get(U,i,j));
      f.set(V,i,j,x.get(V,i,j));
      f.set(OMEGA,i,j,x.get(OMEGA,i,j) + (x.get(U,i,j+1) - x.get(U,i,j))*dhy); 
      f.set(TEMP,i,j, x.get(TEMP,i,j)-x.get(TEMP,i,j+1));
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == my) {
    j = my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=xs; i<xs+xm; i++) {
      f.set(U,i,j,x.get(U,i,j) - lid);
      f.set(V,i,j,x.get(V,i,j));
      f.set(OMEGA,i,j,x.get(OMEGA,i,j) + (x.get(U,i,j) - x.get(U,i,j-1))*dhy);
      f.set(TEMP,i,j,x.get(TEMP,i,j) - x.get(TEMP,i,j-1));
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=ys; j<ys+ym; j++) {
      f.set(U,i,j,x.get(U,i,j));
      f.set(V,i,j,x.get(V,i,j));
      f.set(OMEGA,i,j,x.get(OMEGA,i,j) - (x.get(V,i+1,j) - x.get(V,i,j))*dhx);
      f.set(TEMP,i,j,x.get(TEMP,i,j));
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == mx) {
    i = mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=ys; j<ys+ym; j++) {
      f.set(U,i,j,x.get(U,i,j));
      f.set(V,i,j,x.get(V,i,j));
      f.set(OMEGA,i,j,x.get(OMEGA,i,j) - (x.get(V,i,j) - x.get(V,i-1,j))*dhx);
      f.set(TEMP,i,j,x.get(TEMP,i,j) - (double)(grashof > 0));
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {

	/*
	  convective coefficients for upwinding
        */
        vx = x.get(U,i,j); avx = (vx > 0.0) ? vx : -vx;
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x.get(V,i,j); avy = (vy > 0.0) ? vy : -vy;
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x.get(U,i,j);
        uxx        = (2.0*u - x.get(U,i-1,j) - x.get(U,i+1,j))*hydhx;
        uyy        = (2.0*u - x.get(U,i,j-1) - x.get(U,i,j+1))*hxdhy;
        f.set(U,i,j,uxx + uyy - .5*(x.get(OMEGA,i,j+1)-x.get(OMEGA,i,j-1))*hx);

	/* V velocity */
        u          = x.get(V,i,j);
        uxx        = (2.0*u - x.get(V,i-1,j) - x.get(V,i+1,j))*hydhx;
        uyy        = (2.0*u - x.get(V,i,j-1) - x.get(V,i,j+1))*hxdhy;
        f.set(V,i,j, uxx + uyy + .5*(x.get(OMEGA,i+1,j)-x.get(OMEGA,i-1,j))*hy);

	/* Omega */
        u          = x.get(OMEGA,i,j);
        uxx        = (2.0*u - x.get(OMEGA,i-1,j) - x.get(OMEGA,i+1,j))*hydhx;
        uyy        = (2.0*u - x.get(OMEGA,i,j-1) - x.get(OMEGA,i,j+1))*hxdhy;
	f.set(OMEGA,i,j, uxx + uyy + (vxp*(u - x.get(OMEGA,i-1,j)) +
				      vxm*(x.get(OMEGA,i+1,j) - u)) * hy +
                                     (vyp*(u - x.get(OMEGA,i,j-1)) +
				      vym*(x.get(OMEGA,i,j+1) - u)) * hx -
                                      .5 * grashof * (x.get(TEMP,i+1,j) - x.get(TEMP,i-1,j)) * hy);

        /* Temperature */
        u             = x.get(TEMP,i,j);
        uxx           = (2.0*u - x.get(TEMP,i-1,j) - x.get(TEMP,i+1,j))*hydhx;
        uyy           = (2.0*u - x.get(TEMP,i,j-1) - x.get(TEMP,i,j+1))*hxdhy;
	f.set(TEMP,i,j, uxx + uyy  + prandtl * ((vxp*(u - x.get(TEMP,i-1,j)) + vxm*(x.get(TEMP,i+1,j) - u)) * hy +
						(vyp*(u - x.get(TEMP,i,j-1)) + vym*(x.get(TEMP,i,j+1) - u)) * hx));
    }
  }

  // DO-NOT-DELETE splicer.end(Ex2.System.computeResidual)
}

/**
 * Method:  computeInitialGuess[]
 */
void
Ex2::System_impl::computeInitialGuess_impl (
  /* in array<double> */::sidl::array<double> x ) 
{
  // DO-NOT-DELETE splicer.begin(Ex2.System.computeInitialGuess)
  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  TOPS::Structured::Solver solver = this->solver;
  int xs = x.lower(1);      // first grid point in X and Y directions on this process
  int ys = x.lower(2);
  int xm = x.length(1) - 1;       // number of local grid points in X and Y directions on this process
  int ym = x.length(2) - 1;
  int i,j;
  double dx  = 1.0/(solver.length(0)-1);
  double grashof = this->grashof;  
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x.set(U,i,j,0.0);
      x.set(V,i,j,0.0);
      x.set(OMEGA,i,j,0.0);
      x.set(TEMP,i,j,(grashof>0)*i*dx);
    }
  }
  // DO-NOT-DELETE splicer.end(Ex2.System.computeInitialGuess)
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
Ex2::System_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(Ex2.System.setServices)
  // Insert-Code-Here {Ex2.System.setServices} (setServices method)
  myServices = services;

  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting self to gov::cca::Port \n",
	    __FILE__, __LINE__);
    return;
  }
  
  // Provides ports
  myServices.addProvidesPort(p,
			   "TOPS.System.Initialize.Once",
			   "TOPS.System.Initialize.Once", myServices.createTypeMap());
  
  myServices.addProvidesPort(p,
			   "TOPS.System.Initialize.EverySolve",
			   "TOPS.System.Initialize.EverySolve", myServices.createTypeMap());

  myServices.addProvidesPort(p,
			   "TOPS.System.Compute.InitialGuess",
			   "TOPS.System.Compute.InitialGuess", myServices.createTypeMap());

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

  // DO-NOT-DELETE splicer.end(Ex2.System.setServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex2::System_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex2.System.go)
  // Insert-Code-Here {Ex2.System.go} (go method)

  this->solver = ::babel_cast< TOPS::Structured::Solver >( myServices.getPort("TOPS.Structured.Solver") );
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << ": TOPS.Structured.Solver port is nil, "
              << "possibly not connected." << std::endl;
    return 1;
  }

  this->solver.Initialize();
  this->solver.solve();
  myServices.releasePort("TOPS.StructuredSolver");

  return 0;  // DO-NOT-DELETE splicer.end(Ex2.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex2.System._misc)
// Insert-Code-Here {Ex2.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex2.System._misc)

