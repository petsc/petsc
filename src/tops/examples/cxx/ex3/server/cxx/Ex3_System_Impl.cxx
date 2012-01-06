// 
// File:          Ex3_System_Impl.cxx
// Symbol:        Ex3.System-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex3.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "Ex3_System_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(Ex3.System._includes)
#include <iostream>
// DO-NOT-DELETE splicer.end(Ex3.System._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
Ex3::System_impl::System_impl() : StubBase(reinterpret_cast< void*>(
  ::Ex3::System::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(Ex3.System._ctor2)
  // Insert-Code-Here {Ex3.System._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Ex3.System._ctor2)
}

// user defined constructor
void Ex3::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._ctor)
  // Insert-Code-Here {Ex3.System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._ctor)
}

// user defined destructor
void Ex3::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._dtor)
  // Insert-Code-Here {Ex3.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._dtor)
}

// static class initializer
void Ex3::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._load)
  // Insert-Code-Here {Ex3.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex3.System._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  computeMatrix[]
 */
void
Ex3::System_impl::computeMatrix_impl (
  /* in */::TOPS::Matrix J,
  /* in */::TOPS::Matrix B ) 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeMatrix)
  // Use the TOPS.Structured.Matrix port for getting BB

  TOPS::Structured::Matrix BB = ::babel_cast< TOPS::Structured::Matrix >(B);
  TOPS::Structured::Solver solver = this->solver;
  int xs = BB.getLower(0);      // first grid point in X and Y directions on this process
  int ys = BB.getLower(1);
  int zs = BB.getLower(2);
  int xm = BB.getLength(0);       // number of local grid points in X and Y directions on this process
  int ym = BB.getLength(1);
  int zm = BB.getLength(2);
  int i,j,k;
  int mx = solver.length(0);
  int my = solver.length(1);
  int mz = solver.length(2);

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double hz     = 1.0/(double)(mz-1);
  //double sc     = hx*hy*hz;
  double hxhydhz  = hx*hy/hz; 
  double hyhzdhx  = hy*hz/hx;
  double hxhzdhy  = hx*hz/hy;
 
  /*
     Compute part of matrix over the locally owned part of the grid
  */
  double d = 2.0*(hxhydhz + hxhzdhy + hyhzdhx);
  sidl::array<double> dd = sidl::array<double>::create1d(1,&d);

  double r[7];
  r[0] = r[6] = -hxhydhz;
  r[1] = r[5] = -hxhzdhy;
  r[2] = r[4] = -hyhzdhx;
  r[3] = 2.0*(hxhydhz + hxhzdhy + hyhzdhx);
  sidl::array<double> rr = sidl::array<double>::create1d(7,r);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
	if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1){
          BB.set(i,j,k,dd); // diagonal entry
        } else {
          BB.set(i,j,k,rr);   // seven point stencil
        }
      }
    }
  }
  // DO-NOT-DELETE splicer.end(Ex3.System.computeMatrix)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex3::System_impl::initializeOnce_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeOnce)
  this->solver.setDimen(3);
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeOnce)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex3::System_impl::computeRightHandSide_impl (
  /* in array<double> */::sidl::array<double> b ) 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeRightHandSide)
  TOPS::Structured::Solver solver = this->solver;
  int xs = b.lower(0);      // first grid point in X and Y directions on this process
  int ys = b.lower(1);
  int zs = b.lower(2);
  int xm = b.length(0);       // number of local grid points in X and Y directions on this process
  int ym = b.length(1);
  int zm = b.length(2);
  int i,j,k;
  int mx = solver.length(0);
  int my = solver.length(1);
  int mz = solver.length(2);

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double hz     = 1.0/(double)(mz-1);
  double sc     = hx*hy*hz;
 
  /*
     Compute right hand side over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || i == mx-1 || j == my-1 || k == 0 || k == mz-1) {
          b.set(i,j,k,0.0);
        } else {
          b.set(i,j,k,sc);
        }
      }
    }
  }  
  // DO-NOT-DELETE splicer.end(Ex3.System.computeRightHandSide)
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
Ex3::System_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.setServices)
  // Insert-Code-Here {Ex3.System.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex3::System_impl::setServices"

  myServices = services;

  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting (*this) to gov::cca::Port \n",
	    __FILE__, __LINE__);
    return;
  }
  
  // Provides ports
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
 
  // GoPort (instead of main)
  myServices.addProvidesPort(p, 
			     "DoSolve",
			     "gov.cca.ports.GoPort",
			     myServices.createTypeMap());

  // Uses ports:
  myServices.registerUsesPort("TOPS.Structured.Solver",
			      "TOPS.Structured.Solver", myServices.createTypeMap());

  // DO-NOT-DELETE splicer.end(Ex3.System.setServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex3::System_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.go)
  // Insert-Code-Here {Ex3.System.go} (go method)
#undef __FUNCT__
#define __FUNCT__ "Ex3::System_impl::go"
  
  // Parameter port stuff here (instead of argc, argv);
  // for now pass fake argc and argv to solver

  TOPS::Structured::Solver solver = ::babel_cast< TOPS::Structured::Solver >( myServices.getPort("TOPS.Structured.Solver") );
  this->solver = solver;
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << ": TOPS.Structured.Solver port is nil, "
              << "possibly not connected." << std::endl;
    return 1;
  }

  solver.Initialize();
  
  solver.solve();

  myServices.releasePort("TOPS.StructuredSolver");

  return 0;
  // DO-NOT-DELETE splicer.end(Ex3.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex3.System._misc)
// Insert-Code-Here {Ex3.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex3.System._misc)

