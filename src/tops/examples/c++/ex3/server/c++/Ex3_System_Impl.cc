// 
// File:          Ex3_System_Impl.cc
// Symbol:        Ex3.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for Ex3.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "Ex3_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex3.System._includes)
// DO-NOT-DELETE splicer.end(Ex3.System._includes)

// user-defined constructor.
void Ex3::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._ctor)
  // Insert-Code-Here {Ex3.System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._ctor)
}

// user-defined destructor.
void Ex3::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._dtor)
  // Insert-Code-Here {Ex3.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._dtor)
}

// static class initializer.
void Ex3::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._load)
  // Insert-Code-Here {Ex3.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex3.System._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  computeMatrix[]
 */
void
Ex3::System_impl::computeMatrix (
  /* in */ ::TOPS::Matrix J,
  /* in */ ::TOPS::Matrix B ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeMatrix)
  TOPS::Structured::Matrix BB = (TOPS::Structured::Matrix)B;
  TOPS::Structured::Solver solver = this->solver;
  int xs = BB.lower(0);      // first grid point in X and Y directions on this process
  int ys = BB.lower(1);
  int zs = BB.lower(2);
  int xm = BB.length(0);       // number of local grid points in X and Y directions on this process
  int ym = BB.length(1);
  int zm = BB.length(2);
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
Ex3::System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeOnce)
  this->solver.setDimen(3);
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeOnce)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex3::System_impl::computeRightHandSide (
  /* in */ ::sidl::array<double> b ) 
throw () 
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
Ex3::System_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(Ex3.System.setServices)
  // Insert-Code-Here {Ex3.System.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex3::System_impl::setServices"

  myServices = services;

  gov::cca::Port p = self;      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting self to gov::cca::Port \n",
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
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex3::System_impl::go ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.go)
  // Insert-Code-Here {Ex3.System.go} (go method)
#undef __FUNCT__
#define __FUNCT__ "Ex3::System_impl::go"
  
  // Parameter port stuff here (instead of argc, argv);
  // for now pass fake argc and argv to solver
  int argc = 1; 
  char *argv[1];
  argv[0] = (char*) malloc(10*sizeof(char));
  strcpy(argv[0],"ex3");

  TOPS::Solver solver = myServices.getPort("TOPS.Structured.Solver");
  this->solver = solver;
  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  
  // We don't need to call setSystem since it will be obtained through
  // getPort calls

  solver.solve();

  myServices.releasePort("TOPS.StructuredSolver");

  return 0;
  // DO-NOT-DELETE splicer.end(Ex3.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex3.System._misc)
// Insert-Code-Here {Ex3.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex3.System._misc)

