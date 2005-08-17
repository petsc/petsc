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
// Insert-Code-Here {Ex3.System._includes} (additional includes or code)
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
 * Method:  setSolver[]
 */
void
Ex3::System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.setSolver)
  // Insert-Code-Here {Ex3.System.setSolver} (setSolver method)
  // DO-NOT-DELETE splicer.end(Ex3.System.setSolver)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex3::System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeOnce)
  // Insert-Code-Here {Ex3.System.initializeOnce} (initializeOnce method)
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex3::System_impl::initializeEverySolve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeEverySolve)
  // Insert-Code-Here {Ex3.System.initializeEverySolve} (initializeEverySolve method)
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeEverySolve)
}

/**
 * Method:  computeMatrix[]
 */
void
Ex3::System_impl::computeMatrix (
  /* in */ ::TOPS::Matrix J ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeMatrix)
  TOPS::MatrixStructured B = (TOPS::MatrixStructured)J;
  TOPS::Solver_Structured solver = this->solver;
  int xs = B.lower(0);      // first grid point in X and Y directions on this process
  int ys = B.lower(1);
  int xm = B.length(0) - 1;       // number of local grid points in X and Y directions on this process
  int ym = B.length(1) - 1;
  int i,j;
  int mx = solver.getDimensionX();
  int my = solver.getDimensionY();

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double sc     = hx*hy;
  double hxdhy  = hx/hy; 
  double hydhx  = hy/hx;
 
  /*
     Compute function over the locally owned part of the grid
  */
  double one = 1.0;
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      //  if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        CHKMEMA;
        B.set(i,j,sidl::array<double>::create1d(1,&one));
        CHKMEMA;
	//      }
    }
  }
  // DO-NOT-DELETE splicer.end(Ex3.System.computeMatrix)
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
  // Insert-Code-Here {Ex3.System.computeRightHandSide} (computeRightHandSide method)
  // DO-NOT-DELETE splicer.end(Ex3.System.computeRightHandSide)
}


// DO-NOT-DELETE splicer.begin(Ex3.System._misc)
// Insert-Code-Here {Ex3.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex3.System._misc)

