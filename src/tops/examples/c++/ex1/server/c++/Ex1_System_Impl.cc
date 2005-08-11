// 
// File:          Ex1_System_Impl.cc
// Symbol:        Ex1.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for Ex1.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "Ex1_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex1.System._includes)
#include "petsc.h"
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
 * Method:  computeResidual[]
 */
void
Ex1::System_impl::computeResidual (
  /* in */ ::sidl::array<double> x,
  /* in */ ::sidl::array<double> f ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.computeResidual)
  TOPS::Solver_Structured solver = this->solver;
  int xs = f.lower(0);      // first grid point in X and Y directions on this process
  int ys = f.lower(1);
  int xm = f.length(0) - 1;       // number of local grid points in X and Y directions on this process
  int ym = f.length(1) - 1;
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
 * Method:  setSolver[]
 */
void
Ex1::System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.System.setSolver)
  this->solver = (TOPS::Solver_Structured)solver;
  // DO-NOT-DELETE splicer.end(Ex1.System.setSolver)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex1::System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.System.initializeOnce)
  
  // DO-NOT-DELETE splicer.end(Ex1.System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex1::System_impl::initializeEverySolve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.System.initializeEverySolve)
  // Insert-Code-Here {Ex1.System.initializeEverySolve} (initializeEverySolve method)
  // DO-NOT-DELETE splicer.end(Ex1.System.initializeEverySolve)
}


// DO-NOT-DELETE splicer.begin(Ex1.System._misc)
// Insert-Code-Here {Ex1.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex1.System._misc)

