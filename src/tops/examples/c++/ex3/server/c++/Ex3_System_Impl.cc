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
#include "petsc.h"
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
  this->solver = solver;
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
  this->solver.setDimension(3);
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
  TOPS::SolverStructured solver = this->solver;
  int xs = B.lower(0);      // first grid point in X and Y directions on this process
  int ys = B.lower(1);
  int zs = B.lower(2);
  int xm = B.length(0);       // number of local grid points in X and Y directions on this process
  int ym = B.length(1);
  int zm = B.length(2);
  int i,j,k;
  int mx = solver.getDimensionX();
  int my = solver.getDimensionY();
  int mz = solver.getDimensionZ();

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double hz     = 1.0/(double)(mz-1);
  double sc     = hx*hy*hz;
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
          B.set(i,j,k,dd); // diagonal entry
        } else {
          B.set(i,j,k,rr);   // seven point stencil
        }
      }
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
  TOPS::Solver_Structured solver = this->solver;
  int xs = b.lower(0);      // first grid point in X and Y directions on this process
  int ys = b.lower(1);
  int zs = b.lower(2);
  int xm = b.length(0);       // number of local grid points in X and Y directions on this process
  int ym = b.length(1);
  int zm = b.length(2);
  int i,j,k;
  int mx = solver.getDimensionX();
  int my = solver.getDimensionY();
  int mz = solver.getDimensionZ();

  double hx     = 1.0/(double)(mx-1);
  double hy     = 1.0/(double)(my-1);
  double hz     = 1.0/(double)(mz-1);
  double sc     = hx*hy*hz;
 
  /*
     Compute right hand side over the locally owned part of the grid
  */
  for (k=zs; j<zs+zm; j++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || i == mx-1 || j == my-1 || k == 0 || k == mz-1) {
	  CHKMEMA;
          b.set(i,j,k,0.0);
          CHKMEMA;
        } else {
	  CHKMEMA;
          b.set(i,j,k,sc);
  	  CHKMEMA;
        }
      }
    }
  }  
  // DO-NOT-DELETE splicer.end(Ex3.System.computeRightHandSide)
}


// DO-NOT-DELETE splicer.begin(Ex3.System._misc)
// Insert-Code-Here {Ex3.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex3.System._misc)

