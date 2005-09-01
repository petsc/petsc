// 
// File:          Ex4_System_Impl.cc
// Symbol:        Ex4.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for Ex4.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "Ex4_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex4.System._includes)
// Insert-Code-Here {Ex4.System._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(Ex4.System._includes)

// user-defined constructor.
void Ex4::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._ctor)
  this->n = 4;
  // DO-NOT-DELETE splicer.end(Ex4.System._ctor)
}

// user-defined destructor.
void Ex4::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._dtor)
  // Insert-Code-Here {Ex4.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex4.System._dtor)
}

// static class initializer.
void Ex4::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._load)
  // Insert-Code-Here {Ex4.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex4.System._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSolver[]
 */
void
Ex4::System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.setSolver)
  this->solver = (TOPS::Unstructured::Solver)solver;
  // DO-NOT-DELETE splicer.end(Ex4.System.setSolver)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex4::System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex4.System.initializeOnce)
  this->solver.setLocalSize(this->n);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
  int start = this->n*rank;
  int cnt = 0,g[2];
  if (rank) g[cnt++] = start-1;
  if (rank != size-1) g[cnt++] = start+this->n;
  this->solver.setGhostPoints(sidl::array<int>::create1d(cnt,g));
  // DO-NOT-DELETE splicer.end(Ex4.System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex4::System_impl::initializeEverySolve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex4.System.initializeEverySolve)
  // Insert-Code-Here {Ex4.System.initializeEverySolve} (initializeEverySolve method)
  // DO-NOT-DELETE splicer.end(Ex4.System.initializeEverySolve)
}

/**
 * Method:  computeMatrix[]
 */
void
Ex4::System_impl::computeMatrix (
  /* in */ ::TOPS::Matrix J,
  /* in */ ::TOPS::Matrix B ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.computeMatrix)
  TOPS::Unstructured::Matrix BB = (TOPS::Unstructured::Matrix)B;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
  double hx     = 1.0/(double)(this->n*size);
  int end = this->n,start = 0;
  if (!rank) {
    int ia[1]; ia[0] = 0;
    sidl::array<int> row = sidl::array<int>::create1d(1,ia);
    double iv[1]; iv[0] = 1.0/hx;
    sidl::array<double> values = sidl::array<double>::create1d(1,iv);    
    BB.set(row,row,values);
    start++;
  }
  if (rank == size-1) {
    int ia[1]; ia[0] = this->n - 1;
    sidl::array<int> row = sidl::array<int>::create1d(1,ia);
    double iv[1]; iv[0] = 1.0/hx;
    sidl::array<double> values = sidl::array<double>::create1d(1,iv);    
    BB.set(row,row,values);
    end--;
  }
  sidl::array<int> row = sidl::array<int>::create1d(1);
  sidl::array<int> cols = sidl::array<int>::create1d(3);
  sidl::array<double> values = sidl::array<double>::create1d(3);
  values.set(0,-1.0/hx);
  values.set(1,2.0/hx);
  values.set(2,-1.0/hx);
  int i;
  // loop over local nodes; generally you would be looping over elements
  for (i=start; i<end; i++) {
    row.set(0,i);
    cols.set(0,(i-1 < 0) ? this->n : i-1);
    cols.set(1,i);
    cols.set(2,(i+1 > this->n-1) ? this->n+1-start: i+1);
    BB.set(row,cols,values);
  }
  // DO-NOT-DELETE splicer.end(Ex4.System.computeMatrix)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex4::System_impl::computeRightHandSide (
  /* in */ ::sidl::array<double> b ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.computeRightHandSide)
  int i,nlocal = b.length(0);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
  // For a finite element discretization the local element contributions to the 
  // ghost degrees of freedom would also be computed here. Skipped here.
  if (rank) nlocal--;
  if (rank != size-1) nlocal--;
  for (i=0; i<nlocal; i++) {
    b.set(i,1.0);
  }
  // DO-NOT-DELETE splicer.end(Ex4.System.computeRightHandSide)
}


// DO-NOT-DELETE splicer.begin(Ex4.System._misc)
// Insert-Code-Here {Ex4.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex4.System._misc)

