// 
// File:          Ex4_System_Impl.cxx
// Symbol:        Ex4.System-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for Ex4.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "Ex4_System_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(Ex4.System._includes)
// Insert-Code-Here {Ex4.System._includes} (additional includes or code)
#include <iostream>
// DO-NOT-DELETE splicer.end(Ex4.System._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
Ex4::System_impl::System_impl() : StubBase(reinterpret_cast< void*>(
  ::Ex4::System::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(Ex4.System._ctor2)
  // Insert-Code-Here {Ex4.System._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Ex4.System._ctor2)
}

// user defined constructor
void Ex4::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._ctor)
  this->n = 4;
  // DO-NOT-DELETE splicer.end(Ex4.System._ctor)
}

// user defined destructor
void Ex4::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._dtor)
  // Insert-Code-Here {Ex4.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex4.System._dtor)
}

// static class initializer
void Ex4::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex4.System._load)
  // Insert-Code-Here {Ex4.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex4.System._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  computeMatrix[]
 */
void
Ex4::System_impl::computeMatrix_impl (
  /* in */::TOPS::Matrix J,
  /* in */::TOPS::Matrix B ) 
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.computeMatrix)
#undef __FUNCT__
#define __FUNCT__ "Ex4::System_impl::computeMatrix"

  TOPS::Unstructured::Matrix BB = ::babel_cast< TOPS::Unstructured::Matrix >(B);
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
 * Method:  initializeOnce[]
 */
void
Ex4::System_impl::initializeOnce_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex4.System.initializeOnce)
#undef __FUNCT__
#define __FUNCT__ "Ex4::System_impl::initializeOnce"

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
 * Method:  computeRightHandSide[]
 */
void
Ex4::System_impl::computeRightHandSide_impl (
  /* in array<double> */::sidl::array<double> b ) 
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.computeRightHandSide)
#undef __FUNCT__
#define __FUNCT__ "Ex4::System_impl::computeRightHandSide"
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
Ex4::System_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(Ex4.System.setServices)
  // Insert-Code-Here {Ex4.System.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "Ex4::System_impl::setServices"

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
  myServices.registerUsesPort("TOPS.Unstructured.Solver",
			      "TOPS.Unstructured.Solver", myServices.createTypeMap());
  // DO-NOT-DELETE splicer.end(Ex4.System.setServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
Ex4::System_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(Ex4.System.go)
  // Insert-Code-Here {Ex4.System.go} (go method)

#undef __FUNCT__
#define __FUNCT__ "Ex4::System_impl::go"
  
  // Parameter port stuff here (instead of argc, argv);
  // for now pass fake argc and argv to solver

  TOPS::Unstructured::Solver solver = ::babel_cast< TOPS::Unstructured::Solver>( myServices.getPort("TOPS.Unstructured.Solver") );
  if (solver._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << ": TOPS.Structured.Solver port is nil, "
              << "possibly not connected." << std::endl;
    return 1;
  }
  this->solver = solver;

  solver.Initialize();
  
  solver.solve();

  myServices.releasePort("TOPS.UnstructuredSolver");

  return 0;
  // DO-NOT-DELETE splicer.end(Ex4.System.go)
}


// DO-NOT-DELETE splicer.begin(Ex4.System._misc)
// Insert-Code-Here {Ex4.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex4.System._misc)

