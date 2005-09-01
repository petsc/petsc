// 
// File:          TOPS_UnstructuredSolver_Impl.cc
// Symbol:        TOPS.UnstructuredSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.UnstructuredSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_UnstructuredSolver_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._includes)
#include "TOPS_Unstructured_Matrix_Impl.hh"
static PetscErrorCode FormFunction(SNES snes,Vec uu,Vec f,void *vdmmg)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialGuess(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode FormMatrix(DMMG dmmg,Mat J,Mat B)
{
  PetscFunctionBegin;
  TOPS::UnstructuredSolver *solver = (TOPS::UnstructuredSolver*) dmmg->user;
  TOPS::System::Compute::Matrix system = (TOPS::System::Compute::Matrix) solver->getSystem();
  TOPS::Unstructured::Matrix matrix1 = TOPS::Unstructured::Matrix::_create();
  TOPS::Unstructured::Matrix matrix2 = TOPS::Unstructured::Matrix::_create();

#define GetImpl(A,b) (!(A)b) ? 0 : reinterpret_cast<A ## _impl*>(((A) b)._get_ior()->d_data)

  // currently no support for dof > 1
  TOPS::Unstructured::Matrix_impl *imatrix1 = GetImpl(TOPS::Unstructured::Matrix,matrix1);
  TOPS::Unstructured::Matrix_impl *imatrix2 = GetImpl(TOPS::Unstructured::Matrix,matrix2);
  imatrix1->mat = J;
  imatrix2->mat = B;

  system.computeMatrix(matrix1,matrix2);
  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  if (J != B) {
    MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRightHandSide(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  TOPS::UnstructuredSolver *solver = (TOPS::UnstructuredSolver*) dmmg->user;
  TOPS::System::Compute::RightHandSide system = (TOPS::System::Compute::RightHandSide) solver->getSystem();
  double *uu;
  Vec local;
  VecGhostGetLocalForm(f,&local);
  VecGetArray(local,&uu);
  int nlocal;
  VecGetLocalSize(local,&nlocal);
  sidl::array<double> ua;
  int lower[4],upper[4],stride[4];
  lower[0] = 0; upper[0] = nlocal; stride[0] = 1;
  ua.borrow(uu,1,*&lower,*&upper,*&stride);
  system.computeRightHandSide(ua);
  VecRestoreArray(local,0);
  VecGhostRestoreLocalForm(f,&local);
  VecGhostUpdateBegin(f,ADD_VALUES,SCATTER_REVERSE);
  VecGhostUpdateEnd(f,ADD_VALUES,SCATTER_REVERSE);
  PetscFunctionReturn(0);
}
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._includes)

// user-defined constructor.
void TOPS::UnstructuredSolver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._ctor)
  this->dmmg = PETSC_NULL;
  this->bs   = 1;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._ctor)
}

// user-defined destructor.
void TOPS::UnstructuredSolver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._dtor)
  if (this->dmmg) {DMMGDestroy(this->dmmg);}
  if (this->startedpetsc) {
    PetscFinalize();
  }
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._dtor)
}

// static class initializer.
void TOPS::UnstructuredSolver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._load)
  // Insert-Code-Here {TOPS.UnstructuredSolver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSystem[]
 */
void
TOPS::UnstructuredSolver_impl::setSystem (
  /* in */ ::TOPS::System::System system ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setSystem)
  this->system = system;
  system.setSolver(this->self);
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setSystem)
}

/**
 * Method:  getSystem[]
 */
::TOPS::System::System
TOPS::UnstructuredSolver_impl::getSystem ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getSystem)
  return this->system;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getSystem)
}

/**
 * Method:  Initialize[]
 */
void
TOPS::UnstructuredSolver_impl::Initialize (
  /* in */ ::sidl::array< ::std::string> args ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.Initialize)
  PetscTruth initialized;
  PetscInitialized(&initialized);
  if (initialized) {
    this->startedpetsc = 0;
    return;
  }
  this->startedpetsc = 1;
  int          argc = args.upper(0) + 1;
  char       **argv = new char* [argc];
  std::string  arg;

  for(int i = 0; i < argc; i++) {
    arg     = args[i];
    argv[i] = new char [arg.length()+1];
    arg.copy(argv[i], arg.length(), 0);
    argv[i][arg.length()] = 0;
  }
  int    ierr = PetscInitialize(&argc,&argv,0,0);
  SlicedCreate(PETSC_COMM_WORLD,&this->slice);
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.Initialize)
}

/**
 * Method:  solve[]
 */
void
TOPS::UnstructuredSolver_impl::solve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.solve)
  PetscErrorCode ierr;

  if (!this->dmmg) {
    TOPS::System::Initialize::Once once = (TOPS::System::Initialize::Once)this->system;
    if (once._not_nil()) {    
      once.initializeOnce();
    }
    // create DMMG object 
    DMMGCreate(PETSC_COMM_WORLD,1,(void*)&this->self,&this->dmmg);
    DMMGSetDM(this->dmmg,(DM)this->slice);
    TOPS::System::Compute::Residual residual = (TOPS::System::Compute::Residual) this->system;
    if (residual._not_nil()) {
      ierr = DMMGSetSNES(this->dmmg, FormFunction, 0);
    } else {
      ierr = DMMGSetKSP(this->dmmg,FormRightHandSide,FormMatrix);
    }
    TOPS::System::Compute::InitialGuess guess = (TOPS::System::Compute::InitialGuess) this->system;
    if (guess._not_nil()) {
      ierr = DMMGSetInitialGuess(this->dmmg, FormInitialGuess);
    }
  }
  TOPS::System::Initialize::EverySolve every = (TOPS::System::Initialize::EverySolve)this->system;
  if (every._not_nil()) {    
    every.initializeEverySolve();
  }
  DMMGSolve(this->dmmg);
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.solve)
}

/**
 * Method:  setBlockSize[]
 */
void
TOPS::UnstructuredSolver_impl::setBlockSize (
  /* in */ int32_t bs ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setBlockSize)
  this->bs = bs;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setBlockSize)
}

/**
 * Method:  getSolution[]
 */
::sidl::array<double>
TOPS::UnstructuredSolver_impl::getSolution ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getSolution)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getSolution} (getSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getSolution)
}

/**
 * Method:  setSolution[]
 */
void
TOPS::UnstructuredSolver_impl::setSolution (
  /* in */ ::sidl::array<double> location ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setSolution)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setSolution} (setSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setSolution)
}

/**
 * Method:  setLocalSize[]
 */
void
TOPS::UnstructuredSolver_impl::setLocalSize (
  /* in */ int32_t m ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setLocalSize)
  this->n = m;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setLocalSize)
}

/**
 * Method:  getLocalSize[]
 */
int32_t
TOPS::UnstructuredSolver_impl::getLocalSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getLocalSize)
  return this->n;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getLocalSize)
}

/**
 * Method:  setGhostPoints[]
 */
void
TOPS::UnstructuredSolver_impl::setGhostPoints (
  /* in */ ::sidl::array<int32_t> ghosts ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setGhostPoints)
  SlicedSetGhosts(this->slice,this->bs,this->n,ghosts.length(0),ghosts.first());
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setGhostPoints)
}

/**
 * Method:  getGhostPoints[]
 */
::sidl::array<int32_t>
TOPS::UnstructuredSolver_impl::getGhostPoints ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getGhostPoints)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getGhostPoints} (getGhostPoints method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getGhostPoints)
}

/**
 * Method:  setPreallocation[]
 */
void
TOPS::UnstructuredSolver_impl::setPreallocation (
  /* in */ int32_t d,
  /* in */ int32_t od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setPreallocation)
  SlicedSetPreallocation(this->slice,d,PETSC_NULL,od,PETSC_NULL);
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setPreallocation)
}

/**
 * Method:  setPreallocation[s]
 */
void
TOPS::UnstructuredSolver_impl::setPreallocation (
  /* in */ ::sidl::array<int32_t> d,
  /* in */ ::sidl::array<int32_t> od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setPreallocations)
  SlicedSetPreallocation(this->slice,0,d.first(),0,od.first());
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setPreallocations)
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
TOPS::UnstructuredSolver_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setServices)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setServices} (setServices method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setServices)
}


// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._misc)
// Insert-Code-Here {TOPS.UnstructuredSolver._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._misc)

