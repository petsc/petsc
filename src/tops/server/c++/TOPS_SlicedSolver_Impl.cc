// 
// File:          TOPS_SlicedSolver_Impl.cc
// Symbol:        TOPS.SlicedSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.SlicedSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_SlicedSolver_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver._includes)
#include "TOPS_Sliced_Matrix_Impl.hh"
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
  TOPS::SlicedSolver *solver = (TOPS::SlicedSolver*) dmmg->user;
  TOPS::System::Compute::Matrix system = (TOPS::System::Compute::Matrix) solver->getSystem();
  TOPS::Sliced::Matrix matrix1 = TOPS::Sliced::Matrix::_create();
  TOPS::Sliced::Matrix matrix2 = TOPS::Sliced::Matrix::_create();

#define GetImpl(A,b) (!(A)b) ? 0 : reinterpret_cast<A ## _impl*>(((A) b)._get_ior()->d_data)

  // currently no support for dof > 1
  TOPS::Sliced::Matrix_impl *imatrix1 = GetImpl(TOPS::Sliced::Matrix,matrix1);
  TOPS::Sliced::Matrix_impl *imatrix2 = GetImpl(TOPS::Sliced::Matrix,matrix2);
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
  TOPS::SlicedSolver *solver = (TOPS::SlicedSolver*) dmmg->user;
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
// DO-NOT-DELETE splicer.end(TOPS.SlicedSolver._includes)

// user-defined constructor.
void TOPS::SlicedSolver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver._ctor)
  this->dmmg = PETSC_NULL;
  this->bs   = 1;
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver._ctor)
}

// user-defined destructor.
void TOPS::SlicedSolver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver._dtor)
  if (this->dmmg) {DMMGDestroy(this->dmmg);}
  if (this->startedpetsc) {
    PetscFinalize();
  }
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver._dtor)
}

// static class initializer.
void TOPS::SlicedSolver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver._load)
  // Insert-Code-Here {TOPS.SlicedSolver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSystem[]
 */
void
TOPS::SlicedSolver_impl::setSystem (
  /* in */ ::TOPS::System::System system ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setSystem)
  this->system = system;
  system.setSolver(this->self);
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setSystem)
}

/**
 * Method:  getSystem[]
 */
::TOPS::System::System
TOPS::SlicedSolver_impl::getSystem ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.getSystem)
  return this->system;
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.getSystem)
}

/**
 * Method:  Initialize[]
 */
void
TOPS::SlicedSolver_impl::Initialize (
  /* in */ ::sidl::array< ::std::string> args ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.Initialize)
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
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.Initialize)
}

/**
 * Method:  solve[]
 */
void
TOPS::SlicedSolver_impl::solve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.solve)
  PetscErrorCode ierr;

  if (!this->dmmg) {
    this->system.initializeOnce();
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
  this->system.initializeEverySolve();
  DMMGSolve(this->dmmg);
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.solve)
}

/**
 * Method:  setBlockSize[]
 */
void
TOPS::SlicedSolver_impl::setBlockSize (
  /* in */ int32_t bs ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setBlockSize)
  this->bs = bs;
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setBlockSize)
}

/**
 * Method:  getSolution[]
 */
::sidl::array<double>
TOPS::SlicedSolver_impl::getSolution ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.getSolution)
  // Insert-Code-Here {TOPS.SlicedSolver.getSolution} (getSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.getSolution)
}

/**
 * Method:  setSolution[]
 */
void
TOPS::SlicedSolver_impl::setSolution (
  /* in */ ::sidl::array<double> location ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setSolution)
  // Insert-Code-Here {TOPS.SlicedSolver.setSolution} (setSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setSolution)
}

/**
 * Method:  setLocalSize[]
 */
void
TOPS::SlicedSolver_impl::setLocalSize (
  /* in */ int32_t m ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setLocalSize)
  this->n = m;
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setLocalSize)
}

/**
 * Method:  getLocalSize[]
 */
int32_t
TOPS::SlicedSolver_impl::getLocalSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.getLocalSize)
  return this->n;
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.getLocalSize)
}

/**
 * Method:  setGhostPoints[]
 */
void
TOPS::SlicedSolver_impl::setGhostPoints (
  /* in */ ::sidl::array<int32_t> ghosts ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setGhostPoints)
  SlicedSetGhosts(this->slice,this->bs,this->n,ghosts.length(0),ghosts.first());
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setGhostPoints)
}

/**
 * Method:  setPreallocation[]
 */
void
TOPS::SlicedSolver_impl::setPreallocation (
  /* in */ int32_t d,
  /* in */ int32_t od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setPreallocation)
  SlicedSetPreallocation(this->slice,d,PETSC_NULL,od,PETSC_NULL);
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setPreallocation)
}

/**
 * Method:  setPreallocation[s]
 */
void
TOPS::SlicedSolver_impl::setPreallocation (
  /* in */ ::sidl::array<int32_t> d,
  /* in */ ::sidl::array<int32_t> od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setPreallocations)
  SlicedSetPreallocation(this->slice,0,d.first(),0,od.first());
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setPreallocations)
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
TOPS::SlicedSolver_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver.setServices)
  // Insert-Code-Here {TOPS.SlicedSolver.setServices} (setServices method)
  // DO-NOT-DELETE splicer.end(TOPS.SlicedSolver.setServices)
}


// DO-NOT-DELETE splicer.begin(TOPS.SlicedSolver._misc)
// Insert-Code-Here {TOPS.SlicedSolver._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.SlicedSolver._misc)

