// 
// File:          TOPS_Solver_Structured_Impl.cc
// Symbol:        TOPS.Solver_Structured-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Solver_Structured
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_Solver_Structured_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._includes)
#include "TOPS_Structured_Matrix_Impl.hh"
// Uses ports includes
  // This code is the same as DAVecGetArray() except instead of generating
  // raw C multidimensional arrays it gets a Babel array
::sidl::array<double> DAVecGetArrayBabel(DA da,Vec vec)
{
  double *uu;
  VecGetArray(vec,&uu);
  PetscInt  xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,dim,dof,N;
  DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);
  DAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);
  DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);

  /* Handle case where user passes in global vector as opposed to local */
  VecGetLocalSize(vec,&N);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  }

  sidl::array<double> ua;
  int lower[4],upper[4],stride[4];
  if (dof > 1) {
    dim++;
    lower[0] = 0; upper[0] = dof; stride[0] = 1;
    lower[1] = gxs; lower[2] = gys; lower[3] = gzs;
    upper[1] = gxm + gxs - 1; upper[2] = gym  + gys - 1; upper[3] = gzm + gzs - 1;
    stride[1] = dof; stride[2] = gxm*dof; stride[3] = gxm*gym*dof;
  } else {
    lower[0] = gxs; lower[1] = gys; lower[2] = gzs;
    upper[0] = gxm +gxs - 1; upper[1] = gym + gys - 1 ; upper[2] = gzm + gzs - 1;
    stride[0] = 1; stride[1] = gxm; stride[2] = gxm*gym;
  }
  ua.borrow(uu,dim,*&lower,*&upper,*&stride);
  return ua;
}

static PetscErrorCode FormFunction(SNES snes,Vec uu,Vec f,void *vdmmg)
{
  PetscFunctionBegin;
  DMMG dmmg = (DMMG) vdmmg;
  TOPS::Solver_Structured *solver = (TOPS::Solver_Structured*) dmmg->user;
  TOPS::System::Compute::Residual system = (TOPS::System::Compute::Residual) solver->getSystem();
  DA da = (DA) dmmg->dm;
  Vec u; 
  DAGetLocalVector(da,&u);
  DAGlobalToLocalBegin(da,uu,INSERT_VALUES,u);
  DAGlobalToLocalEnd(da,uu,INSERT_VALUES,u);

  int mx,my,mz;
  DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setDimensionX(mx);
  solver->setDimensionY(my);
  solver->setDimensionZ(mz);
  sidl::array<double> ua = DAVecGetArrayBabel(da,u);
  sidl::array<double> fa = DAVecGetArrayBabel(da,f);;
  system.computeResidual(ua,fa);
  VecRestoreArray(u,0);
  DARestoreLocalVector(da,&u);
  VecRestoreArray(f,0);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialGuess(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  TOPS::Solver_Structured *solver = (TOPS::Solver_Structured*) dmmg->user;
  TOPS::System::Compute::InitialGuess system = (TOPS::System::Compute::InitialGuess) solver->getSystem();

  int mx,my,mz;
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setDimensionX(mx);
  solver->setDimensionY(my);
  solver->setDimensionZ(mz);
  sidl::array<double> fa = DAVecGetArrayBabel((DA)dmmg->dm,f);;
  system.computeInitialGuess(fa);
  VecRestoreArray(f,0);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormMatrix(DMMG dmmg,Mat J,Mat B)
{
  PetscFunctionBegin;
  TOPS::Solver_Structured *solver = (TOPS::Solver_Structured*) dmmg->user;
  TOPS::System::Compute::Matrix system = (TOPS::System::Compute::Matrix) solver->getSystem();
  TOPS::Structured::Matrix matrix1 = TOPS::Structured::Matrix::_create();
  TOPS::Structured::Matrix matrix2 = TOPS::Structured::Matrix::_create();

  PetscInt  xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,dim,dof,mx,my,mz;
  DAGetCorners((DA)dmmg->dm,&xs,&ys,&zs,&xm,&ym,&zm);
  DAGetGhostCorners((DA)dmmg->dm,&gxs,&gys,&gzs,&gxm,&gym,&gzm);
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);
#define GetImpl(A,b) (!(A)b) ? 0 : reinterpret_cast<A ## _impl*>(((A) b)._get_ior()->d_data)

  // currently no support for dof > 1
  TOPS::Structured::Matrix_impl *imatrix1 = GetImpl(TOPS::Structured::Matrix,matrix1);
  imatrix1->vlength[0] = xm; imatrix1->vlength[1] = ym; imatrix1->vlength[2] = zm; 
  imatrix1->vlower[0] = xs; imatrix1->vlower[1] = ys; imatrix1->vlower[2] = zs; 
  imatrix1->gghostlength[0] = gxm; imatrix1->gghostlength[1] = gym; imatrix1->gghostlength[2] = gzm; 
  imatrix1->gghostlower[0] = gxs; imatrix1->gghostlower[1] = gys; imatrix1->gghostlower[2] = gzs; 
  imatrix1->vdimen = dof;

  TOPS::Structured::Matrix_impl *imatrix2 = GetImpl(TOPS::Structured::Matrix,matrix2);
  imatrix2->vlength[0] = xm; imatrix2->vlength[1] = ym; imatrix2->vlength[2] = zm; 
  imatrix2->vlower[0] = xs; imatrix2->vlower[1] = ys; imatrix2->vlower[2] = zs; 
  imatrix2->gghostlength[0] = gxm; imatrix2->gghostlength[1] = gym; imatrix2->gghostlength[2] = gzm; 
  imatrix2->gghostlower[0] = gxs; imatrix2->gghostlower[1] = gys; imatrix2->gghostlower[2] = gzs; 
  imatrix2->vdimen = dof;

  imatrix1->mat = J;
  imatrix2->mat = B;
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setDimensionX(mx);
  solver->setDimensionY(my);
  solver->setDimensionZ(mz);

  system.computeMatrix(matrix1,matrix2);
  MatAssemblyBegin(dmmg->B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dmmg->B,MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRightHandSide(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  TOPS::Solver_Structured *solver = (TOPS::Solver_Structured*) dmmg->user;
  TOPS::System::Compute::RightHandSide system = (TOPS::System::Compute::RightHandSide) solver->getSystem();

  int mx,my,mz;
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setDimensionX(mx);
  solver->setDimensionY(my);
  solver->setDimensionZ(mz);
  sidl::array<double> fa = DAVecGetArrayBabel((DA)dmmg->dm,f);;
  system.computeRightHandSide(fa);
  VecRestoreArray(f,0);
  PetscFunctionReturn(0);
}
// DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._includes)

// user-defined constructor.
void TOPS::Solver_Structured_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._ctor)
  this->dmmg = PETSC_NULL;
  this->da   = PETSC_NULL;
  this->m    = PETSC_DECIDE;
  this->n    = PETSC_DECIDE;
  this->p    = PETSC_DECIDE;
  this->M    = 3;
  this->N    = 3;
  this->P    = 3;
  this->dim  = 2;
  this->s    = 1;
  this->wrap = DA_NONPERIODIC;
  this->bs   = 1;
  this->stencil_type = DA_STENCIL_STAR;
  this->levels       = 3;
  this->system       = PETSC_NULL;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._ctor)
}

// user-defined destructor.
void TOPS::Solver_Structured_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._dtor)
  if (this->dmmg) {DMMGDestroy(this->dmmg);}
  if (this->startedpetsc) {
    PetscFinalize();
  }
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._dtor)
}

// static class initializer.
void TOPS::Solver_Structured_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._load)
  // Insert-Code-Here {TOPS.Solver_Structured._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSystem[]
 */
void
TOPS::Solver_Structured_impl::setSystem (
  /* in */ ::TOPS::System::System system ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setSystem)
  this->system = system;
  system.setSolver(this->self);
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setSystem)
}

/**
 * Method:  getSystem[]
 */
::TOPS::System::System
TOPS::Solver_Structured_impl::getSystem ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getSystem)
  return this->system;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getSystem)
}

/**
 * Method:  Initialize[]
 */
void
TOPS::Solver_Structured_impl::Initialize (
  /* in */ ::sidl::array< ::std::string> args ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.Initialize)
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
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.Initialize)
}

/**
 * Method:  solve[]
 */
void
TOPS::Solver_Structured_impl::solve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.solve)
  PetscErrorCode ierr;

  if (!this->dmmg) {
    this->system.initializeOnce();
    // create DMMG object 
    DMMGCreate(PETSC_COMM_WORLD,this->levels,(void*)&this->self,&this->dmmg);
    DACreate(PETSC_COMM_WORLD,this->dim,this->wrap,this->stencil_type,this->M,this->N,this->P,this->m,this->n,
             this->p,this->bs,this->s,PETSC_NULL,PETSC_NULL,PETSC_NULL,&this->da);
    DMMGSetDM(this->dmmg,(DM)this->da);
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
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.solve)
}

/**
 * Method:  setBlockSize[]
 */
void
TOPS::Solver_Structured_impl::setBlockSize (
  /* in */ int32_t bs ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setBlockSize)
  this->bs = bs;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setBlockSize)
}

/**
 * Method:  getSolution[]
 */
::sidl::array<double>
TOPS::Solver_Structured_impl::getSolution ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getSolution)
  // Insert-Code-Here {TOPS.Solver_Structured.getSolution} (getSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getSolution)
}

/**
 * Method:  setSolution[]
 */
void
TOPS::Solver_Structured_impl::setSolution (
  /* in */ ::sidl::array<double> location ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setSolution)
  // Insert-Code-Here {TOPS.Solver_Structured.setSolution} (setSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setSolution)
}

/**
 * Method:  setDimension[]
 */
void
TOPS::Solver_Structured_impl::setDimension (
  /* in */ int32_t dim ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setDimension)
  this->dim = dim;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setDimension)
}

/**
 * Method:  getDimension[]
 */
int32_t
TOPS::Solver_Structured_impl::getDimension ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getDimension)
  return this->dim;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getDimension)
}

/**
 * Method:  setDimensionX[]
 */
void
TOPS::Solver_Structured_impl::setDimensionX (
  /* in */ int32_t dim ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setDimensionX)
  this->M = dim;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setDimensionX)
}

/**
 * Method:  getDimensionX[]
 */
int32_t
TOPS::Solver_Structured_impl::getDimensionX ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getDimensionX)
  return this->M;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getDimensionX)
}

/**
 * Method:  setDimensionY[]
 */
void
TOPS::Solver_Structured_impl::setDimensionY (
  /* in */ int32_t dim ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setDimensionY)
  this->N = dim;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setDimensionY)
}

/**
 * Method:  getDimensionY[]
 */
int32_t
TOPS::Solver_Structured_impl::getDimensionY ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getDimensionY)
  return this->N;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getDimensionY)
}

/**
 * Method:  setDimensionZ[]
 */
void
TOPS::Solver_Structured_impl::setDimensionZ (
  /* in */ int32_t dim ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setDimensionZ)
  this->P = dim;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setDimensionZ)
}

/**
 * Method:  getDimensionZ[]
 */
int32_t
TOPS::Solver_Structured_impl::getDimensionZ ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getDimensionZ)
  return this->P;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getDimensionZ)
}

/**
 * Method:  setStencilWidth[]
 */
void
TOPS::Solver_Structured_impl::setStencilWidth (
  /* in */ int32_t width ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setStencilWidth)
  // Insert-Code-Here {TOPS.Solver_Structured.setStencilWidth} (setStencilWidth method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setStencilWidth)
}

/**
 * Method:  getStencilWidth[]
 */
int32_t
TOPS::Solver_Structured_impl::getStencilWidth ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.getStencilWidth)
  // Insert-Code-Here {TOPS.Solver_Structured.getStencilWidth} (getStencilWidth method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.getStencilWidth)
}

/**
 * Method:  setLevels[]
 */
void
TOPS::Solver_Structured_impl::setLevels (
  /* in */ int32_t levels ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setLevels)
  this->levels = levels;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setLevels)
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
TOPS::Solver_Structured_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured.setServices)
  // Insert-Code-Here {TOPS.Solver_Structured.setServices} (setServices method)

  myServices = services;
  gov::cca::TypeMap tm = services.createTypeMap();
  if(tm._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: gov::cca::TypeMap is nil\n",
	    __FILE__, __LINE__);
    exit(1);
  }
  gov::cca::Port p = self;      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting self to gov::cca::Port \n",
	    __FILE__, __LINE__);
    exit(1);
  }
  
  // Provides port
  services.addProvidesPort(p,
			   "TOPS.Structured.Solver",
			   "TOPS.Structured.Solver", tm);
  
  // Uses ports
  services.registerUsesPort("TOPS.System.Compute.InitialGuess",
			    "TOPS.System.Compute.InitialGuess", tm);

  services.registerUsesPort("TOPS.System.Compute.Matrix",
			    "TOPS.System.Compute.Matrix", tm);

  services.registerUsesPort("TOPS.System.Compute.RightHandSide",
			    "TOPS.System.Compute.RightHandSide", tm);

  services.registerUsesPort("TOPS.System.Compute.Residual",
			    "TOPS.System.Compute.Residual", tm);

  return;
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Structured.setServices)
}


// DO-NOT-DELETE splicer.begin(TOPS.Solver_Structured._misc)
// Insert-Code-Here {TOPS.Solver_Structured._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Structured._misc)

