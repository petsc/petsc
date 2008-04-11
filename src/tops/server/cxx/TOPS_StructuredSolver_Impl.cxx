// 
// File:          TOPS_StructuredSolver_Impl.cxx
// Symbol:        TOPS.StructuredSolver-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.StructuredSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "TOPS_StructuredSolver_Impl.hxx"

// 
// Includes for all method dependencies.
// 
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
// DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._includes)

#include <iostream>
#include "TOPS_StructuredMatrix_Impl.hxx"
#include "TOPS_ParameterHandling.hxx"  // not from SIDL
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

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
static PetscErrorCode FormFunction(SNES snes,Vec uu,Vec f,void *vdmmg)
{
  PetscFunctionBegin;
  DMMG dmmg = (DMMG) vdmmg;
  TOPS::StructuredSolver *solver = (TOPS::StructuredSolver*) dmmg->user;
  TOPS::System::Compute::Residual system;
  system = ::babel_cast< TOPS::System::Compute::Residual >(
  	solver->getServices().getPort("TOPS.System.Compute.Residual"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.System.Compute.Residual port is nil, " 
	      << "possibly not connected." << std::endl;
    PetscFunctionReturn(1);
  }

  DA da = (DA) dmmg->dm;
  Vec u; 
  DAGetLocalVector(da,&u);
  DAGlobalToLocalBegin(da,uu,INSERT_VALUES,u);
  DAGlobalToLocalEnd(da,uu,INSERT_VALUES,u);

  int mx,my,mz;
  DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setLength(0,mx);
  solver->setLength(1,my);
  solver->setLength(2,mz);
  sidl::array<double> ua = DAVecGetArrayBabel(da,u);
  sidl::array<double> fa = DAVecGetArrayBabel(da,f);;
  system.computeResidual(ua,fa);
  VecRestoreArray(u,0);
  DARestoreLocalVector(da,&u);
  VecRestoreArray(f,0);

  solver->getServices().releasePort("TOPS.System.Compute.Residual");
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialGuess(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  TOPS::StructuredSolver *solver = (TOPS::StructuredSolver*) dmmg->user;
  TOPS::System::Compute::InitialGuess system;

  system = ::babel_cast< TOPS::System::Compute::InitialGuess >(
  	solver->getServices().getPort("TOPS.System.Compute.InitialGuess"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.System.Compute.InitialGuess port is nil, " 
	      << "possibly not connected." << std::endl;
    PetscFunctionReturn(1);
  }

  int mx,my,mz;
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setLength(0,mx);
  solver->setLength(1,my);
  solver->setLength(2,mz);
  sidl::array<double> fa = DAVecGetArrayBabel((DA)dmmg->dm,f);

  system.computeInitialGuess(fa);
  VecRestoreArray(f,0);
  solver->getServices().releasePort("TOPS.System.Compute.InitialGuess");
  PetscFunctionReturn(0);
}

static PetscErrorCode FormMatrix(DMMG dmmg,Mat J,Mat B)
{
  PetscFunctionBegin;
  TOPS::StructuredSolver *solver = (TOPS::StructuredSolver*) dmmg->user;
  TOPS::System::Compute::Matrix system;

  // Replace following with ports
  //TOPS::Structured::Matrix matrix1 = TOPS::Structured::Matrix::_create();
  //TOPS::Structured::Matrix matrix2 = TOPS::Structured::Matrix::_create();

  PetscInt  xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,dof,mx,my,mz;
  DAGetCorners((DA)dmmg->dm,&xs,&ys,&zs,&xm,&ym,&zm);
  DAGetGhostCorners((DA)dmmg->dm,&gxs,&gys,&gzs,&gxm,&gym,&gzm);
  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);
  
  // Jacobian settings
  int32_t l[1] = {0}, u[1] = {2};
  int32_t lower_data[3] = {xs, ys, zs}, glower_data[3] = {gxs, gys, gzs};	
  int32_t length_data[3] = {xm, ym, zm}, glength_data[3] = {gxm, gym, gzm};
    
  sidl::array<int32_t> lower = sidl::array<int32_t>::createRow(1,l,u);
  sidl::array<int32_t> length = sidl::array<int32_t>::createRow(1,l,u);
  sidl::array<int32_t> glower = sidl::array<int32_t>::createRow(1,l,u);
  sidl::array<int32_t> glength = sidl::array<int32_t>::createRow(1,l,u);
  // Populate arrays
  for (int i = l[0]; i < u[0]; ++i) {
  	lower.set(i, lower_data[i]);
  	length.set(i, length_data[i]);
  	glower.set(i, glower_data[i]);
  	glength.set(i, glength_data[i]);
  }
  
  // Use the TOPS.Structured.Matrix port to configure J
  TOPS::Structured::Matrix JMatrix;
  JMatrix = ::babel_cast< TOPS::Structured::Matrix >(
  	solver->getServices().getPort("JacobianMatrix"));
  if (JMatrix._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.Structured.Matrix port is nil, could not connect to Jacobian component." 
	      << std::endl;
    PetscFunctionReturn(1);
  }
  JMatrix.setDimen(dof);
  JMatrix.setLower(lower);
  JMatrix.setLength(length);
  JMatrix.setGhostLower(glower);
  JMatrix.setGhostLength(glength);
  JMatrix.setMat(J);
 
  // Use the TOPS.Structured.Matrix port to configure B
  TOPS::Structured::Matrix BMatrix;
  BMatrix = ::babel_cast< TOPS::Structured::Matrix >(
  	solver->getServices().getPort("PreconditionerMatrix"));
  if (BMatrix._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.Structured.Matrix port is nil, could not connect to matrix B component." 
	      << std::endl;
    PetscFunctionReturn(1);
  }
  BMatrix.setDimen(dof);
  BMatrix.setLower(lower);
  BMatrix.setLength(length);
  BMatrix.setGhostLower(glower);
  BMatrix.setGhostLength(glength);
  BMatrix.setMat(B);
   
//#define GetImpl(A,b) (!(A)b) ? 0 : reinterpret_cast<A ## _impl*>(((A) b)._get_ior()->d_data)
//
//  // currently no support for dof > 1
//  TOPS::Structured::Matrix_impl *imatrix1 = GetImpl(TOPS::Structured::Matrix,matrix1);
//  imatrix1->vlength[0] = xm; imatrix1->vlength[1] = ym; imatrix1->vlength[2] = zm; 
//  imatrix1->vlower[0] = xs; imatrix1->vlower[1] = ys; imatrix1->vlower[2] = zs; 
//  imatrix1->gghostlength[0] = gxm; imatrix1->gghostlength[1] = gym; 
//  imatrix1->gghostlength[2] = gzm; 
//  imatrix1->gghostlower[0] = gxs; imatrix1->gghostlower[1] = gys; 
//  imatrix1->gghostlower[2] = gzs; 
//  imatrix1->vdimen = dof;
//
//  TOPS::Structured::Matrix_impl *imatrix2 = GetImpl(TOPS::Structured::Matrix,matrix2);
//  imatrix2->vlength[0] = xm; imatrix2->vlength[1] = ym; imatrix2->vlength[2] = zm; 
//  imatrix2->vlower[0] = xs; imatrix2->vlower[1] = ys; imatrix2->vlower[2] = zs; 
//  imatrix2->gghostlength[0] = gxm; imatrix2->gghostlength[1] = gym; 
//  imatrix2->gghostlength[2] = gzm; 
//  imatrix2->gghostlower[0] = gxs; imatrix2->gghostlower[1] = gys; 
//  imatrix2->gghostlower[2] = gzs; 
//  imatrix2->vdimen = dof;
//
//  imatrix1->mat = J;
//  imatrix2->mat = B;

  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setLength(0,mx);
  solver->setLength(1,my);
  solver->setLength(2,mz);

  system = ::babel_cast< TOPS::System::Compute::Matrix >(
  	solver->getServices().getPort("TOPS.System.Compute.Matrix"));
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.System.Compute.Matrix port is nil, " 
	      << "possibly not connected." << std::endl;
    PetscFunctionReturn(1);
  }

  // Use the port
  if (system._not_nil()) {
    system.computeMatrix(::babel_cast<  ::TOPS::Matrix>(JMatrix),
    					::babel_cast< ::TOPS::Matrix>(BMatrix) );
  }
 
  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  if (J != B) {
    MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  }
  
  solver->getServices().releasePort("TOPS.System.Compute.Matrix");
  solver->getServices().releasePort("JacobianMatrix");
  solver->getServices().releasePort("PreconditionerMatrix");
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormRightHandSide"
static PetscErrorCode FormRightHandSide(DMMG dmmg,Vec f)
{
  PetscFunctionBegin;
  int mx,my,mz;
  TOPS::StructuredSolver *solver = (TOPS::StructuredSolver*) dmmg->user;
  TOPS::System::Compute::RightHandSide system;

  system = ::babel_cast< TOPS::System::Compute::RightHandSide >(
  	solver->getServices().getPort("TOPS.System.Compute.RightHandSide"));

  DAGetInfo((DA)dmmg->dm,0,&mx,&my,&mz,0,0,0,0,0,0,0);
  solver->setLength(0,mx);
  solver->setLength(1,my);
  solver->setLength(2,mz);
  sidl::array<double> fa = DAVecGetArrayBabel((DA)dmmg->dm,f);;

  if (system._not_nil()) {
    system.computeRightHandSide(fa);
  }

  solver->getServices().releasePort("TOPS.System.Compute.RightHandSide");

  VecRestoreArray(f,0);
  PetscFunctionReturn(0);
}
// DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
TOPS::StructuredSolver_impl::StructuredSolver_impl() : StubBase(
  reinterpret_cast< void*>(::TOPS::StructuredSolver::_wrapObj(reinterpret_cast< 
  void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._ctor2)
  // Insert-Code-Here {TOPS.StructuredSolver._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._ctor2)
}

// user defined constructor
void TOPS::StructuredSolver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._ctor)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::_ctor()"

  this->dmmg = PETSC_NULL;
  this->da   = PETSC_NULL;
  this->m    = PETSC_DECIDE;
  this->n    = PETSC_DECIDE;
  this->p    = PETSC_DECIDE;
  this->lengths[0] = 3;
  this->lengths[1] = 3;
  this->lengths[2] = 3;
  this->dim  = 2;
  this->s    = 1;
  this->wrap = DA_NONPERIODIC;
  this->bs   = 1;
  this->stencil_type = DA_STENCIL_STAR;
  this->levels       = 3;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._ctor)
}

// user defined destructor
void TOPS::StructuredSolver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._dtor)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::_dtor()"

  if (this->dmmg) {DMMGDestroy(this->dmmg);}
  if (this->startedpetsc) {
    PetscFinalize();
  }
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._dtor)
}

// static class initializer
void TOPS::StructuredSolver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._load)
  // Insert-Code-Here {TOPS.StructuredSolver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getServices[]
 */
::gov::cca::Services
TOPS::StructuredSolver_impl::getServices_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.getServices)
  // Insert-Code-Here {TOPS.StructuredSolver.getServices} (getServices method)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::getServices()"

  return this->myServices;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.getServices)
}

/**
 * Method:  dimen[]
 */
int32_t
TOPS::StructuredSolver_impl::dimen_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.dimen)
  return dim;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.dimen)
}

/**
 * Method:  length[]
 */
int32_t
TOPS::StructuredSolver_impl::length_impl (
  /* in */int32_t a ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.length)
  return this->lengths[a];
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.length)
}

/**
 * Method:  setDimen[]
 */
void
TOPS::StructuredSolver_impl::setDimen_impl (
  /* in */int32_t dim ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setDimen)
  this->dim = dim;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setDimen)
}

/**
 * Method:  setLength[]
 */
void
TOPS::StructuredSolver_impl::setLength_impl (
  /* in */int32_t a,
  /* in */int32_t l ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setLength)
  this->lengths[a] = l;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setLength)
}

/**
 * Method:  setStencilWidth[]
 */
void
TOPS::StructuredSolver_impl::setStencilWidth_impl (
  /* in */int32_t width ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setStencilWidth)
  // Insert-Code-Here {TOPS.StructuredSolver.setStencilWidth} (setStencilWidth method)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setStencilWidth)
}

/**
 * Method:  getStencilWidth[]
 */
int32_t
TOPS::StructuredSolver_impl::getStencilWidth_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.getStencilWidth)
  // Insert-Code-Here {TOPS.StructuredSolver.getStencilWidth} (getStencilWidth method)
  return 0;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.getStencilWidth)
}

/**
 * Method:  setLevels[]
 */
void
TOPS::StructuredSolver_impl::setLevels_impl (
  /* in */int32_t levels ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setLevels)
  this->levels = levels;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setLevels)
}

/**
 * Method:  Initialize[]
 */
void
TOPS::StructuredSolver_impl::Initialize_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.Initialize)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::Initialize"
  PetscTruth initialized;
  PetscInitialized(&initialized);
  if (initialized) {
    this->startedpetsc = 0;
    return;
  }
  this->startedpetsc = 1;
  PetscInitializeNoArguments(); 

  // Process runtime parameters
  params = ::babel_cast< gov::cca::ports::ParameterPort >( myServices.getPort("tops_options") );
  std::string options = params.readConfigurationMap().getString("options","-help");
  processTOPSOptions(options);

  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.Initialize)
}

/**
 * Method:  solve[]
 */
void
TOPS::StructuredSolver_impl::solve_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.solve)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::solve()"
  PetscErrorCode ierr;

  if (!this->dmmg) {
    TOPS::System::Initialize::Once once;
    once = ::babel_cast< TOPS::System::Initialize::Once >( myServices.getPort("TOPS.System.Initialize.Once"));
    if (once._not_nil()) {    
      once.initializeOnce();
    }
    myServices.releasePort("TOPS.System.Initialize.Once");

    // create DMMG object 
    DMMGCreate(PETSC_COMM_WORLD,this->levels,(void*)this,&this->dmmg);
    DACreate(PETSC_COMM_WORLD,this->dim,this->wrap,this->stencil_type,this->lengths[0],
	     this->lengths[1],this->lengths[2],this->m,this->n,
             this->p,this->bs,this->s,PETSC_NULL,PETSC_NULL,PETSC_NULL,&this->da);
    DMMGSetDM(this->dmmg,(DM)this->da);

    TOPS::System::Compute::Residual residual;
    residual = ::babel_cast< TOPS::System::Compute::Residual >( myServices.getPort("TOPS.System.Compute.Residual"));
    if (residual._not_nil()) {
      ierr = DMMGSetSNES(this->dmmg, FormFunction, 0);
      ierr = DMMGSetFromOptions(this->dmmg);
    } else {
      ierr = DMMGSetKSP(this->dmmg,FormRightHandSide,FormMatrix);
    }
    myServices.releasePort("TOPS.System.Compute.Residual");

    TOPS::System::Compute::InitialGuess guess;
    guess = ::babel_cast< TOPS::System::Compute::InitialGuess >( myServices.getPort("TOPS.System.Compute.InitialGuess") );

    if (guess._not_nil()) {
      ierr = DMMGSetInitialGuess(this->dmmg, FormInitialGuess);
    }
  }
  myServices.releasePort("TOPS.System.Compute.InitialGuess");
  
  TOPS::System::Initialize::EverySolve every;
  every = ::babel_cast< TOPS::System::Initialize::EverySolve >( myServices.getPort("TOPS.System.Initialize.EverySolve") );
  if (every._not_nil()) {    
    every.initializeEverySolve();
  }
  myServices.releasePort("TOPS.System.Initialize.EverySolve");

  DMMGSolve(this->dmmg);
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.solve)
}

/**
 * Method:  setBlockSize[]
 */
void
TOPS::StructuredSolver_impl::setBlockSize_impl (
  /* in */int32_t bs ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setBlockSize)
  this->bs = bs;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setBlockSize)
}

/**
 * Method:  getSolution[]
 */
::sidl::array<double>
TOPS::StructuredSolver_impl::getSolution_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.getSolution)
  // Insert-Code-Here {TOPS.StructuredSolver.getSolution} (getSolution method)
  return 0;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.getSolution)
}

/**
 * Method:  setSolution[]
 */
void
TOPS::StructuredSolver_impl::setSolution_impl (
  /* in array<double> */::sidl::array<double> location ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setSolution)
  // Insert-Code-Here {TOPS.StructuredSolver.setSolution} (setSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setSolution)
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
TOPS::StructuredSolver_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.setServices)

#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::setServices"

  myServices = services;
  gov::cca::TypeMap tm = services.createTypeMap();
  if(tm._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: gov::cca::TypeMap is nil\n",
	    __FILE__, __LINE__);
    exit(1);
  }
  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting (*this) to gov::cca::Port \n",
	    __FILE__, __LINE__);
    exit(1);
  }
  
  // Provides port
  services.addProvidesPort(p,
			   "TOPS.Structured.Solver",
			   "TOPS.Structured.Solver", tm);
  
  // Uses ports
  services.registerUsesPort("JacobianMatrix",
			    "TOPS.Structured.Matrix", tm);

  services.registerUsesPort("PreconditionerMatrix",
			    "TOPS.Structured.Matrix", tm);

  services.registerUsesPort("TOPS.System.Initialize.Once",
			    "TOPS.System.Initialize.Once", tm);

  services.registerUsesPort("TOPS.System.Initialize.EverySolve",
			    "TOPS.System.Initialize.EverySolve", tm);

  services.registerUsesPort("TOPS.System.Compute.InitialGuess",
			    "TOPS.System.Compute.InitialGuess", tm);

  services.registerUsesPort("TOPS.System.Compute.Matrix",
			    "TOPS.System.Compute.Matrix", tm);

  services.registerUsesPort("TOPS.System.Compute.RightHandSide",
			    "TOPS.System.Compute.RightHandSide", tm);

  services.registerUsesPort("TOPS.System.Compute.Jacobian",
			    "TOPS.System.Compute.Jacobian", tm);

  services.registerUsesPort("TOPS.System.Compute.Residual",
  			  "TOPS.System.Compute.Residual", tm);
  
  // Parameter port
  myServices.registerUsesPort(std::string("ParameterPortFactory"),
			      std::string("gov.cca.ports.ParameterPortFactory"), tm);

  // Set up parameter port
  if (this->setupParameterPort() != 0) {
    std::cerr << "TOPS::StructuredSolver_impl::go: errors during setup of ParameterPort" << std::endl;
  }

  myServices.unregisterUsesPort(std::string("ParameterPortFactory"));

  return;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.setServices)
}

/**
 *  Inform the listener that someone is about to fetch their 
 * typemap. The return should be true if the listener
 * has changed the ParameterPort definitions.
 */
bool
TOPS::StructuredSolver_impl::updateParameterPort_impl (
  /* in */const ::std::string& portName ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.updateParameterPort)
  // Insert-Code-Here {TOPS.StructuredSolver.updateParameterPort} (updateParameterPort method)

  std::cout << "TOPS::StructuredSolver_impl::updatedParameterPort called" << std::endl;
  // Get the runtime parameters
  params = ::babel_cast< gov::cca::ports::ParameterPort >( myServices.getPort("tops_options") );
  std::string options = params.readConfigurationMap().getString("options","-help");
  processTOPSOptions(options);
  SNES snes = DMMGGetSNES(dmmg);
  if (snes) SNESSetFromOptions(snes);
  return true;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.updateParameterPort)
}

/**
 *  The component wishing to be told after a parameter is changed
 * implements this function.
 * @param portName the name of the port (typemap) on which the
 * value was set.
 * @param fieldName the name of the value in the typemap.
 */
void
TOPS::StructuredSolver_impl::updatedParameterValue_impl (
  /* in */const ::std::string& portName,
  /* in */const ::std::string& fieldName ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver.updatedParameterValue)
  // Insert-Code-Here {TOPS.StructuredSolver.updatedParameterValue} (updatedParameterValue method)
  std::cout << "TOPS::StructuredSolver_impl::updatedParameterValue called" << std::endl;
  PetscTruth flg;
  PetscInt ierr;
  ierr = PetscInitialized(&flg); 
  if (!flg) ierr = PetscInitializeNoArguments();
  params = ::babel_cast< gov::cca::ports::ParameterPort >( myServices.getPort("tops_options") );
  std::string options = params.readConfigurationMap().getString("options","-help");
  processTOPSOptions(options);
  SNES snes = DMMGGetSNES(dmmg);
  if (snes)  SNESSetFromOptions(snes);
  // DO-NOT-DELETE splicer.end(TOPS.StructuredSolver.updatedParameterValue)
}


// DO-NOT-DELETE splicer.begin(TOPS.StructuredSolver._misc)
// Insert-Code-Here {TOPS.StructuredSolver._misc} (miscellaneous code)
int TOPS::StructuredSolver_impl::setupParameterPort() {

#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredSolver_impl::setupParameterPort"

  // First, get parameters
  ppf = ::babel_cast< gov::cca::ports::ParameterPortFactory >( myServices.getPort("ParameterPortFactory") );
  if (ppf._is_nil()) {
    std::cerr << "TOPS::StructuredSolver_impl::setupParameterPort: called without ParameterPortFactory connected." << std::endl;
    return -1;
  }
  gov::cca::TypeMap tm = myServices.createTypeMap();
  if (tm._is_nil()) {
    std::cerr << "TOPS::StructuredSolver_impl::setupParameterPort: myServices.createTypeMap failed." << std::endl;
    return -1;
  }

  ppf.initParameterData(tm, "tops_options");
  ppf.setBatchTitle(tm, "TOPS Options");
  ppf.addRequestString(tm, "options", "Space-separated list of TOPS options", 
			  "Enter runtime TOPS options", "-help");

  // We may want to respond to changes
  gov::cca::ports::ParameterSetListener paramSetListener = (*this);
  ppf.registerUpdatedListener(tm, paramSetListener);

  // We may want to change the parameters before sharing them
  gov::cca::ports::ParameterGetListener paramGetListener = (*this);
  ppf.registerUpdater(tm, paramGetListener);

  // publish the parameter port and release the parameter port factory
  ppf.addParameterPort(tm, myServices);
  myServices.releasePort("ParameterPortFactory");

  return 0;
}
// DO-NOT-DELETE splicer.end(TOPS.StructuredSolver._misc)

