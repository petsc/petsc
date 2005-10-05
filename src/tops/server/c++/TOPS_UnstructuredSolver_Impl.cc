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
#include <iostream>
#include "TOPS_ParameterHandling.hh" // not from SIDL

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
  TOPS::System::Compute::Matrix system;
  TOPS::Unstructured::Matrix matrix1 = TOPS::Unstructured::Matrix::_create();
  TOPS::Unstructured::Matrix matrix2 = TOPS::Unstructured::Matrix::_create();

#define GetImpl(A,b) (!(A)b) ? 0 : reinterpret_cast<A ## _impl*>(((A) b)._get_ior()->d_data)

  // currently no support for dof > 1
  TOPS::Unstructured::Matrix_impl *imatrix1 = GetImpl(TOPS::Unstructured::Matrix,matrix1);
  TOPS::Unstructured::Matrix_impl *imatrix2 = GetImpl(TOPS::Unstructured::Matrix,matrix2);
  imatrix1->mat = J;
  imatrix2->mat = B;

#ifdef HAVE_CCA
  system = solver->getServices().getPort("TOPS.System.Compute.Matrix");
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.System.Compute.Matrix port is nil, " 
	      << "possibly not connected." << std::endl;
    PetscFunctionReturn(1);
  }
#else
  system = (TOPS::System::Compute::Matrix) solver->getSystem();
#endif

  // Use the port
  CHKMEMQ;
  system.computeMatrix(matrix1,matrix2);
  CHKMEMQ;

#ifdef HAVE_CCA
  solver->getServices().releasePort("TOPS.System.Compute.Matrix");
#endif

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
  TOPS::System::Compute::RightHandSide system;
  double *uu;
  Vec local;
  VecGhostGetLocalForm(f,&local);
  VecGetArray(local,&uu);
  int nlocal;
  VecGetLocalSize(local,&nlocal);
  sidl::array<double> ua;
  int lower[4],upper[4],stride[4];
  lower[0] = 0; upper[0] = nlocal-1; stride[0] = 1;
  ua.borrow(uu,1,*&lower,*&upper,*&stride);

#ifdef HAVE_CCA
  system = solver->getServices().getPort("TOPS.System.Compute.RightHandSide");  
  if (system._is_nil()) {
    std::cerr << "Error at " << __FILE__ << ":" << __LINE__ 
	      << ": TOPS.System.Compute.RightHandSide port is nil, " 
	      << "possibly not connected." << std::endl;
    PetscFunctionReturn(1);
  }
#else
  system = (TOPS::System::Compute::RightHandSide) solver->getSystem();
#endif
  CHKMEMQ;
  // Use the port
  system.computeRightHandSide(ua);
  CHKMEMQ;

#ifdef HAVE_CCA
  solver->getServices().releasePort("TOPS.System.Compute.RightHandSide");
#endif

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
 * Method:  getServices[]
 */
::gov::cca::Services
TOPS::UnstructuredSolver_impl::getServices ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getServices)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getServices} (getServices method)
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::getServices()"

  return this->myServices;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getServices)
}

/**
 * Method:  setSystem[]
 */
void
TOPS::UnstructuredSolver_impl::setSystem (
  /* in */ ::TOPS::System::System sys ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setSystem)
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setSystem"

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::getSystem"

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::Initialize"

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
  PetscInitialize(&argc,&argv,0,0);
  SlicedCreate(PETSC_COMM_WORLD,&this->slice);

  // Process runtime parameters
  params = myServices.getPort("tops_options");
  std::string options = params.readConfigurationMap().getString("options","-help");
  processTOPSOptions(options);

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::solve"

  PetscErrorCode ierr;

  if (!this->dmmg) {
    TOPS::System::Initialize::Once once;

#ifdef HAVE_CCA
    once = myServices.getPort("TOPS.System.Initialize.Once");
#else
    once = (TOPS::System::Initialize::Once)this->system;
#endif
    if (once._not_nil()) {    
      once.initializeOnce();
    }
#ifdef HAVE_CCA
    myServices.releasePort("TOPS.System.Initialize.Once");
#endif

    // create DMMG object 
    DMMGCreate(PETSC_COMM_WORLD,1,(void*)&this->self,&this->dmmg);
    DMMGSetDM(this->dmmg,(DM)this->slice);
    TOPS::System::Compute::Residual residual;

#ifdef HAVE_CCA
    residual = myServices.getPort("TOPS.System.Compute.Residual");
#else
    residual = (TOPS::System::Compute::Residual) this->system;
#endif
    if (residual._not_nil()) {
      ierr = DMMGSetSNES(this->dmmg, FormFunction, 0);
    } else {
      ierr = DMMGSetKSP(this->dmmg,FormRightHandSide,FormMatrix);
    }
#ifdef HAVE_CCA
    myServices.releasePort("TOPS.System.Compute.Residual");
#endif

    TOPS::System::Compute::InitialGuess guess;

#ifdef HAVE_CCA
    guess = myServices.getPort("TOPS.System.Compute.InitialGuess");
#else
    guess = (TOPS::System::Compute::InitialGuess) this->system;
#endif
    if (guess._not_nil()) {
      ierr = DMMGSetInitialGuess(this->dmmg, FormInitialGuess);
    }
  }
#ifdef HAVE_CCA
  myServices.releasePort("TOPS.System.Compute.InitialGuess");
#endif

  TOPS::System::Initialize::EverySolve every;

#ifdef HAVE_CCA
  every = myServices.getPort("TOPS.System.Initialize.EverySolve");
#else
  every = (TOPS::System::Initialize::EverySolve)this->system;
#endif
  if (every._not_nil()) {    
    every.initializeEverySolve();
  }
#ifdef HAVE_CCA
    myServices.releasePort("TOPS.System.Initialize.EverySolve");
#endif

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setBlockSize"

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
  return 0;
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
 * Method:  setValue[]
 */
void
TOPS::UnstructuredSolver_impl::setValue (
  /* in */ const ::std::string& key,
  /* in */ const ::std::string& value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setValue)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setValue} (setValue method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setValue)
}

/**
 * Method:  setValue[Int]
 */
void
TOPS::UnstructuredSolver_impl::setValue (
  /* in */ const ::std::string& key,
  /* in */ int32_t value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setValueInt)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setValueInt} (setValue method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setValueInt)
}

/**
 * Method:  setValue[Bool]
 */
void
TOPS::UnstructuredSolver_impl::setValue (
  /* in */ const ::std::string& key,
  /* in */ bool value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setValueBool)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setValueBool} (setValue method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setValueBool)
}

/**
 * Method:  setValue[Double]
 */
void
TOPS::UnstructuredSolver_impl::setValue (
  /* in */ const ::std::string& key,
  /* in */ double value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.setValueDouble)
  // Insert-Code-Here {TOPS.UnstructuredSolver.setValueDouble} (setValue method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setValueDouble)
}

/**
 * Method:  getValue[]
 */
::std::string
TOPS::UnstructuredSolver_impl::getValue (
  /* in */ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getValue)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getValue} (getValue method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getValue)
}

/**
 * Method:  getValueInt[]
 */
int32_t
TOPS::UnstructuredSolver_impl::getValueInt (
  /* in */ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getValueInt)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getValueInt} (getValueInt method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getValueInt)
}

/**
 * Method:  getValueBool[]
 */
bool
TOPS::UnstructuredSolver_impl::getValueBool (
  /* in */ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getValueBool)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getValueBool} (getValueBool method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getValueBool)
}

/**
 * Method:  getValueDouble[]
 */
double
TOPS::UnstructuredSolver_impl::getValueDouble (
  /* in */ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.getValueDouble)
  // Insert-Code-Here {TOPS.UnstructuredSolver.getValueDouble} (getValueDouble method)
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.getValueDouble)
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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::getLocalSize"

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setGhostPoints"

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
  return 0;
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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setPreallocation"

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setPreallocation"

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
#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setServices"

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
			   "TOPS.Unstructured.Solver",
			   "TOPS.Unstructured.Solver", tm);
  
  // Uses ports
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

  services.registerUsesPort("TOPS.System.Compute.Residual",
			    "TOPS.System.Compute.Residual", tm);

  // Parameter port
  myServices.registerUsesPort("ParameterPortFactory",
			      "gov.cca.ports.ParameterPortFactory", tm);

  // Set up parameter port
  if (this->setupParameterPort() != 0) {
    std::cerr << "TOPS::UnstructuredSolver_impl::go: errors during setup of ParameterPort" << std::endl;
  }

  return;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.setServices)
}

/**
 * Inform the listener that someone is about to fetch their 
 * typemap. The return should be true if the listener
 * has changed the ParameterPort definitions.
 */
bool
TOPS::UnstructuredSolver_impl::updateParameterPort (
  /* in */ const ::std::string& portName ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.updateParameterPort)
  // Insert-Code-Here {TOPS.UnstructuredSolver.updateParameterPort} (updateParameterPort method)
  std::cout << "TOPS::UnstructuredSolver_impl::updatedParameterPort called" << std::endl;
  // Get the runtime parameters
  params = myServices.getPort("tops_options");
  std::string options = params.readConfigurationMap().getString("options","-help");
  processTOPSOptions(options);
  return true;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.updateParameterPort)
}

/**
 * The component wishing to be told after a parameter is changed
 * implements this function.
 * @param portName the name of the port (typemap) on which the
 * value was set.
 * @param fieldName the name of the value in the typemap.
 */
void
TOPS::UnstructuredSolver_impl::updatedParameterValue (
  /* in */ const ::std::string& portName,
  /* in */ const ::std::string& fieldName ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver.updatedParameterValue)
  // Insert-Code-Here {TOPS.UnstructuredSolver.updatedParameterValue} (updatedParameterValue method)
  std::cout << "TOPS::UnstructuredSolver_impl::updatedParameterValue called" << std::endl;
  // DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver.updatedParameterValue)
}


// DO-NOT-DELETE splicer.begin(TOPS.UnstructuredSolver._misc)
// Insert-Code-Here {TOPS.UnstructuredSolver._misc} (miscellaneous code)
int TOPS::UnstructuredSolver_impl::setupParameterPort() {

#undef __FUNCT__
#define __FUNCT__ "TOPS::UnstructuredSolver_impl::setupParameterPort"

  // First, get parameters
  ppf = myServices.getPort("ParameterPortFactory");
  if (ppf._is_nil()) {
    std::cerr << "TOPS::UnstructuredSolver_impl::setupParameterPort: called without ParameterPortFactory connected." << std::endl;
    return -1;
  }
  gov::cca::TypeMap tm = myServices.createTypeMap();
  if (tm._is_nil()) {
    std::cerr << "TOPS::UnstructuredSolver_impl::setupParameterPort: myServices.createTypeMap failed." << std::endl;
    return -1;
  }

  ppf.initParameterData(tm, "tops_options");
  ppf.setBatchTitle(tm, "TOPS Options");
  ppf.addRequestString(tm, "options", "Space-separated list of TOPS options", 
			  "Enter runtime TOPS options", "-help");

  // We may want to respond to changes
  gov::cca::ports::ParameterSetListener paramSetListener = self;
  ppf.registerUpdatedListener(tm, paramSetListener);

  // We may want to change the parameters before sharing them
  gov::cca::ports::ParameterGetListener paramGetListener = self;
  ppf.registerUpdater(tm, paramGetListener);

  // publish the parameter port and release the parameter port factory
  ppf.addParameterPort(tm, myServices);
  myServices.releasePort("ParameterPortFactory");

  return 0;
}
// DO-NOT-DELETE splicer.end(TOPS.UnstructuredSolver._misc)

