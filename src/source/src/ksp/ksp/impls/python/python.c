/* -------------------------------------------------------------------------- */

#include "src/inline/python.h"
#include "include/private/kspimpl.h"

/* -------------------------------------------------------------------------- */

#define KSPPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPCreatePython(MPI_Comm,const char*,const char*,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetContext(KSP,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonGetContext(KSP,void**);
PETSC_EXTERN_CXX_END

/* -------------------------------------------------------------------------- */

typedef struct {
  PyObject *self;
  char     *module;
  char     *factory;
} KSP_Py;

/* -------------------------------------------------------------------------- */

#define KSP_Py_Self(ksp) (((KSP_Py*)(ksp)->data)->self)

#define KSP_PYTHON_CALL_HEAD(ksp, PyMethod) \
  PETSC_PYTHON_CALL_HEAD(KSP_Py_Self(ksp), PyMethod)
#define KSP_PYTHON_CALL_JUMP(ksp, LABEL) \
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define KSP_PYTHON_CALL_BODY(ksp, ARGS)	\
  PETSC_PYTHON_CALL_BODY(ARGS)
#define KSP_PYTHON_CALL_TAIL(ksp, PyMethod) \
  PETSC_PYTHON_CALL_TAIL()

#define KSP_PYTHON_CALL(ksp, PyMethod, ARGS) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);	     \
  KSP_PYTHON_CALL_BODY(ksp, ARGS);	     \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)	     \
/**/  

#define KSP_PYTHON_CALL_NOARGS(ksp, PyMethod) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);	      \
  KSP_PYTHON_CALL_BODY(ksp, ("", NULL));      \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)	      \
/**/

#define KSP_PYTHON_CALL_KSPARG(ksp, PyMethod)		  \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);			  \
  KSP_PYTHON_CALL_BODY(ksp, ("O&", PyPetscKSP_New, ksp)); \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)			  \
/**/

#define KSP_PYTHON_CALL_MAYBE(ksp, PyMethod, ARGS, LABEL) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);			  \
  KSP_PYTHON_CALL_JUMP(ksp, LABEL);			  \
  KSP_PYTHON_CALL_BODY(ksp, ARGS);			  \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)			  \
/**/

#define KSP_PYTHON_SETERRSUP(ksp, PyMethod)			  \
  SETERRQ1(PETSC_ERR_SUP,"method %s() not implemented",PyMethod); \
  PetscFunctionReturn(PETSC_ERR_SUP)				  \
/**/

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__  
#define __FUNCT__ "KSPSetUp_Python"
static PetscErrorCode KSPSetFromOptions_Python(KSP ksp)
{
  char           *modcls[2] = {0, 0};
  PetscInt       nmax = 2;
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Python options");CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-ksp_python","Python module and class/factory",
				 "KSPCreatePython", modcls,&nmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg) {
    if (nmax == 2) {
      PyObject *self = NULL;
      ierr = PetscCreatePythonObject(modcls[0],modcls[1],&self);CHKERRQ(ierr);
      ierr = KSPPythonSetContext(ksp,self);Py_DecRef(self);CHKERRQ(ierr);
    }
    ierr = PetscStrfree(modcls[0]);CHKERRQ(ierr);
    ierr = PetscStrfree(modcls[1]);CHKERRQ(ierr);
  }
  KSP_PYTHON_CALL_KSPARG(ksp, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__  
#define __FUNCT__ "KSPSetUp_Python"
static PetscErrorCode KSPSetUp_Python(KSP ksp)
{
  PetscFunctionBegin;
  KSP_PYTHON_CALL_KSPARG(ksp, "setUp");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__  
#define __FUNCT__ "KSPSolve_Python"
static PetscErrorCode KSPSolve_Python(KSP ksp)
{
  PetscFunctionBegin;
  ksp->its    = 0;
  ksp->rnorm  = 0;
  ksp->rnorm0 = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  if (!ksp->transpose_solve) {
    KSP_PYTHON_CALL_MAYBE(ksp, "solve",
			 ("O&", PyPetscKSP_New, ksp),
			  notimplemented1);
  } else {
    KSP_PYTHON_CALL_MAYBE(ksp, "solveTranspose",
			  ("O&", PyPetscKSP_New, ksp),
			  notimplemented2);
  }
  if (!ksp->reason) ksp->reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
 notimplemented1:
  PC_PYTHON_SETERRSUP(ksp, "solve");
 notimplemented2:
  PC_PYTHON_SETERRSUP(ksp, "solveTranspose");
}

#undef  __FUNCT__  
#define __FUNCT__ "KSPBuildSolution_Python"
static PetscErrorCode KSPBuildSolution_Python(KSP ksp, Vec v, Vec *V)
{
  PetscErrorCode ierr;
  ierr = KSPDefaultBuildSolution(ksp, v, V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__  
#define __FUNCT__ "KSPBuildResidual_Python"
static PetscErrorCode KSPBuildResidual_Python(KSP ksp, Vec t, Vec v, Vec *V)
{
  PetscErrorCode ierr;
  ierr = KSPDefaultBuildResidual(ksp, t, v, V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPView_Python"
static PetscErrorCode KSPView_Python(KSP ksp,PetscViewer viewer)
{
  KSP_Py         *py = (KSP_Py*)ksp->data;
  PetscTruth     isascii,isstring;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (isascii || isstring) {
    ierr = PetscStrfree(py->module);CHKERRQ(ierr); 
    ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
    ierr = PetscPythonGetModuleAndClass(py->self,&py->module,&py->factory);CHKERRQ(ierr);
  }
  if (isascii) {
    const char* module  = py->module  ? py->module  : "no yet set";
    const char* factory = py->factory ? py->factory : "no yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  module:  %s\n",module);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  factory: %s\n",factory);CHKERRQ(ierr);
  }
  if (isstring) {
    const char* module  = py->module  ? py->module  : "<module>";
    const char* factory = py->factory ? py->factory : "<factory>";
    ierr = PetscViewerStringSPrintf(viewer,"%s.%s",module,factory);CHKERRQ(ierr);
  }
  KSP_PYTHON_CALL(ksp, "view", ("O&O&",
				PyPetscKSP_New,     ksp,
				PyPetscViewer_New,  viewer));

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Python"
static PetscErrorCode KSPDestroy_Python(KSP ksp)
{
  KSP_Py         *py   = (KSP_Py *)ksp->data;
  PyObject       *self = py->self;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (Py_IsInitialized()) {
    KSP_PYTHON_CALL_NOARGS(ksp, "destroy");
    py->self = NULL; Py_DecRef(self);
  }
  ierr = PetscStrfree(py->module);CHKERRQ(ierr);
  ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ksp->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
   KSPPYTHON - .

   Level: intermediate

   Contributed by Lisandro Dalcin <dalcinl at gmail dot com>

.seealso:  KSP, KSPCreate(), KSPSetType(), KSPType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_Python"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_Python(KSP ksp)
{
  KSP_Py      *py;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  ierr = PetscInitializePython();CHKERRQ(ierr);

  ierr = PetscNew(KSP_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSP_Py));CHKERRQ(ierr);
  ksp->data  = (void*)py;

  /* Python */
  py->self    = NULL;
  py->module  = NULL;
  py->factory = NULL;

  /* PETSc */
  ksp->ops->destroy              = KSPDestroy_Python;
  ksp->ops->view                 = KSPView_Python;
  ksp->ops->setfromoptions       = KSPSetFromOptions_Python;
  ksp->ops->setup                = KSPSetUp_Python;
  ksp->ops->solve                = KSPSolve_Python;
  ksp->ops->buildsolution        = KSPBuildSolution_Python;
  ksp->ops->buildresidual        = KSPBuildResidual_Python;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "KSPPythonGetContext"
/*@
   KSPPythonGetContext - .

   Input Parameter:
.  ksp - KSP context

   Output Parameter:
.  ctx - Python context

   Level: beginner

.keywords: KSP, preconditioner, create

.seealso: KSP, KSPCreate(), KSPSetType(), KSPPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonGetContext(KSP ksp,void **ctx)
{
  KSP_Py        *py;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(ctx,2);
  *ctx = NULL;
  ierr = PetscTypeCompare((PetscObject)ksp,KSPPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (KSP_Py *) ksp->data;
  *ctx = (void *) py->self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPythonSetContext"
/*@
   KSPPythonSetContext - .

   Collective on KSP

   Input Parameters:
.  ksp - KSP context
.  ctx - Python context

   Level: beginner

.keywords: KSP, create

.seealso: KSP, KSPCreate(), KSPSetType(), KSPPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetContext(KSP ksp,void *ctx)
{
  KSP_Py        *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)ksp,KSPPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (KSP_Py *) ksp->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the KSP object */
  KSP_PYTHON_CALL_NOARGS(ksp, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the KSP object  */
  py->self = (PyObject *) self; Py_IncRef(py->self);
  KSP_PYTHON_CALL_KSPARG(ksp, "create");
  if (ksp->setupcalled) ksp->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCreatePython"
/*@
   KSPCreatePython - Creates a Python linear solver context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator 
.  modname - module name
.  clsname - factory/class name

   Output Parameter:
.  ksp - location to put the linear solver context

   Level: beginner

.keywords: KSP,  create

.seealso: KSP, KSPCreate(), KSPSetType(), KSPPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreatePython(MPI_Comm comm,
						  const char *modname,
						  const char *clsname,
						  KSP *ksp)
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (modname) PetscValidCharPointer(modname,2);
  if (clsname) PetscValidCharPointer(clsname,3);
  /* create the KSP context and set its type */
  ierr = KSPCreate(comm,ksp);CHKERRQ(ierr);
  ierr = KSPSetType(*ksp,KSPPYTHON);CHKERRQ(ierr);
  if (modname == PETSC_NULL) PetscFunctionReturn(0);
  if (clsname == PETSC_NULL) PetscFunctionReturn(0);
  /* create the Python object from module and class/factory  */
  ierr = PetscCreatePythonObject(modname,clsname,&self);CHKERRQ(ierr);
  /* set the created Python object in KSP context */
  ierr = KSPPythonSetContext(*ksp,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
