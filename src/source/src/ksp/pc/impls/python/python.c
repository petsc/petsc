#define PETSCPC_DLL

/* -------------------------------------------------------------------------- */

#include "src/inline/python.h"
#include "include/private/pcimpl.h"                      /*I   "petscpc.h"   I*/

/* -------------------------------------------------------------------------- */

/* backward compatibility hacks */

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 2 &&	\
     PETSC_VERSION_RELEASE  == 1)
#endif

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 3 &&	\
     PETSC_VERSION_RELEASE  == 1)
#endif

/* -------------------------------------------------------------------------- */

#define PCPYTHON "python"

PETSC_EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreatePython(MPI_Comm,const char*,const char*,PC*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetContext(PC,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonGetContext(PC,void**);
PETSC_EXTERN_C_END

/* -------------------------------------------------------------------------- */

typedef struct {
  /**/
  PyObject   *self;
  char       *module;
  char       *factory;
} PC_Py;

/* -------------------------------------------------------------------------- */

#define PC_Py_Self(pc) (((PC_Py*)(pc)->data)->self)

#define PC_PYTHON_CALL_HEAD(pc, PyMethod)		\
  PETSC_PYTHON_CALL_HEAD(PC_Py_Self(pc), PyMethod)
#define PC_PYTHON_CALL_JUMP(pc, LABEL)			\
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define PC_PYTHON_CALL_BODY(pc, ARGS)			\
  PETSC_PYTHON_CALL_BODY(ARGS)
#define PC_PYTHON_CALL_TAIL(pc, PyMethod)		\
  PETSC_PYTHON_CALL_TAIL()

#define PC_PYTHON_CALL(pc, PyMethod, ARGS)		\
  PC_PYTHON_CALL_HEAD(pc, PyMethod);			\
  PC_PYTHON_CALL_BODY(pc, ARGS);			\
  PC_PYTHON_CALL_TAIL(pc, PyMethod)			\
/**/  

#define PC_PYTHON_CALL_NOARGS(pc, PyMethod)		\
  PC_PYTHON_CALL_HEAD(pc, PyMethod);			\
  PC_PYTHON_CALL_BODY(pc, ("", NULL));			\
  PC_PYTHON_CALL_TAIL(pc, PyMethod)			\
/**/

#define PC_PYTHON_CALL_PCARG(pc, PyMethod)		\
  PC_PYTHON_CALL_HEAD(pc, PyMethod);			\
  PC_PYTHON_CALL_BODY(pc, ("O&",PyPetscPC_New,pc));	\
  PC_PYTHON_CALL_TAIL(pc, PyMethod)			\
/**/

#define PC_PYTHON_CALL_MAYBE(pc, PyMethod, ARGS, LABEL)	\
  PC_PYTHON_CALL_HEAD(pc, PyMethod);				\
  PC_PYTHON_CALL_JUMP(pc, LABEL);				\
  PC_PYTHON_CALL_BODY(pc, ARGS);				\
  PC_PYTHON_CALL_TAIL(pc, PyMethod)				\
/**/

#define PC_PYTHON_SETERRSUP(pc, PyMethod)				\
  SETERRQ1(PETSC_ERR_SUP,"method %s() not implemented",PyMethod);	\
  PetscFunctionReturn(PETSC_ERR_SUP)					\
/**/

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Python"
static PetscErrorCode PCDestroy_Python(PC pc)
{
  PC_Py          *py   = (PC_Py *)pc->data;
  PyObject       *self = py->self;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Py_IsInitialized()) {
    PC_PYTHON_CALL_NOARGS(pc, "destroy");
    py->self = NULL; Py_DecRef(self);
  }
  ierr = PetscStrfree(py->module);CHKERRQ(ierr);
  ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  pc->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Python"
static PetscErrorCode PCSetFromOptions_Python(PC pc)
{
  char           *modcls[2] = {0, 0};
  PetscInt       nmax = 2;
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Python options");CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-pc_python","Python module and class/factory",
				 "PCCreatePython",modcls,&nmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg) {
    if (nmax == 2) {
      PyObject *self = NULL;
      ierr = PetscCreatePythonObject(modcls[0],modcls[1],&self);CHKERRQ(ierr);
      ierr = PCPythonSetContext(pc,self);Py_DecRef(self);CHKERRQ(ierr);
    }
    ierr = PetscStrfree(modcls[0]);CHKERRQ(ierr);
    ierr = PetscStrfree(modcls[1]);CHKERRQ(ierr);
  }
  PC_PYTHON_CALL_PCARG(pc, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Python"
static PetscErrorCode PCView_Python(PC pc,PetscViewer viewer)
{
  PC_Py          *py = (PC_Py *)pc->data;
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
  PC_PYTHON_CALL(pc, "view", ("O&O&",
			      PyPetscPC_New,     pc,
			      PyPetscViewer_New, viewer));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PCPreSolve_Python"
static PetscErrorCode PCPreSolve_Python(PC pc, KSP ksp, Vec b, Vec x)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL(pc, "preSolve", ("O&O&O&O&",
				  PyPetscPC_New,  pc,
				  PyPetscKSP_New, ksp,
				  PyPetscVec_New, b,
				  PyPetscVec_New, x   ));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCPostSolve_Python"
static PetscErrorCode PCPostSolve_Python(PC pc, KSP ksp, Vec b, Vec x)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL(pc, "postSolve", ("O&O&O&O&",
				   PyPetscPC_New,  pc,
				   PyPetscKSP_New, ksp,
				   PyPetscVec_New, b,
				   PyPetscVec_New, x   ));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Python"
static PetscErrorCode PCApply_Python(PC pc,Vec x, Vec y)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL_MAYBE(pc, "apply", 
		       ("O&O&O&",
			PyPetscPC_New,  pc,
			PyPetscVec_New, x,
			PyPetscVec_New, y  ),
		       notimplemented);
  PetscFunctionReturn(0);
 notimplemented:
  PC_PYTHON_SETERRSUP(pc, "apply");
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft_Python"
static PetscErrorCode PCApplySymmetricLeft_Python(PC pc,Vec x, Vec y)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL_MAYBE(pc, "applySymmetricLeft", 
		       ("O&O&O&",
			PyPetscPC_New,  pc,
			PyPetscVec_New, x,
			PyPetscVec_New, y  ),
		       notimplemented);
  PetscFunctionReturn(0);
 notimplemented:
  PC_PYTHON_SETERRSUP(pc, "applySymmetricLeft");
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight_Python"
static PetscErrorCode PCApplySymmetricRight_Python(PC pc,Vec x, Vec y)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL_MAYBE(pc, "applySymmetricRight", 
		       ("O&O&O&",
			PyPetscPC_New,  pc,
			PyPetscVec_New, x,
			PyPetscVec_New, y  ),
		       notimplemented);
  PetscFunctionReturn(0);
 notimplemented:
  PC_PYTHON_SETERRSUP(pc, "applySymmetricRight");
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Python"
static PetscErrorCode PCApplyTranspose_Python(PC pc,Vec x, Vec y)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL_MAYBE(pc, "applyTranspose",
		       ("O&O&O&",
			PyPetscPC_New,  pc,
			PyPetscVec_New, x,
			PyPetscVec_New, y  ),
		       notimplemented);
  PetscFunctionReturn(0);
 notimplemented:
  PC_PYTHON_SETERRSUP(pc, "applyTranspose");
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_Python"
static PetscErrorCode PCApplyRichardson_Python(PC pc,Vec x, Vec y, Vec w,
					       PetscReal rtol,PetscReal atol,PetscReal dtol,PetscInt its)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL_MAYBE(pc, "applyRichardson",
		       ("O&O&O&O&(dddl)",
			PyPetscPC_New,  pc,
			PyPetscVec_New, x,
			PyPetscVec_New, y,
			PyPetscVec_New, w,
			(double)rtol,(double)atol,(double)dtol,(long)its),
		       notimplemented);
  PetscFunctionReturn(0);
 notimplemented:
  PC_PYTHON_SETERRSUP(pc, "applyRichardson");
}
#endif

static int PCPythonHasOperation(PC pc, const char operation[])
{
  PC_Py    *py = (PC_Py *)pc->data;
  PyObject *attr = NULL;
  if (py->self == NULL || py->self == Py_None) return 0;
  attr = PetscPyObjectGetAttrStr(py->self, operation);
  if      (attr == NULL)    { PyErr_Clear();   return 0; }
  else if (attr == Py_None) { Py_DecRef(attr); return 0; }
  return 1;
}

#undef __FUNCT__  
#define __FUNCT__ "PCPythonFillOperations"
static PetscErrorCode PCPythonFillOperations(PC pc)
{
  PetscFunctionBegin;

  pc->ops->applysymmetricleft =
    PCPythonHasOperation(pc, "applySymmetricLeft") ?
    PCApplySymmetricLeft_Python : PETSC_NULL;
  
  pc->ops->applysymmetricright =
    PCPythonHasOperation(pc, "applySymmetricRight") ?
    PCApplySymmetricRight_Python : PETSC_NULL;

  pc->ops->applytranspose =
    PCPythonHasOperation(pc, "applyTranspose") ?
    PCApplyTranspose_Python : PETSC_NULL;

#if 0
  pc->ops->applyrichardson =
    PCPythonHasOperation(pc, "applyRichardson") ?
    PCApplyRichardson_Python : PETSC_NULL;
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Python"
static PetscErrorCode PCSetUp_Python(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PC_PYTHON_CALL_PCARG(pc, "setUp");
  ierr = PCPythonFillOperations(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
      PCPYTHON - 

  Level: beginner

.seealso:  PC, PCCreate(), PCSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Python"
PetscErrorCode PETSCTS_DLLEXPORT PCCreate_Python(PC pc)
{
  PC_Py          *py;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscInitializePython();CHKERRQ(ierr);
  
  ierr = PetscNew(PC_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Py));CHKERRQ(ierr);
  pc->data = (void*)py;

  /* Python */
  py->self    = NULL;
  py->module  = NULL;
  py->factory = NULL;

  /* PETSc  */
  pc->ops->destroy         = PCDestroy_Python;
  pc->ops->setfromoptions  = PCSetFromOptions_Python;
  pc->ops->view            = PCView_Python;
  pc->ops->setup           = PCSetUp_Python;
  pc->ops->apply           = PCApply_Python;
  pc->ops->presolve        = PCPreSolve_Python;
  pc->ops->postsolve       = PCPostSolve_Python;
  /* To be filled later in PCSetUp_Python() ... */
  pc->ops->applysymmetricleft  = PETSC_NULL/*PCApplySymmetricLeft_Python*/;
  pc->ops->applysymmetricright = PETSC_NULL/*PCApplySymmetricRight_Python*/;
  pc->ops->applytranspose      = PETSC_NULL/*PCApplyTranspose_Python*/;
  pc->ops->applyrichardson     = PETSC_NULL/*PCApplyRichardson_Python*/;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCPythonGetContext"
/*@
   PCPythonGetContext - .

   Input Parameter:
.  pc - PC context

   Output Parameter:
.  ctx - Python context

   Level: beginner

.keywords: PC, preconditioner, create

.seealso: PC, PCCreate(), PCSetType(), PCPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCPythonGetContext(PC pc,void **ctx)
{
  PC_Py          *py;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(ctx,2);
  *ctx = NULL;
  ierr = PetscTypeCompare((PetscObject)pc,PCPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (PC_Py *) pc->data;
  *ctx = (void *) py->self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPythonSetContext"
/*@
   PCPythonSetContext - .

   Collective on PC

   Input Parameters:
.  pc - PC context
.  ctx - Python context

   Level: beginner

.keywords: PC, preconditioner, create

.seealso: PC, PCCreate(), PCSetType(), PCPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetContext(PC pc,void *ctx)
{
  PC_Py          *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)pc,PCPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (PC_Py *) pc->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the PC object */
  PC_PYTHON_CALL_NOARGS(pc, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the PC object  */
  py->self = (PyObject *) self; Py_IncRef(py->self);
  PC_PYTHON_CALL_PCARG(pc, "create");
  if (pc->setupcalled) pc->setupcalled = 1;
#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
  ierr = PCPythonFillOperations(pc);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreatePython"
/*@
   PCCreatePython - Creates a Python preconditioner context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator 
.  modname - module name
.  clsname - factory/class name

   Output Parameter:
.  pc - location to put the preconditioner context

   Level: beginner

.keywords: PC, preconditioner, create

.seealso: PC, PCCreate(), PCSetType(), PCPYTHON
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCreatePython(MPI_Comm comm,
						 const char *modname,
						 const char *clsname,
						 PC *pc)
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (modname) PetscValidCharPointer(modname,2);
  if (clsname) PetscValidCharPointer(clsname,3);
  /* create the PC context and set its type */
  ierr = PCCreate(comm,pc);CHKERRQ(ierr);
  ierr = PCSetType(*pc,PCPYTHON);CHKERRQ(ierr);
  if (modname == PETSC_NULL) PetscFunctionReturn(0);
  if (clsname == PETSC_NULL) PetscFunctionReturn(0);
  /* create the Python object from module and class/factory  */
  ierr = PetscCreatePythonObject(modname,clsname,&self);CHKERRQ(ierr);
  /* set the created Python object in PC context */
  ierr = PCPythonSetContext(*pc,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
