/* -------------------------------------------------------------------------- */

#include "python_core.h"
#include "private/pcimpl.h"

/* -------------------------------------------------------------------------- */

#define PCPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetContext(PC,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonGetContext(PC,void**);
PETSC_EXTERN_CXX_END

/* -------------------------------------------------------------------------- */

typedef struct {
  /**/
  PyObject   *self;
  char       *pyname;
} PC_Py;

/* -------------------------------------------------------------------------- */

#define PC_Py_Self(pc) (((PC_Py*)(pc)->data)->self)

#define PC_PYTHON_CALL_HEAD(pc, PyMethod)               \
  PETSC_PYTHON_CALL_HEAD(PC_Py_Self(pc), PyMethod)
#define PC_PYTHON_CALL_JUMP(pc, LABEL)                  \
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define PC_PYTHON_CALL_BODY(pc, ARGS)                   \
  PETSC_PYTHON_CALL_BODY(ARGS)
#define PC_PYTHON_CALL_TAIL(pc, PyMethod)               \
  PETSC_PYTHON_CALL_TAIL()

#define PC_PYTHON_CALL(pc, PyMethod, ARGS)              \
  PC_PYTHON_CALL_HEAD(pc, PyMethod);                    \
  PC_PYTHON_CALL_BODY(pc, ARGS);                        \
  PC_PYTHON_CALL_TAIL(pc, PyMethod)                     \
/**/

#define PC_PYTHON_CALL_NOARGS(pc, PyMethod)             \
  PC_PYTHON_CALL_HEAD(pc, PyMethod);                    \
  PC_PYTHON_CALL_BODY(pc, ("", NULL));                  \
  PC_PYTHON_CALL_TAIL(pc, PyMethod)                     \
/**/

#define PC_PYTHON_CALL_PCARG(pc, PyMethod)              \
  PC_PYTHON_CALL_HEAD(pc, PyMethod);                    \
  PC_PYTHON_CALL_BODY(pc, ("O&",PyPetscPC_New,pc));     \
  PC_PYTHON_CALL_TAIL(pc, PyMethod)                     \
/**/

#define PC_PYTHON_CALL_MAYBE(pc, PyMethod, ARGS, LABEL) \
  PC_PYTHON_CALL_HEAD(pc, PyMethod);                            \
  PC_PYTHON_CALL_JUMP(pc, LABEL);                               \
  PC_PYTHON_CALL_BODY(pc, ARGS);                                \
  PC_PYTHON_CALL_TAIL(pc, PyMethod)                             \
/**/

#define PC_PYTHON_SETERRSUP(pc, PyMethod)    \
  PETSC_PYTHON_NOTIMPLEMENTED(pd, PyMethod); \
  PetscFunctionReturn(PETSC_ERR_SUP)         \
/**/

/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCPythonSetType_PYTHON"
PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetType_PYTHON(PC pc,const char pyname[])
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* create the Python object from module/class/function  */
  ierr = PetscCreatePythonObject(pyname,&self);CHKERRQ(ierr);
  /* set the created Python object in PC context */
  ierr = PCPythonSetContext(pc,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  pc->data = PETSC_NULL;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCPythonSetType_C",
                                    "",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PCSetFromOptions_Python"
static PetscErrorCode PCSetFromOptions_Python(PC pc)
{
  PC_Py          *py = (PC_Py *)pc->data;
  char           pyname[2*PETSC_MAX_PATH_LEN+3];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("PC Python options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-pc_python_type","Python package.module[.{class|function}]",
                            "PCPythonSetType",py->pyname,pyname,sizeof(pyname),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg && pyname[0]) {
    ierr = PetscStrcmp(py->pyname,pyname,&flg);CHKERRQ(ierr);
    if (!flg) { ierr = PCPythonSetType_PYTHON(pc,pyname);CHKERRQ(ierr); }
  }
  PC_PYTHON_CALL_PCARG(pc, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_Python"
static PetscErrorCode PCView_Python(PC pc,PetscViewer viewer)
{
  PC_Py          *py = (PC_Py *)pc->data;
  PetscBool      isascii,isstring;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    const char* pyname  = py->pyname ? py->pyname  : "no yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  Python: %s\n",pyname);CHKERRQ(ierr);
  }
  if (isstring) {
    const char* pyname  = py->pyname ? py->pyname  : "<unknown>";
    ierr = PetscViewerStringSPrintf(viewer,"%s",pyname);CHKERRQ(ierr);
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
  else                      { Py_DecRef(attr); return 1; }
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
  PC_Py          *py = (PC_Py *)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!py->self) {
    SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
             "Python context not set, call one of \n"
             " * PCPythonSetType(pc,\"[package.]module.class\")\n"
             " * PCSetFromOptions(pc) and pass option -pc_python_type [package.]module.class");
  }
  PC_PYTHON_CALL_PCARG(pc, "setUp");
  ierr = PCPythonFillOperations(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !(PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "PCReset_Python"
static PetscErrorCode PCReset_Python(PC pc)
{
  PetscFunctionBegin;
  PC_PYTHON_CALL(pc, "reset", ("O&", PyPetscPC_New,  pc));
  PetscFunctionReturn(0);
}
#endif

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

  ierr = PetscPythonImportPetsc4Py();CHKERRQ(ierr);

  ierr = PetscNew(PC_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Py));CHKERRQ(ierr);
  pc->data = (void*)py;

  /* Python */
  py->self    = NULL;
  py->pyname  = NULL;

  /* PETSc  */
  pc->ops->destroy         = PCDestroy_Python;
  pc->ops->setfromoptions  = PCSetFromOptions_Python;
  pc->ops->view            = PCView_Python;
  pc->ops->setup           = PCSetUp_Python;
  pc->ops->apply           = PCApply_Python;
  pc->ops->presolve        = PCPreSolve_Python;
  pc->ops->postsolve       = PCPostSolve_Python;
#if !(PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
  pc->ops->reset           = PCReset_Python;
#endif
  /* To be filled later in PCSetUp_Python() ... */
  pc->ops->applysymmetricleft  = PETSC_NULL/*PCApplySymmetricLeft_Python*/;
  pc->ops->applysymmetricright = PETSC_NULL/*PCApplySymmetricRight_Python*/;
  pc->ops->applytranspose      = PETSC_NULL/*PCApplyTranspose_Python*/;
  pc->ops->applyrichardson     = PETSC_NULL/*PCApplyRichardson_Python*/;

  ierr = PetscObjectComposeFunction((PetscObject)pc,
                                    "PCPythonSetType_C","PCPythonSetType_PYTHON",
                                    (PetscVoidFunction)PCPythonSetType_PYTHON);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
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
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
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
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = PetscPythonGetFullName(py->self,&py->pyname);CHKERRQ(ierr);
  PC_PYTHON_CALL_PCARG(pc, "create");
  if (pc->setupcalled) pc->setupcalled = 1;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#if 0

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT PCPythonSetType(PC,const char[]);
PETSC_EXTERN_CXX_END

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetType(PC,const char[]);
PETSC_EXTERN_CXX_END

#undef __FUNCT__
#define __FUNCT__ "PCPythonSetType"
/*@C
   PCPythonSetType - Initalize a PC object implemented in Python.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner (PC) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -pc_python_type <pyname>

   Level: intermediate

.keywords: PC, Python

.seealso: PCCreate(), PCSetType(), PCPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetType(PC pc,const char pyname[])
{
  PetscErrorCode (*f)(PC, const char[]) = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCPythonSetType_C",(PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(pc,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
