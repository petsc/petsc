/* -------------------------------------------------------------------------- */

#include "python_core.h"
#include "private/kspimpl.h"

/* -------------------------------------------------------------------------- */

#define KSPPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetContext(KSP,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonGetContext(KSP,void**);
PETSC_EXTERN_CXX_END

/* -------------------------------------------------------------------------- */

typedef struct {
  PyObject *self;
  char     *pyname;
} KSP_Py;

/* -------------------------------------------------------------------------- */

#define KSP_Py_Self(ksp) (((KSP_Py*)(ksp)->data)->self)

#define KSP_PYTHON_CALL_HEAD(ksp, PyMethod) \
  PETSC_PYTHON_CALL_HEAD(KSP_Py_Self(ksp), PyMethod)
#define KSP_PYTHON_CALL_JUMP(ksp, LABEL) \
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define KSP_PYTHON_CALL_BODY(ksp, ARGS) \
  PETSC_PYTHON_CALL_BODY(ARGS)
#define KSP_PYTHON_CALL_TAIL(ksp, PyMethod) \
  PETSC_PYTHON_CALL_TAIL()

#define KSP_PYTHON_CALL(ksp, PyMethod, ARGS) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);       \
  KSP_PYTHON_CALL_BODY(ksp, ARGS);           \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)        \
/**/

#define KSP_PYTHON_CALL_NOARGS(ksp, PyMethod) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);        \
  KSP_PYTHON_CALL_BODY(ksp, ("", NULL));      \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)         \
/**/

#define KSP_PYTHON_CALL_KSPARG(ksp, PyMethod)             \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);                    \
  KSP_PYTHON_CALL_BODY(ksp, ("O&", PyPetscKSP_New, ksp)); \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)                     \
/**/

#define KSP_PYTHON_CALL_MAYBE(ksp, PyMethod, ARGS, LABEL) \
  KSP_PYTHON_CALL_HEAD(ksp, PyMethod);                    \
  KSP_PYTHON_CALL_JUMP(ksp, LABEL);                       \
  KSP_PYTHON_CALL_BODY(ksp, ARGS);                        \
  KSP_PYTHON_CALL_TAIL(ksp, PyMethod)                     \
/**/

#define KSP_PYTHON_SETERRSUP(ksp, PyMethod)   \
  PETSC_PYTHON_NOTIMPLEMENTED(ksp, PyMethod); \
  PetscFunctionReturn(PETSC_ERR_SUP)          \
/**/

/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPPythonSetType_PYTHON"
PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType_PYTHON(KSP ksp,const char pyname[])
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* create the Python object from module/class/function  */
  ierr = PetscCreatePythonObject(pyname,&self);CHKERRQ(ierr);
  /* set the created Python object in KSP context */
  ierr = KSPPythonSetContext(ksp,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ksp->data = PETSC_NULL;
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPPythonSetType_C",
                                    "",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_Python"
static PetscErrorCode KSPSetFromOptions_Python(KSP ksp)
{
  KSP_Py         *py = (KSP_Py *)ksp->data;
  char           pyname[2*PETSC_MAX_PATH_LEN+3];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Python options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-ksp_python_type","Python package.module[.{class|function}]",
                            "KSPPythonSetType",py->pyname,pyname,sizeof(pyname),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg && pyname[0]) {
    ierr = PetscStrcmp(py->pyname,pyname,&flg);CHKERRQ(ierr);
    if (!flg) { ierr = KSPPythonSetType_PYTHON(ksp,pyname);CHKERRQ(ierr); }
  }
  KSP_PYTHON_CALL_KSPARG(ksp, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_Python"
static PetscErrorCode KSPView_Python(KSP ksp,PetscViewer viewer)
{
  KSP_Py         *py = (KSP_Py*)ksp->data;
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
  KSP_PYTHON_CALL(ksp, "view", ("O&O&",
                                PyPetscKSP_New,     ksp,
                                PyPetscViewer_New,  viewer));

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "KSPSetUp_Python"
static PetscErrorCode KSPSetUp_Python(KSP ksp)
{
  KSP_Py *py = (KSP_Py*)ksp->data;
  PetscFunctionBegin;
  if (!py->self) {
    SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
             "Python context not set, call one of \n"
             " * KSPPythonSetType(ksp,\"[package.]module.class\")\n"
             " * KSPSetFromOptions(ksp) and pass option -ksp_python_type [package.]module.class");
  }
  KSP_PYTHON_CALL_KSPARG(ksp, "setUp");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "KSPSolve_Python"
static PetscErrorCode KSPSolve_Python(KSP ksp)
{
  const char *solveMeth = 0;
  PetscFunctionBegin;
  if (!ksp->transpose_solve)
    solveMeth = "solve";
  else
    solveMeth = "solveTranspose";
  ksp->its    = 0;
  ksp->rnorm  = 0;
  ksp->rnorm0 = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  KSP_PYTHON_CALL_MAYBE(ksp, solveMeth, ("O&O&O&",
                                         PyPetscKSP_New, ksp,
                                         PyPetscVec_New, ksp->vec_rhs,
                                         PyPetscVec_New, ksp->vec_sol),
                        notimplemented);
  if (!ksp->reason) ksp->reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
 notimplemented:
  KSP_PYTHON_SETERRSUP(ksp, solveMeth);
}

#undef  __FUNCT__
#define __FUNCT__ "KSPBuildSolution_Python"
static PetscErrorCode KSPBuildSolution_Python(KSP ksp, Vec v, Vec *V)
{
  const char     *key = "__petsc4py_KSP_python_work_vec_sol";
  Vec            x    = v;
  PetscErrorCode ierr;
  if (!x) {
    ierr = PetscObjectQuery((PetscObject)ksp,key,(PetscObject*)&x);CHKERRQ(ierr);
    if (!x) {
      ierr = VecDuplicate(ksp->vec_sol,&x);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ksp,key,(PetscObject)x);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)x);CHKERRQ(ierr);
    }
  }
  KSP_PYTHON_CALL_MAYBE(ksp, "buildSolution", ("O&O&",
                                               PyPetscKSP_New, ksp,
                                               PyPetscVec_New, x),
                        notimplemented);
  if (V) { *V = x; }
  PetscFunctionReturn(0);
 notimplemented:
  ierr = KSPDefaultBuildSolution(ksp, v, V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "KSPBuildResidual_Python"
static PetscErrorCode KSPBuildResidual_Python(KSP ksp, Vec t, Vec v, Vec *V)
{
  PetscErrorCode ierr;
  KSP_PYTHON_CALL_MAYBE(ksp, "buildResidual", ("O&O&O&",
                                               PyPetscKSP_New, ksp,
                                               PyPetscVec_New, t,
                                               PyPetscVec_New, v),
                        notimplemented);
  if (V) { *V = v; }
  PetscFunctionReturn(0);
 notimplemented:
  ierr = KSPDefaultBuildResidual(ksp, t, v, V);CHKERRQ(ierr);
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
  KSP_Py         *py;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscPythonImportPetsc4Py();CHKERRQ(ierr);

  ierr = PetscNew(KSP_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSP_Py));CHKERRQ(ierr);
  ksp->data  = (void*)py;

  /* Python */
  py->self    = NULL;
  py->pyname  = NULL;

  /* PETSc */
  ksp->ops->destroy              = KSPDestroy_Python;
  ksp->ops->view                 = KSPView_Python;
  ksp->ops->setfromoptions       = KSPSetFromOptions_Python;
  ksp->ops->setup                = KSPSetUp_Python;
  ksp->ops->solve                = KSPSolve_Python;
  ksp->ops->buildsolution        = KSPBuildSolution_Python;
  ksp->ops->buildresidual        = KSPBuildResidual_Python;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,
                                    "KSPPythonSetType_C","KSPPythonSetType_PYTHON",
                                    (PetscVoidFunction)KSPPythonSetType_PYTHON);CHKERRQ(ierr);

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
  KSP_Py         *py;
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
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
  KSP_Py         *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
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
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = PetscPythonGetFullName(py->self,&py->pyname);CHKERRQ(ierr);
  KSP_PYTHON_CALL_KSPARG(ksp, "create");

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
  ksp->setupcalled = 0;
#else
  ksp->setupstage = KSP_SETUP_NEW;
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#if 0

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType(KSP,const char[]);
PETSC_EXTERN_CXX_END

#undef __FUNCT__
#define __FUNCT__ "KSPPythonSetType"
/*@C
   KSPPythonSetType - Initalize a KSP object implemented in Python.

   Collective on KSP

   Input Parameter:
+  ksp - the linear solver (KSP) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -ksp_python_type <pyname>

   Level: intermediate

.keywords: KSP, Python

.seealso: KSPCreate(), KSPSetType(), KSPPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType(KSP ksp,const char pyname[])
{
  PetscErrorCode (*f)(KSP, const char[]) = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPPythonSetType_C",(PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(ksp,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
