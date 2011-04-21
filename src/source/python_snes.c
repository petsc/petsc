/*  -------------------------------------------------------------------- */

#include "python_core.h"
#include "private/snesimpl.h"

/*  -------------------------------------------------------------------- */

#define SNESPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
extern PetscErrorCode SNESPythonSetContext(SNES,void*);
extern PetscErrorCode SNESPythonGetContext(SNES,void**);
PETSC_EXTERN_CXX_END

/*  -------------------------------------------------------------------- */

typedef struct _SNESPyOps *SNESPyOps;

struct _SNESPyOps {
  PetscErrorCode (*presolve)(SNES);
  PetscErrorCode (*postsolve)(SNES);
  PetscErrorCode (*prestep)(SNES,PetscInt);
  PetscErrorCode (*poststep)(SNES,PetscInt);

  PetscErrorCode (*computefunction)(SNES,Vec,Vec);
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*);
  PetscErrorCode (*linearsolve)(SNES,Vec,Vec,PetscBool*,PetscInt*);
  PetscErrorCode (*linesearch)(SNES,Vec,Vec,Vec,PetscBool*);
};

typedef struct {
  PyObject *self;
  char     *pyname;
  /* XXX get rid of this ? */
  SNESPyOps  ops;
  struct _SNESPyOps _ops;
} SNES_Py;


/* -------------------------------------------------------------------------- */

#define SNES_Py_Self(snes) (((SNES_Py*)(snes)->data)->self)

#define SNES_PYTHON_CALL_HEAD(snes, PyMethod)           \
  PETSC_PYTHON_CALL_HEAD(SNES_Py_Self(snes), PyMethod)
#define SNES_PYTHON_CALL_JUMP(snes, LABEL)              \
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define SNES_PYTHON_CALL_BODY(snes, ARGS)               \
  PETSC_PYTHON_CALL_BODY(ARGS)
#define SNES_PYTHON_CALL_TAIL(snes, PyMethod)           \
  PETSC_PYTHON_CALL_TAIL()


#define SNES_PYTHON_CALL(snes, PyMethod, ARGS)          \
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);                \
  SNES_PYTHON_CALL_BODY(snes, ARGS);                    \
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)                 \
/**/

#define SNES_PYTHON_CALL_SNESARG(snes, PyMethod)            \
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);                    \
  SNES_PYTHON_CALL_BODY(snes, ("O&",PyPetscSNES_New,snes)); \
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)                     \
/**/

#define SNES_PYTHON_CALL_MAYBE(snes, PyMethod, ARGS, LABEL) \
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);                    \
  SNES_PYTHON_CALL_JUMP(snes, LABEL);                       \
  SNES_PYTHON_CALL_BODY(snes, ARGS);                        \
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)                     \
/**/

#define SNES_PYTHON_CALL_MAYBE_RET(snes, PyMethod, ARGS, LABEL, Obj2Val, ValP) \
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);                                \
  SNES_PYTHON_CALL_JUMP(snes, LABEL);                                   \
  SNES_PYTHON_CALL_BODY(snes, ARGS);                                    \
  _retv = Obj2Val(_retv, ValP);                                         \
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)                                 \
/**/

/* -------------------------------------------------------------------------- */

#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#define SNES_DIVERGED_LINE_SEARCH SNES_DIVERGED_LS_FAILURE
#endif

#undef __FUNCT__
#define __FUNCT__ "SNESPreSolve_Python"
static PetscErrorCode SNESPreSolve_Python(SNES snes)
{
  PetscFunctionBegin;
  SNES_PYTHON_CALL_SNESARG(snes, "preSolve");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESPostSolve_Python"
static PetscErrorCode SNESPostSolve_Python(SNES snes)
{
  PetscFunctionBegin;
  SNES_PYTHON_CALL_SNESARG(snes, "postSolve");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESPreStep_Python"
static PetscErrorCode SNESPreStep_Python(SNES snes, PetscInt its)
{
  PetscFunctionBegin;
  SNES_PYTHON_CALL(snes, "preStep", ("O&l",
                                     PyPetscSNES_New, snes,
                                     (long)           its  ));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESPostStep_Python"
static PetscErrorCode SNESPostStep_Python(SNES snes, PetscInt its)
{
  PetscFunctionBegin;
  SNES_PYTHON_CALL(snes, "postStep", ("O&l",
                                      PyPetscSNES_New, snes,
                                      (long)           its  ));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunction_Python"
static PetscErrorCode SNESComputeFunction_Python(SNES snes, Vec x, Vec F)
{
  PetscInt       nfuncs;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  nfuncs = snes->nfuncs; /* backup function call counter */
  SNES_PYTHON_CALL_MAYBE(snes, "computeFunction", ("O&O&O&",
                                                   PyPetscSNES_New, snes,
                                                   PyPetscVec_New,  x,
                                                   PyPetscVec_New,  F    ),
                        notimplemented);
  if (nfuncs == snes->nfuncs) { /* snes->ops->computefunction was not called */
    if (snes->vec_rhs) { ierr = VecAXPY(F,-1.0,snes->vec_rhs);CHKERRQ(ierr); }
    snes->nfuncs++; /* increment function call counter*/
  }
  PetscFunctionReturn(0);
notimplemented:
  ierr = SNESComputeFunction(snes,x,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PyObject * SNESPyObjToMatStructure(PyObject *value, MatStructure *outflag)
{
  long flag = DIFFERENT_NONZERO_PATTERN;
  if (value == NULL)
    return NULL;
  if (value == Py_None) {
    flag = SAME_NONZERO_PATTERN;
  } else if (value == Py_False) {
    flag = SAME_NONZERO_PATTERN;
  } else if (value == Py_True) {
    flag = DIFFERENT_NONZERO_PATTERN;
  } else if (PyInt_Check(value)) {
    flag = PyInt_AsLong(value);
    if (flag < SAME_NONZERO_PATTERN ||
        flag > SUBSET_NONZERO_PATTERN) {
      PyErr_SetString(PyExc_ValueError,
                      "Jacobian routine returned an out of range "
                      "integer value for MatStructure");
      goto fail;
    }
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Jacobian routine must return None, Boolean, "
                    "or a valid integer value for MatStructure");
    goto fail;
  }
  *outflag = (MatStructure)flag;
  return value;
 fail:
  Py_DecRef(value);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeJacobian_Python"
static PetscErrorCode SNESComputeJacobian_Python(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SNES_PYTHON_CALL_MAYBE_RET(snes, "computeJacobian", ("O&O&O&O&",
                                                       PyPetscSNES_New, snes,
                                                       PyPetscVec_New,  x,
                                                       PyPetscMat_New,  *A,
                                                       PyPetscMat_New,  *B   ),
                             notimplemented,
                             SNESPyObjToMatStructure, flg);
  PetscFunctionReturn(0);
 notimplemented:
  ierr = SNESComputeJacobian(snes,x,A,B,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PyObject * SNESPyObjToPetscBool(PyObject *value, PetscBool *outsucceed)
{
  PetscBool succeed = PETSC_FALSE;
  if (value == NULL)
    return NULL;
  if (value == Py_None) {
    succeed = PETSC_TRUE;
  } else if (value == Py_False) {
    succeed = PETSC_FALSE;
  } else if (value == Py_True) {
    succeed = PETSC_TRUE;
  }  else {
    PyErr_SetString(PyExc_TypeError, "routine must return None or Boolean");
    goto fail;
  }
  *outsucceed = succeed;
  return value;
 fail:
  Py_DecRef(value);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "SNESLinearSolve_Python"
static PetscErrorCode SNESLinearSolve_Python(SNES snes,Vec b,Vec x,PetscBool *succeed, PetscInt *its)
{
  KSPConvergedReason kspreason;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  *succeed = PETSC_TRUE; *its=0;
  SNES_PYTHON_CALL_MAYBE_RET(snes, "linearSolve", ("O&O&O&",
                                                   PyPetscSNES_New, snes,
                                                   PyPetscVec_New,  b,
                                                   PyPetscVec_New,  x    ),
                             notimplemented,
                             SNESPyObjToPetscBool, succeed);
 finally:
  if (!(*succeed)) {
    if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
      ierr = PetscInfo2(snes,"iter=%D, reached maximum allowed linear solve failures %D\n",
                        snes->iter,snes->maxLinearSolveFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
    } else *succeed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
notimplemented: /* default linear solve */
  *succeed = PETSC_TRUE; *its=0;
  ierr = SNES_KSPSolve(snes,snes->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(snes->ksp,its);CHKERRQ(ierr);
  if (kspreason < 0) *succeed = PETSC_FALSE;
  goto finally;
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearch_Python"
static PetscErrorCode SNESLineSearch_Python(SNES snes,Vec x,Vec y, Vec F,PetscBool *succeed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *succeed = PETSC_TRUE;
  SNES_PYTHON_CALL_MAYBE_RET(snes, "lineSearch", ("O&O&O&O&",
                                                  PyPetscSNES_New, snes,
                                                  PyPetscVec_New,  x,
                                                  PyPetscVec_New,  y,
                                                  PyPetscVec_New,  F    ),
                             notimplemented,
                             SNESPyObjToPetscBool, succeed);
 finally:
  if (!(*succeed)) {
    if (++snes->numFailures >= snes->maxFailures) {
      ierr = PetscInfo2(snes,"iter=%D, reached maximum allowed line search failures %D\n",
                        snes->iter,snes->maxFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LINE_SEARCH;
    } else *succeed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
 notimplemented: /* default, no line search */
  ierr = VecAXPY(x,-1.0,y);CHKERRQ(ierr);                     /* X <- X - Y       */
  ierr = SNESComputeFunction_Python(snes,x,F);CHKERRQ(ierr);  /* F <- function(X) */
  *succeed = PETSC_TRUE;
  goto finally;
}

/* -------------------------------------------------------------------------- */
/*
   SNESSolve_Python - Solves a nonlinear system

   Input Parameters:
.  snes - the SNES context

   Application Interface Routine: SNESSolve()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_Python"
static PetscErrorCode SNESSolve_Python(SNES snes)
{
  SNES_Py        *py = (SNES_Py *)snes->data;
  Vec            X,F,Y;
  PetscBool      succeed;
  PetscInt       i=0,lits=0;
  PetscReal      fnorm,ynorm=0,xnorm=0;
  MatStructure   flg = DIFFERENT_NONZERO_PATTERN;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!snes->vec_sol_update) {
    ierr = VecDuplicate(snes->vec_sol,&snes->vec_sol_update);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->vec_sol_update);CHKERRQ(ierr);
  }
  if (!snes->ops->converged) {
    snes->ops->converged = SNESSkipConverged;
  }

  snes->iter   = 0;
  snes->norm   = 0;
  snes->ttol   = 0;
  snes->reason = SNES_CONVERGED_ITERATING;

  /* Call user presolve routine */
  ierr = (*py->ops->presolve)(snes);CHKERRQ(ierr);

  X = snes->vec_sol;        /* solution vector */
  F = snes->vec_func;       /* residual vector */
  Y = snes->vec_sol_update; /* update vector */

  ierr = (*py->ops->computefunction)(snes,X,F);CHKERRQ(ierr); /* F <- function(X) */
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);              /* fnorm <- ||F||   */
  /* Check function norm */
  if (fnorm != fnorm)
    SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  snes->norm = fnorm;
  /* Set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  /* Monitor convergence */
  SNESMonitor(snes,0,fnorm);
  SNESLogConvHistory(snes,fnorm,0);

  /* Test for convergence */
  ierr = (*snes->ops->converged)(snes,0,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);

  for (i=0; !snes->reason && i<snes->max_its; i++) {

    /* call user prestep routine */
    ierr = (*py->ops->prestep)(snes, i);CHKERRQ(ierr);

    /* Solve J Y = F, where J <- jacobian(X) is the Jacobian matrix,  */
    ierr = (*py->ops->computejacobian)(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    if (snes->jacobian) { ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr); }
    ierr = (*py->ops->linearsolve)(snes,F,Y,&succeed,&lits);CHKERRQ(ierr);
    if (!succeed) { ierr = PetscInfo(snes,"linear solve failure, stopping solve\n");CHKERRQ(ierr); break; }
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",i,lits);CHKERRQ(ierr);
    snes->linear_its += lits;

    /* Line Search */
    ierr = (*py->ops->linesearch)(snes,X,Y,F,&succeed);CHKERRQ(ierr);
    if (!succeed) { ierr = PetscInfo(snes,"line search failure, stopping solve\n");CHKERRQ(ierr); break; }
    ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);  /* xnorm <- || X || */
    ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);  /* ynorm <- || Y || */
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);  /* fnorm <- || F || */
    if (xnorm != xnorm)
      SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    if (fnorm != fnorm)
      SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    if (ynorm != ynorm)
      SETERRQQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    ierr = PetscInfo4(snes,"iter=%D, xnorm=%18.16e, ynorm=%18.16e, fnorm=%18.16e\n",i,xnorm,ynorm,fnorm);CHKERRQ(ierr);
    snes->norm = fnorm;

    /* Call user poststep routine */
    ierr = (*py->ops->poststep)(snes, i);CHKERRQ(ierr);

    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,i+1,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);

    /* Monitor convergence */
    SNESMonitor(snes,i+1,fnorm);
    SNESLogConvHistory(snes,fnorm,lits);

    snes->iter = i+1;
  }

  if (i == snes->max_its) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
    if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }

  /* Call user postsolve routine */
  ierr = (*py->ops->postsolve)(snes);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_Python - Sets up the internal data structures for the later use
   of the SNESPYTHON nonlinear solver.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetUp()
 */
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_Python"
static PetscErrorCode SNESSetUp_Python(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!snes->vec_sol_update) {
    if (snes->vec_sol) {
      ierr = VecDuplicate(snes->vec_sol,&snes->vec_sol_update);CHKERRQ(ierr);
    } else if (snes->jacobian) {
      ierr = MatGetVecs(snes->jacobian,&snes->vec_sol_update,PETSC_NULL);CHKERRQ(ierr);
    }
    if (snes->vec_sol_update) {
      ierr = PetscLogObjectParent(snes,snes->vec_sol_update);CHKERRQ(ierr);
    }
  }
  SNES_PYTHON_CALL_SNESARG(snes, "setUp");
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESPythonSetType_PYTHON"
PetscErrorCode SNESPythonSetType_PYTHON(SNES snes,const char pyname[])
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* create the Python object from module/class/function  */
  ierr = PetscCreatePythonObject(pyname,&self);CHKERRQ(ierr);
  /* set the created Python object in SNES context */
  ierr = SNESPythonSetContext(snes,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
   SNESView_Python - Prints info from the SNESPYTHON data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESView_Python"
static PetscErrorCode SNESView_Python(SNES snes,PetscViewer viewer)
{
  SNES_Py        *py = (SNES_Py *)snes->data;
  PetscBool      isascii,isstring;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    const char* pyname  = py->pyname ? py->pyname : "no yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  Python: %s\n",pyname);CHKERRQ(ierr);
  }
  if (isstring) {
    const char* pyname  = py->pyname ? py->pyname : "<unknown>";
    ierr = PetscViewerStringSPrintf(viewer,"%s",pyname);CHKERRQ(ierr);
  }
  SNES_PYTHON_CALL(snes, "view", ("O&O&",
                                  PyPetscSNES_New,   snes,
                                  PyPetscViewer_New, viewer));
  PetscFunctionReturn(0);
}

/*
   SNESSetFromOptions_Python - Sets various parameters for the SNESPYTHON method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef  __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Python"
static PetscErrorCode SNESSetFromOptions_Python(SNES snes)
{
  SNES_Py        *py = (SNES_Py *)snes->data;
  char           pyname[2*PETSC_MAX_PATH_LEN+3];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Python options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-snes_python_type","Python package.module[.{class|function}]",
                            "SNESPythonSetType",py->pyname,pyname,sizeof(pyname),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg && pyname[0]) {
    ierr = PetscStrcmp(py->pyname,pyname,&flg);CHKERRQ(ierr);
    if (!flg) { ierr = SNESPythonSetType_PYTHON(snes,pyname);CHKERRQ(ierr); }
  }
  SNES_PYTHON_CALL_SNESARG(snes, "setFromOptions");
  PetscFunctionReturn(0);
}

#if !(PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "SNESReset_Python"
static PetscErrorCode SNESReset_Python(SNES snes)
{
  PetscFunctionBegin;
  PETSC_PYTHON_INCREF(snes);
  SNES_PYTHON_CALL_SNESARG(snes, "reset");
  PETSC_PYTHON_DECREF(snes);
  PetscFunctionReturn(0);
}
#endif

/*
   SNESDestroy_Python - Destroys the private SNES_Py context that was created
   with SNESCreate_Python().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_Python"
static PetscErrorCode SNESDestroy_Python(SNES snes)
{
  SNES_Py        *py = (SNES_Py *)snes->data;
  PyObject       *self = py->self;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (Py_IsInitialized()) {
    PETSC_PYTHON_INCREF(snes);
    SNES_PYTHON_CALL_SNESARG(snes, "destroy");
    PETSC_PYTHON_DECREF(snes);
    py->self = NULL; Py_DecRef(self);
  }
#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
  if (snes->vec_sol_update) {
    ierr = VecDestroy(snes->vec_sol_update);CHKERRQ(ierr);
    snes->vec_sol_update = PETSC_NULL;
  }
#endif
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  snes->data = PETSC_NULL;
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESPythonSetType_C",
                                    "",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
      SNESPYTHON -

  Level: beginner

.seealso:  SNES, SNESCreate(), SNESSetType(), SNESLS, SNESTR

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_Python"
PetscErrorCode SNESCreate_Python(SNES snes)
{
  SNES_Py        *py;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscPythonImportPetsc4Py();CHKERRQ(ierr);

  ierr       = PetscNew(SNES_Py,&py);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory(snes, sizeof(SNES_Py));CHKERRQ(ierr);
  snes->data = (void*)py;

  /* Python */
  py->self    = NULL;
  py->pyname  = NULL;
  py->ops     = &py->_ops;

  py->ops->presolve          = SNESPreSolve_Python;
  py->ops->postsolve         = SNESPostSolve_Python;
  py->ops->prestep           = SNESPreStep_Python;
  py->ops->poststep          = SNESPostStep_Python;

  py->ops->computefunction   = SNESComputeFunction_Python;
  py->ops->computejacobian   = SNESComputeJacobian_Python;
  py->ops->linearsolve       = SNESLinearSolve_Python;
  py->ops->linesearch        = SNESLineSearch_Python;

  /* PETSc */
  snes->vec_sol_update = PETSC_NULL;

  snes->ops->converged       = SNESDefaultConverged;
  snes->ops->computescaling  = PETSC_NULL;
  snes->ops->update          = PETSC_NULL;

  snes->ops->destroy         = SNESDestroy_Python;
  snes->ops->setfromoptions  = SNESSetFromOptions_Python;
  snes->ops->view            = SNESView_Python;
  snes->ops->setup           = SNESSetUp_Python;
  snes->ops->solve           = SNESSolve_Python;
#if !(PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
  snes->ops->reset           = SNESReset_Python;
#endif

  ierr = PetscObjectComposeFunction((PetscObject)snes,
                                    "SNESPythonSetType_C","SNESPythonSetType_PYTHON",
                                    (PetscVoidFunction)SNESPythonSetType_PYTHON);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESPythonGetContext"
/*@
   SNESPythonGetContext - .

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  ctx - Python context

   Level: beginner

.keywords: SNES, preconditioner, create

.seealso: SNES, SNESCreate(), SNESSetType(), SNESPYTHON
@*/
PetscErrorCode SNESPythonGetContext(SNES snes,void **ctx)
{
  SNES_Py        *py;
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(ctx,2);
  *ctx = NULL;
  ierr = PetscTypeCompare((PetscObject)snes,SNESPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (SNES_Py *) snes->data;
  *ctx = (void *) py->self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESPythonSetContext"
/*@
   SNESPythonSetContext - .

   Collective on SNES

   Input Parameters:
.  snes - SNES context
.  ctx - Python context

   Level: beginner

.keywords: SNES, create

.seealso: SNES, SNESCreate(), SNESSetType(), SNESPYTHON
@*/
PetscErrorCode SNESPythonSetContext(SNES snes,void *ctx)
{
  SNES_Py        *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscBool      ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)snes,SNESPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (SNES_Py *) snes->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the SNES object */
  SNES_PYTHON_CALL_SNESARG(snes, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the SNES object  */
  py->self = (PyObject *) self; Py_IncRef(py->self);
  ierr = PetscFree(py->pyname);CHKERRQ(ierr);
  ierr = PetscPythonGetFullName(py->self,&py->pyname);CHKERRQ(ierr);
  SNES_PYTHON_CALL_SNESARG(snes, "create");
  if (snes->setupcalled) snes->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#if 0

PETSC_EXTERN_CXX_BEGIN
extern PetscErrorCode SNESPythonSetType(SNES,const char[]);
PETSC_EXTERN_CXX_END

#undef __FUNCT__
#define __FUNCT__ "SNESPythonSetType"
/*@C
   SNESPythonSetType - Initalize a SNES object implemented in Python.

   Collective on SNES

   Input Parameter:
+  snes - the nonlinear solver (SNES) context.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -snes_python_type <pyname>

   Level: intermediate

.keywords: SNES, Python

.seealso: SNESCreate(), SNESSetType(), SNESPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode SNESPythonSetType(SNES snes,const char pyname[])
{
  PetscErrorCode (*f)(SNES, const char[]) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESPythonSetType_C",
                                  (PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(snes,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
