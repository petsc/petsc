/*  -------------------------------------------------------------------- */

#include "src/inline/python.h"
#include "include/private/snesimpl.h"

/*  -------------------------------------------------------------------- */

/* backward compatibility hacks */

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 2 &&	\
     PETSC_VERSION_RELEASE  == 1)
#define SNES_KSPSolve(snes,ksp,b,x) KSPSolve(ksp,b,x)
#define SNESDefaultConverged SNESConverged_LS
#define SNES_CONVERGED_ITS   ((SNESConvergedReason)5)
#define vec_sol_update       vec_sol_update_always
#define vec_rhs              afine
#endif

#if (PETSC_VERSION_MAJOR    == 2 &&	\
     PETSC_VERSION_MINOR    == 3 &&	\
     PETSC_VERSION_SUBMINOR == 3 &&	\
     PETSC_VERSION_RELEASE  == 1)
#define vec_sol_update       vec_sol_update_always
#define vec_rhs              afine
#endif

/*  -------------------------------------------------------------------- */

#define SNESPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreatePython(MPI_Comm,const char *,const char *,SNES*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonSetContext(SNES,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonGetContext(SNES,void**);
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
  PetscErrorCode (*linearsolve)(SNES,Vec,Vec,PetscTruth*,PetscInt*);
  PetscErrorCode (*linesearch)(SNES,Vec,Vec,Vec,PetscTruth*);

  PetscErrorCode (*monitor)(SNES,PetscInt,PetscReal);
  PetscErrorCode (*converged)(SNES,PetscInt,
			      PetscReal,PetscReal,PetscReal,
			      SNESConvergedReason*);
};

typedef struct {
  PyObject   *self;
  char       *module;
  char       *factory;
  /* XXX get rid of this ? */
  SNESPyOps  ops;
  struct _SNESPyOps _ops;
} SNES_Py;


/* -------------------------------------------------------------------------- */

#define SNES_Py_Self(snes) (((SNES_Py*)(snes)->data)->self)

#define SNES_PYTHON_CALL_HEAD(snes, PyMethod)		\
  PETSC_PYTHON_CALL_HEAD(SNES_Py_Self(snes), PyMethod)
#define SNES_PYTHON_CALL_JUMP(snes, LABEL)		\
  PETSC_PYTHON_CALL_JUMP(LABEL)				
#define SNES_PYTHON_CALL_BODY(snes, ARGS)		\
  PETSC_PYTHON_CALL_BODY(ARGS)
#define SNES_PYTHON_CALL_TAIL(snes, PyMethod)		\
  PETSC_PYTHON_CALL_TAIL()


#define SNES_PYTHON_CALL(snes, PyMethod, ARGS)		\
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);		\
  SNES_PYTHON_CALL_BODY(snes, ARGS);			\
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)			\
/**/  

#define SNES_PYTHON_CALL_NOARGS(snes, PyMethod)			\
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);			\
  SNES_PYTHON_CALL_BODY(snes, ("", NULL));			\
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)				\
  /**/

#define SNES_PYTHON_CALL_SNESARG(snes, PyMethod)		\
  SNES_PYTHON_CALL_HEAD(snes, PyMethod);			\
  SNES_PYTHON_CALL_BODY(snes, ("O&",PyPetscSNES_New,snes));	\
  SNES_PYTHON_CALL_TAIL(snes, PyMethod)				\
/**/

/* -------------------------------------------------------------------------- */

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
  SNES_PYTHON_CALL_HEAD(snes, "computeFunction");
  SNES_PYTHON_CALL_JUMP(snes, computefunc);
  SNES_PYTHON_CALL_BODY(snes, ("O&O&O&",
			       PyPetscSNES_New, snes,
			       PyPetscVec_New,  x,
			       PyPetscVec_New,  F    ));
  SNES_PYTHON_CALL_TAIL(snes, "computeFunction");
  if (nfuncs == snes->nfuncs) { /* snes->ops->computefunction was not called */
    if (snes->vec_rhs) { ierr = VecAXPY(F,-1.0,snes->vec_rhs);CHKERRQ(ierr); }
    snes->nfuncs++; /* increment function call counter*/
  }
  PetscFunctionReturn(0);
 computefunc:
  ierr = SNESComputeFunction(snes,x,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PyObject * SNESPyObjToMatStructure(PyObject *value, MatStructure *outflag)
{
  MatStructure flag = DIFFERENT_NONZERO_PATTERN;
  if (value == NULL) 
    return NULL;
  if (value == Py_None) {
    flag = SAME_NONZERO_PATTERN;
  } else if (value == Py_False) {
    flag = SAME_NONZERO_PATTERN;
  } else if (value == Py_True) {
    flag = DIFFERENT_NONZERO_PATTERN;
  } else if (PyInt_Check(value)) {
    flag = (MatStructure) PyInt_AsLong(value);
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
  *outflag = flag;
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
  SNES_PYTHON_CALL_HEAD(snes, "computeJacobian");
  SNES_PYTHON_CALL_JUMP(snes, computejac);
  SNES_PYTHON_CALL_BODY(snes, ("O&O&O&O&",
			       PyPetscSNES_New, snes,
			       PyPetscVec_New,  x,
			       PyPetscMat_New,  *A,
			       PyPetscMat_New,  *B    ));
  _retv = SNESPyObjToMatStructure(_retv, flg);
  SNES_PYTHON_CALL_TAIL(snes, "computeJacobian");
  PetscFunctionReturn(0);
 computejac:
  ierr = SNESComputeJacobian(snes,x,A,B,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PyObject * SNESPyObjToPetscTruth(PyObject *value, PetscTruth *outsucceed)
{
  PetscTruth succeed = PETSC_FALSE;
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
static PetscErrorCode SNESLinearSolve_Python(SNES snes,Vec b,Vec x,PetscTruth *succeed, PetscInt *its)
{
  KSPConvergedReason kspreason;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  *succeed = PETSC_TRUE; *its=0;
  SNES_PYTHON_CALL_HEAD(snes, "linearSolve");
  SNES_PYTHON_CALL_JUMP(snes, linearsolve);
  SNES_PYTHON_CALL_BODY(snes, ("O&O&O&",
			       PyPetscSNES_New, snes ,
			       PyPetscVec_New,  b    ,
			       PyPetscVec_New,  x    ));
  _retv = SNESPyObjToPetscTruth(_retv, succeed);
  *its=0; /* XXX Fix this, try to get result */
  SNES_PYTHON_CALL_TAIL(snes, "linearSolve");
 finally:
  if (!(*succeed)) {
    if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
      ierr = PetscInfo2(snes,"iter=%D, reached maximum allowed linear solve failures %D\n",
			snes->iter,snes->maxLinearSolveFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
    } else *succeed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
 linearsolve: /* default linear solve */
  *succeed = PETSC_TRUE; *its=0;
  ierr = SNES_KSPSolve(snes,snes->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(snes->ksp,its);CHKERRQ(ierr);
  if (kspreason < 0) *succeed = PETSC_FALSE;
  goto finally;
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearch_Python"
static PetscErrorCode SNESLineSearch_Python(SNES snes,Vec x,Vec y, Vec F,PetscTruth *succeed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *succeed = PETSC_TRUE;
  SNES_PYTHON_CALL_HEAD(snes, "lineSearch");
  SNES_PYTHON_CALL_JUMP(snes, linesearch);
  SNES_PYTHON_CALL_BODY(snes, ("O&O&O&O&",
			       PyPetscSNES_New, snes ,
			       PyPetscVec_New,  x    ,
			       PyPetscVec_New,  y    ,
			       PyPetscVec_New,  F    ));
  _retv = SNESPyObjToPetscTruth(_retv, succeed);
  SNES_PYTHON_CALL_TAIL(snes, "lineSearch");
 finally:
  if (!(*succeed)) {
    if (++snes->numFailures >= snes->maxFailures) {
      ierr = PetscInfo2(snes,"iter=%D, reached maximum allowed line search failures %D\n",
			snes->iter,snes->maxFailures);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_LS_FAILURE;
    } else *succeed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
 linesearch: /* default, no line search */
  ierr = VecAXPY(x,-1.0,y);CHKERRQ(ierr);                     /* X <- X - Y       */
  ierr = SNESComputeFunction_Python(snes,x,F);CHKERRQ(ierr);  /* F <- function(X) */
  *succeed = PETSC_TRUE;
  goto finally;
}


#undef __FUNCT__
#define __FUNCT__ "SNESMonitor_Python"
static PetscErrorCode SNESMonitor_Python(SNES snes, PetscInt its, PetscReal fnorm)
{
  PetscFunctionBegin;
  SNES_PYTHON_CALL(snes, "monitor", ("O&ld",
				     PyPetscSNES_New, snes,
				     (long)            its, 
				     (double)          fnorm ));
  /* call default monitors anyway */
  SNESMonitor(snes,its,fnorm);
  PetscFunctionReturn(0);
}

static PyObject * SNESPyObjToConvReason(PyObject *value, SNESConvergedReason *outreason)
{
  SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
  if (value == NULL) 
    return NULL;
  if (value == Py_None) {
    reason = SNES_CONVERGED_ITERATING;
  } else if (value == Py_False) {
    reason = SNES_CONVERGED_ITERATING;
  } else if (value == Py_True) {
    reason = SNES_CONVERGED_ITS;
  } else if (PyInt_Check(value)) {
    reason = (SNESConvergedReason) PyInt_AsLong(value);
    if (reason < SNES_DIVERGED_LOCAL_MIN ||
	reason > SNES_CONVERGED_TR_DELTA) {
      PyErr_SetString(PyExc_ValueError,
		      "convergence test returned an out of range "
		      "integer value for SNESConvergedReason"); 
      goto fail;
    }
  } else {
    PyErr_SetString(PyExc_TypeError,
		    "convergence test must return None, Boolean, "
		    "or a valid integer value for SNESConvergedReason");
    goto fail;
  }
  *outreason = reason;
  return value;
 fail:
  Py_DecRef(value);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "SNESConverged_Python"
static PetscErrorCode SNESConverged_Python(SNES snes, PetscInt its,
					   PetscReal xnorm, PetscReal ynorm, PetscReal fnorm,
					   SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  SNES_PYTHON_CALL_HEAD(snes, "converged");
  SNES_PYTHON_CALL_BODY(snes, ("O&lddd",
			       PyPetscSNES_New,  snes,
			       (long)             its,
			       (double)           xnorm,
			       (double)           ynorm,
			       (double)           fnorm));
  _retv = SNESPyObjToConvReason(_retv, reason);
  SNES_PYTHON_CALL_TAIL(snes, "converged");
  /* call default convergence test anyway if not converged */
  if (!(*reason) && snes->ops->converged) {
    ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,reason,snes->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  PetscInt       i=0,lits=0;
  PetscReal      fnorm,ynorm=0,xnorm=0;
  MatStructure   flg = DIFFERENT_NONZERO_PATTERN;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  snes->iter = 0;
  snes->norm = 0;
  snes->ttol  = 0;
  snes->reason = SNES_CONVERGED_ITERATING;

  /* Call user presolve routine */
  ierr = (*py->ops->presolve)(snes);CHKERRQ(ierr);

  X = snes->vec_sol;	    /* solution vector */
  F = snes->vec_func;	    /* residual vector */
  Y = snes->vec_sol_update; /* update vector */

  ierr = (*py->ops->computefunction)(snes,X,F);CHKERRQ(ierr); /* F <- function(X) */
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);               /* fnorm <- ||F||   */
  /* Check function norm */
  if (fnorm != fnorm) SETERRQ(PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  snes->norm = fnorm;
  /* Set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  /* Monitor convergence */
  ierr = (*py->ops->monitor)(snes,snes->iter,snes->norm);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);

  /* Test for convergence */
  ierr = (*py->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason);CHKERRQ(ierr);

  for (i=0; !snes->reason && i<snes->max_its; i++) {
    
    PetscTruth succeed;

    /* call user prestep routine */
    ierr = (*py->ops->prestep)(snes, i);CHKERRQ(ierr);

    /* Solve J Y = F, where J <- jacobian(X) is the Jacobian matrix,  */
    ierr = (*py->ops->computejacobian)(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
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
    if (xnorm != xnorm) SETERRQ(PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    if (fnorm != fnorm) SETERRQ(PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    if (ynorm != ynorm) SETERRQ(PETSC_ERR_FP,"User provided routine generated a Not-a-Number");
    ierr = PetscInfo4(snes,"iter=%D, xnorm=%18.16e, ynorm=%18.16e, fnorm=%18.16e\n",i,xnorm,ynorm,fnorm);CHKERRQ(ierr);
    snes->norm = fnorm;
    
    /* Call user poststep routine */
    ierr = (*py->ops->poststep)(snes, i);CHKERRQ(ierr);

    /* Test for convergence */
    ierr = (*py->ops->converged)(snes,i+1,xnorm,ynorm,fnorm,&snes->reason);CHKERRQ(ierr);

    /* Monitor convergence */
    ierr = (*py->ops->monitor)(snes,i+1,fnorm);CHKERRQ(ierr);
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
    } else {
      ierr = MatGetVecs(snes->jacobian,&snes->vec_sol_update, PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent(snes,snes->vec_sol_update);CHKERRQ(ierr);
  }
  SNES_PYTHON_CALL_SNESARG(snes, "setUp");
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
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
  SNES_PYTHON_CALL(snes, "view", ("O&O&", 
				  PyPetscSNES_New,   snes,
				  PyPetscViewer_New, viewer));
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_Python - Sets various parameters for the SNESPYTHON method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Python"
static PetscErrorCode SNESSetFromOptions_Python(SNES snes)
{
  char           *modcls[2] = { 0, 0};
  PetscInt       nmax = 2;
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Python options");CHKERRQ(ierr);
  ierr = PetscOptionsStringArray("-snes_python","Python module and class/factory",
				 "SNESCreatePython", modcls,&nmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg) {
    if (nmax == 2) {
      PyObject *self = NULL;
      ierr = PetscCreatePythonObject(modcls[0],modcls[1],&self);CHKERRQ(ierr);
      ierr = SNESPythonSetContext(snes,self);Py_DecRef(self);CHKERRQ(ierr);
    }
    ierr = PetscStrfree(modcls[0]); CHKERRQ(ierr);
    ierr = PetscStrfree(modcls[1]); CHKERRQ(ierr);
  }
  SNES_PYTHON_CALL_SNESARG(snes, "setFromOptions");
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
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
    SNES_PYTHON_CALL_NOARGS(snes, "destroy");
    py->self = NULL; Py_DecRef(self);
  }
  if (snes->vec_sol_update) {
    ierr = VecDestroy(snes->vec_sol_update);CHKERRQ(ierr);
    snes->vec_sol_update = PETSC_NULL;
  }
  ierr = PetscStrfree(py->module);CHKERRQ(ierr);
  ierr = PetscStrfree(py->factory);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  snes->data = PETSC_NULL;
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
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_Python(SNES snes)
{
  SNES_Py        *py;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscInitializePython();CHKERRQ(ierr);

  ierr       = PetscNew(SNES_Py,&py);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory(snes, sizeof(SNES_Py));CHKERRQ(ierr);
  snes->data = (void*)py;

  /* Python */
  py->self    = NULL;
  py->module  = NULL;
  py->factory = NULL;
  py->ops = &py->_ops;

  py->ops->presolve          = SNESPreSolve_Python;
  py->ops->postsolve         = SNESPostSolve_Python;
  py->ops->prestep           = SNESPreStep_Python;
  py->ops->poststep          = SNESPostStep_Python;

  py->ops->computefunction   = SNESComputeFunction_Python;
  py->ops->computejacobian   = SNESComputeJacobian_Python;
  py->ops->linearsolve       = SNESLinearSolve_Python;
  py->ops->linesearch        = SNESLineSearch_Python;

  py->ops->monitor           = SNESMonitor_Python;
  py->ops->converged         = SNESConverged_Python;
  
  /* PETSc */
  snes->ops->converged       = SNESDefaultConverged;
  snes->ops->computescaling  = PETSC_NULL;
  snes->ops->update          = PETSC_NULL;

  snes->ops->destroy	     = SNESDestroy_Python;
  snes->ops->setfromoptions  = SNESSetFromOptions_Python;
  snes->ops->view            = SNESView_Python;
  snes->ops->setup	     = SNESSetUp_Python;
  snes->ops->solve	     = SNESSolve_Python;

  snes->vec_sol_update = PETSC_NULL;

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
PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonGetContext(SNES snes,void **ctx)
{
  SNES_Py        *py;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
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
PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonSetContext(SNES snes,void *ctx)
{
  SNES_Py        *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)snes,SNESPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (SNES_Py *) snes->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the SNES object */
  SNES_PYTHON_CALL_NOARGS(snes, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the SNES object  */
  py->self = (PyObject *) self; Py_IncRef(py->self);
  SNES_PYTHON_CALL_SNESARG(snes, "create");
  if (snes->setupcalled) snes->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCreatePython"
/*@
   SNESCreatePython - Creates a Python nonlinear solver context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator 
.  modname - module name
.  clsname - factory/class name

   Output Parameter:
.  snes - location to put the nonlinear solver context

   Level: beginner

.keywords: SNES,  create

.seealso: SNES, SNESCreate(), SNESSetType(), SNESPYTHON
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreatePython(MPI_Comm comm,
						    const char *modname,
						    const char *clsname,
						    SNES *snes)
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (modname) PetscValidCharPointer(modname,2);
  if (clsname) PetscValidCharPointer(clsname,3);
  /* create the SNES context and set its type */
  ierr = SNESCreate(comm,snes);CHKERRQ(ierr);
  ierr = SNESSetType(*snes,SNESPYTHON);CHKERRQ(ierr);
  if (modname == PETSC_NULL) PetscFunctionReturn(0);
  if (clsname == PETSC_NULL) PetscFunctionReturn(0);
  /* create the Python object from module and class/factory  */
  ierr = PetscCreatePythonObject(modname,clsname,&self);CHKERRQ(ierr);
  /* set the created Python object in SNES context */
  ierr = SNESPythonSetContext(*snes,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#if defined(vec_sol_update)
#undef vec_sol_update
#endif

#if defined(vec_rhs)
#undef vec_rhs
#endif

/* -------------------------------------------------------------------------- */
