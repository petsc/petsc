#include "petsc/private/vecimpl.h"
#include "petsc/private/matimpl.h"
#include "petsc/private/pcimpl.h"
#include "petsc/private/kspimpl.h"
#include "petsc/private/snesimpl.h"
#include "petsc/private/tsimpl.h"

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char*);

static inline
PetscErrorCode PetscObjectComposedDataGetIntPy(PetscObject o, PetscInt id, PetscInt *v, PetscBool *exist)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposedDataGetInt(o,id,*v,*exist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode PetscObjectComposedDataSetIntPy(PetscObject o, PetscInt id, PetscInt v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposedDataSetInt(o,id,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode PetscObjectComposedDataRegisterPy(PetscInt *id)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposedDataRegister(id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode KSPLogHistory(KSP ksp,PetscReal rnorm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode SNESLogHistory(SNES snes,PetscReal rnorm,PetscInt lits)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESLogConvergenceHistory(snes,rnorm,lits);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode KSPConverged(KSP ksp,
                            PetscInt iter,PetscReal rnorm,
                            KSPConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (reason) PetscValidPointer(reason,2);
  if (!iter) ksp->rnorm0 = rnorm;
  if (!iter) {
    ksp->reason = KSP_CONVERGED_ITERATING;
    ksp->ttol = PetscMax(rnorm*ksp->rtol,ksp->abstol);
  }
  if (ksp->converged) {
    ierr = (*ksp->converged)(ksp,iter,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  } else {
    ierr = KSPConvergedSkip(ksp,iter,rnorm,&ksp->reason,NULL);CHKERRQ(ierr);
    /*ierr = KSPConvergedDefault(ksp,iter,rnorm,&ksp->reason,NULL);CHKERRQ(ierr);*/
  }
  ksp->rnorm = rnorm;
  if (reason) *reason = ksp->reason;
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode SNESConverged(SNES snes,
                             PetscInt iter,PetscReal xnorm,PetscReal ynorm,PetscReal fnorm,
                             SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (reason) PetscValidPointer(reason,2);
  if (!iter) {
    snes->reason = SNES_CONVERGED_ITERATING;
    snes->ttol = fnorm*snes->rtol;
  }
  if (snes->ops->converged) {
    ierr = (*snes->ops->converged)(snes,iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  } else {
    ierr = SNESConvergedSkip(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);
    /*ierr = SNESConvergedDefault(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);*/
  }
  snes->norm = fnorm;
  if (reason) *reason = snes->reason;
  PetscFunctionReturn(0);
}

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON ((PetscErrorCode)(-1))
#endif

#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
        PetscError(comm,__LINE__,FUNCT,__FILE__,n,t,msg,arg)

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initlibpetsc4py(void);
#else
#undef  CYTHON_PEP489_MULTI_PHASE_INIT
#define CYTHON_PEP489_MULTI_PHASE_INIT 0
PyMODINIT_FUNC PyInit_libpetsc4py(void);
static void initlibpetsc4py(void)
{
  PyObject *M, *m;
  M = PyImport_GetModuleDict();
  if (!M) return;
  m = PyInit_libpetsc4py();
  if (!m) return;
  PyDict_SetItemString(M, "libpetsc4py", m);
  Py_DECREF(m);
}
#endif
