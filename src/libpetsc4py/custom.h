#include "petsc/private/vecimpl.h"
#include "petsc/private/matimpl.h"
#include "petsc/private/pcimpl.h"
#include "petsc/private/kspimpl.h"
#include "petsc/private/snesimpl.h"
#include "petsc/private/tsimpl.h"

#if PETSC_VERSION_LT(3,5,0)
typedef PetscObject Tao;
#endif

#if PETSC_VERSION_LT(3,4,0)
#define PetscObjectComposeFunction(o,n,f) \
        PetscObjectComposeFunction(o,n,"",(PetscVoidFunction)(f))
#define MatRegister(s,f)  MatRegister(s,0,0,f)
#define PCRegister(s,f)   PCRegister(s,0,0,f)
#define KSPRegister(s,f)  KSPRegister(s,0,0,f)
#define SNESRegister(s,f) SNESRegister(s,0,0,f)
#define TSRegister(s,f)   TSRegister(s,0,0,f)
#endif

EXTERN_C_BEGIN
extern PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char*);
EXTERN_C_END

#if PETSC_VERSION_LT(3,4,0)
#define KSPBuildSolutionDefault KSPDefaultBuildSolution
#define KSPBuildResidualDefault KSPDefaultBuildResidual
#endif

#if PETSC_VERSION_LT(3,5,0)
#define KSPConvergedSkip  KSPSkipConverged
#define SNESConvergedSkip SNESSkipConverged
#endif

#undef __FUNCT__
#define __FUNCT__ "KSPLogHistory"
PETSC_STATIC_INLINE
PetscErrorCode KSPLogHistory(KSP ksp,PetscReal rnorm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
#if PETSC_VERSION_LT(3,4,0)
  ierr=0;KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
#else
  ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLogHistory"
PETSC_STATIC_INLINE
PetscErrorCode SNESLogHistory(SNES snes,PetscReal rnorm,PetscInt lits)
{
    PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
#if PETSC_VERSION_LT(3,4,0)
  ierr=0;SNESLogConvHistory(snes,rnorm,lits);CHKERRQ(ierr);
#else
  ierr = SNESLogConvergenceHistory(snes,rnorm,lits);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPConverged"
PETSC_STATIC_INLINE
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

#undef __FUNCT__
#define __FUNCT__ "SNESConverged"
PETSC_STATIC_INLINE
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

#if PETSC_VERSION_LT(3,5,0)
#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
  PetscError(comm,__LINE__,FUNCT,__FILE__,__SDIR__,n,t,msg,arg)
#else
#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
  PetscError(comm,__LINE__,FUNCT,__FILE__,n,t,msg,arg)
#endif

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initlibpetsc4py(void);
#else
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

#undef  __FUNCT__
#define __FUNCT__ "<libpetsc4py>"
