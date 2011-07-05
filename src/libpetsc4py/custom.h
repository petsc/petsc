#include <private/vecimpl.h>
#include <private/matimpl.h>
#include <private/pcimpl.h>
#include <private/kspimpl.h>
#include <private/snesimpl.h>
#include <private/tsimpl.h>

EXTERN_C_BEGIN
extern PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char*);
EXTERN_C_END

EXTERN_C_BEGIN
#define PetscFwkPythonCall_C        PetscFwkPythonCall
#define PetscFwkPythonLoadVTable_C  PetscFwkPythonLoadVTable
#define PetscFwkPythonClearVTable_C PetscFwkPythonClearVTable
extern PetscErrorCode (*PetscFwkPythonCall_C)(PetscFwk,const char*,void*);
extern PetscErrorCode (*PetscFwkPythonLoadVTable_C)(PetscFwk,const char*,const char*,void**);
extern PetscErrorCode (*PetscFwkPythonClearVTable_C)(PetscFwk,void**);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPLogHistory"
PETSC_STATIC_INLINE
PetscErrorCode KSPLogHistory(KSP ksp,PetscInt iter,PetscReal rnorm)
{
  /*PetscErrorCode ierr;*/
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  KSPLogResidualHistory(ksp,rnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLogHistory"
PETSC_STATIC_INLINE
PetscErrorCode SNESLogHistory(SNES snes,PetscInt iter,PetscReal rnorm,PetscInt lits)
{
  /*PetscErrorCode ierr;*/
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SNESLogConvHistory(snes,rnorm,lits);
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
    ierr = KSPSkipConverged(ksp,iter,rnorm,&ksp->reason,NULL);CHKERRQ(ierr);
    /*ierr = KSPDefaultConverged(ksp,iter,rnorm,&ksp->reason,NULL);CHKERRQ(ierr);*/
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
    ierr = SNESSkipConverged(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);
    /*ierr = SNESDefaultConverged(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);*/
  }
  snes->norm = fnorm;
  if (reason) *reason = snes->reason;
  PetscFunctionReturn(0);
}


#if (PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0))
#define PETSC_ERROR_INITIAL 1
#define PETSC_ERROR_REPEAT  0
#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
  PetscError(__LINE__,FUNCT,__FILE__,__SDIR__,n,t,msg,arg)
#else
#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
  PetscError(comm,__LINE__,FUNCT,__FILE__,__SDIR__,n,t,msg,arg)
#endif

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initlibpetsc4py(void);
#else
PyMODINIT_FUNC PyInit_libpetsc4py(void);
#endif

#if PY_MAJOR_VERSION < 3
static PyObject *PyInit_libpetsc4py(void)
{
  PyObject *modules, *mod=NULL;
  initlibpetsc4py();
  if (PyErr_Occurred()) goto bad;
  modules = PyImport_GetModuleDict();
  if (!modules) goto bad;
  mod = PyDict_GetItemString(modules, "libpetsc4py");
  if (!mod) goto bad;
  Py_INCREF(mod);
  if (PyDict_DelItemString(modules, "libpetsc4py") < 0) goto bad;
  return mod;
 bad:
  Py_XDECREF(mod);
  return NULL;
}
#endif
