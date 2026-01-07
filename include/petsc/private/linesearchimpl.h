#pragma once

#include <petscsnes.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      SNESLineSearchRegisterAllCalled;
PETSC_EXTERN PetscErrorCode SNESLineSearchRegisterAll(void);
PETSC_EXTERN PetscLogEvent  SNESLINESEARCH_Apply;

typedef struct _LineSearchOps *LineSearchOps;

struct _LineSearchOps {
  PetscErrorCode (*view)(SNESLineSearch, PetscViewer);
  SNESLineSearchApplyFn *apply;
  PetscErrorCode (*precheck)(SNESLineSearch, Vec, Vec, PetscBool *, void *);
  SNESLineSearchVIProjectFn  *viproject;
  SNESLineSearchVINormFn     *vinorm;
  SNESLineSearchVIDirDerivFn *vidirderiv;
  PetscErrorCode (*postcheck)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  PetscErrorCode (*setfromoptions)(SNESLineSearch, PetscOptionItems);
  PetscErrorCode (*reset)(SNESLineSearch);
  PetscErrorCode (*destroy)(SNESLineSearch);
  PetscErrorCode (*setup)(SNESLineSearch);
  PetscErrorCode (*snesfunc)(SNES, Vec, Vec);
};

#define MAXSNESLSMONITORS 5

struct _p_LineSearch {
  PETSCHEADER(struct _LineSearchOps);

  SNES snes;

  void *data;

  PetscBool setupcalled;

  Vec vec_sol;
  Vec vec_sol_new;
  Vec vec_func;
  Vec vec_func_new;
  Vec vec_update;

  PetscInt nwork;
  Vec     *work;

  PetscReal lambda;

  PetscBool norms;
  PetscReal fnorm;
  PetscReal ynorm;
  PetscReal xnorm;
  PetscBool keeplambda;

  PetscReal damping;
  PetscReal maxlambda;
  PetscReal minlambda;
  PetscInt  max_it;
  PetscReal rtol;
  PetscReal atol;
  PetscReal ltol;
  PetscInt  order;

  PetscReal precheck_picard_angle;

  void *precheckctx;
  void *postcheckctx;

  PetscBool jacobiandomainerror; /* set with SNESSetJacobianDomainError() */
  PetscBool checkjacdomainerror; /* does it check Jacobian domain error after Jacobian evaluations */

  SNESLineSearchReason reason;

  PetscViewer monitor;
  PetscErrorCode (*monitorftns[MAXSNESLSMONITORS])(SNESLineSearch, void *); /* monitor routine */
  PetscCtxDestroyFn *monitordestroy[MAXSNESLSMONITORS];                     /* monitor context destroy routine */
  void              *monitorcontext[MAXSNESLSMONITORS];                     /* monitor context */
  PetscInt           numbermonitors;                                        /* number of monitors */
};

/*MC
  SNESLineSearchCheckFunctionDomainError - Called after a `SNESComputeFunction()` and `VecNorm()` in a `SNES` line search to check if the function norm is infinity or NaN and
  if the function callback set with `SNESSetFunction()` called `SNESSetFunctionDomainError()`.

  Synopsis:
  #include <snesimpl.h>
  void SNESLineSearchCheckFunctionDomainError(SNES snes, SNESLineSearch ls, PetscReal fnorm)

  Collective

  Input Parameters:
+  snes  - the `SNES` object
.  ls    - the `SNESLineSearch` object
-  fnorm - the value of the norm

 Level: developer

 Notes:
 If `fnorm` is infinity or NaN and `SNESSetErrorIfNotConverged()` was set, this immediately generates a `PETSC_ERR_CONV_FAILED`.

 If `fnorm` is infinity or NaN and `SNESSetFunctionDomainError()` was called, this sets the `SNESLineSearchReason` to `SNES_LINESEARCH_FAILED_FUNCTION_DOMAIN`
 and exits the solver

 Otherwise, if `fnorm` is infinity or NaN, this sets the `SNESLineSearchReason` to `SNES_LINESEARCH_FAILED_NANORINF` and exits the line search

 See `SNESCheckFunctionDomainError()` for an explanation of the design

.seealso: [](ch_snes), `SNESCheckFunctionDomainError()`, `SNESSetFunctionDomainError()`, `PETSC_ERR_CONV_FAILED`, `SNESSetErrorIfNotConverged()`, `SNES_DIVERGED_FUNCTION_DOMAIN`,
          `SNESConvergedReason`, `SNES_DIVERGED_FUNCTION_NAN`
MC*/
#define SNESLineSearchCheckFunctionDomainError(snes, ls, fnorm) \
  do { \
    if (PetscIsInfOrNanReal(fnorm)) { \
      PetscCheck(!snes->errorifnotconverged, PetscObjectComm((PetscObject)ls), PETSC_ERR_NOT_CONVERGED, "SNES line search failure due to infinity or NaN norm"); \
      { \
        PetscBool functiondomainerror; \
        PetscCallMPI(MPIU_Allreduce(&snes->functiondomainerror, &functiondomainerror, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)ls))); \
        if (functiondomainerror) { \
          ls->reason                = SNES_LINESEARCH_FAILED_FUNCTION_DOMAIN; \
          snes->functiondomainerror = PETSC_FALSE; \
        } else ls->reason = SNES_LINESEARCH_FAILED_NANORINF; \
        PetscFunctionReturn(PETSC_SUCCESS); \
      } \
    } \
  } while (0)

/*MC
  SNESLineSearchCheckObjectiveDomainError - Called after a `SNESComputeObjective()` in a `SNESLineSearch` to check if the objective value is infinity or NaN and/or
  if the function callback set with `SNESSetObjective()` called `SNESSetObjectiveDomainError()`.

  Synopsis:
  #include <snesimpl.h>
  void SNESLineSearchCheckObjectiveDomainError(SNES snes, PetscReal fobj)

  Collective

  Input Parameters:
+ snes - the `SNES` solver object
- fobj - the value of the objective function

  Level: developer

  Notes:
  If `fobj` is infinity or NaN and `SNESSetErrorIfNotConverged()` was set, this immediately generates a `PETSC_ERR_CONV_FAILED`.

  If `SNESSetObjectiveDomainError()` was called, this sets the `SNESLineSearchReason` to `SNES_LINESEARCH_FAILED_OBJECTIVE_DOMAIN`
  and exits the line search

  Otherwise if `fobj` is infinity or NaN, this sets the `SNESLineSearchReason` to `SNES_LINESEARCH_FAILED_NANORINF` and exits the line search

.seealso: [](ch_snes), `SNESSetObjectiveDomainError()`, `PETSC_ERR_CONV_FAILED`, `SNESSetErrorIfNotConverged()`, `SNES_DIVERGED_OBJECTIVE_DOMAIN`, `SNES_DIVERGED_FUNCTION_DOMAIN`,
          `SNESSetFunctionDomainError()`, `SNESConvergedReason`, `SNES_DIVERGED_OBJECTIVE_NANORINF`, `SNES_DIVERGED_FUNCTION_NAN`, `SNESLineSearchCheckObjectiveDomainError()`
MC*/
#define SNESLineSearchCheckObjectiveDomainError(snes, fobj) \
  do { \
    if (snes->errorifnotconverged) { \
      PetscCheck(!snes->objectivedomainerror, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due objective domain error"); \
      PetscCheck(!PetscIsInfOrNanReal(fobj), PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due to infinity or NaN norm"); \
    } \
    if (snes->objectivedomainerror) { \
      snes->linesearch->reason   = SNES_LINESEARCH_FAILED_OBJECTIVE_DOMAIN; \
      snes->objectivedomainerror = PETSC_FALSE; \
      PetscFunctionReturn(PETSC_SUCCESS); \
    } else if (PetscIsInfOrNanReal(fobj)) { \
      snes->linesearch->reason = SNES_LINESEARCH_FAILED_NANORINF; \
      PetscFunctionReturn(PETSC_SUCCESS); \
    } \
  } while (0)

/*MC
  SNESLineSearchCheckJacobianDomainError - Called after a `SNESComputeJacobian()` in a SNES line search to check if `SNESSetJacobianDomainError()` has been called.

  Synopsis:
  #include <snesimpl.h>
  void SNESLineSearchCheckJacobian(SNES snes, SNESLineSearch ls)

  Collective

  Input Parameters:
+ snes - the `SNES` solver object
- ls   - the `SNESLineSearch` object

  Level: developer

  Notes:
  This turns the non-collective `SNESSetJacobianDomainError()` into a collective operation

  This check is done in debug mode or if `SNESSetCheckJacobianDomainError()` has been called

.seealso: [](ch_snes), `SNESSetCheckJacobianDomainError()`, `SNESSetFunctionDomainError()`, `PETSC_ERR_CONV_FAILED`, `SNESSetErrorIfNotConverged()`, `SNES_DIVERGED_FUNCTION_DOMAIN`,
          `SNESConvergedReason`, `SNES_DIVERGED_FUNCTION_NAN`
MC*/
#define SNESLineSearchCheckJacobianDomainError(snes, ls) \
  do { \
    if (snes->checkjacdomainerror) { \
      PetscBool jacobiandomainerror; \
      PetscCallMPI(MPIU_Allreduce(&snes->jacobiandomainerror, &jacobiandomainerror, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)ls))); \
      if (jacobiandomainerror) { \
        ls->reason                = SNES_LINESEARCH_FAILED_JACOBIAN_DOMAIN; \
        snes->jacobiandomainerror = PETSC_FALSE; \
        PetscCheck(!snes->errorifnotconverged, PetscObjectComm((PetscObject)ls), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due to Jacobian domain error"); \
        PetscFunctionReturn(PETSC_SUCCESS); \
      } \
    } \
  } while (0)
