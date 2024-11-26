#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_PREONLY(KSP ksp)
{
  PetscReal      norm;
  PetscBool      flg;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &flg));
  PetscCheck(!flg, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  if (!ksp->guess_zero) {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)ksp->pc, &flg, PCREDISTRIBUTE, PCMPI, ""));
    PetscCheck(flg, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "KSP of type preonly (application of preconditioner only) doesn't make sense with nonzero initial guess you probably want a KSP of type Richardson");
  }
  ksp->its = 0;
  if (ksp->numbermonitors) {
    PetscCall(VecNorm(ksp->vec_rhs, NORM_2, &norm));
    PetscCall(KSPMonitor(ksp, 0, norm));
  }
  PetscCall(KSP_PCApply(ksp, ksp->vec_rhs, ksp->vec_sol));
  PetscCall(PCReduceFailedReason(ksp->pc));
  PetscCall(PCGetFailedReason(ksp->pc, &pcreason));
  PetscCall(VecFlag(ksp->vec_sol, pcreason));
  if (pcreason) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged with PCFailedReason %s", PCFailedReasons[pcreason]);
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
  }
  if (ksp->numbermonitors) {
    Vec v;
    Mat A;

    PetscCall(VecDuplicate(ksp->vec_rhs, &v));
    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    PetscCall(KSP_MatMult(ksp, A, ksp->vec_sol, v));
    PetscCall(VecAYPX(v, -1.0, ksp->vec_rhs));
    PetscCall(VecNorm(v, NORM_2, &norm));
    PetscCall(VecDestroy(&v));
    PetscCall(KSPMonitor(ksp, 1, norm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMatSolve_PREONLY(KSP ksp, Mat B, Mat X)
{
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  PetscCheck(ksp->guess_zero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Running KSP of preonly doesn't make sense with nonzero initial guess you probably want a KSP type of Richardson");
  ksp->its = 0;
  PetscCall(KSP_PCMatApply(ksp, B, X));
  PetscCall(PCGetFailedReason(ksp->pc, &pcreason));
  /* Note: only some ranks may have this set; this may lead to problems if the caller assumes ksp->reason is set on all processes or just uses the result */
  if (pcreason) {
    PetscCall(MatSetInf(X));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   KSPNONE - An alias for `KSPPREONLY`

   Options Database Key:
.   -ksp_type none - use a single application of the preconditioner only

   Level: beginner

   Note:
   See `KSPPREONLY` for more details

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSPPREONLY`, `KSP`, `KSPRICHARDSON`, `KSPCHEBYSHEV`, `KSPGetPC()`, `KSPSetInitialGuessNonzero()`,
          `PCREDISTRIBUTE`, `PCRedistributeGetKSP()`, `KSPPREONLY`
M*/

/*MC
   KSPPREONLY - This implements a method that applies ONLY the preconditioner exactly once.

   It is commonly used with the direct solver preconditioners like `PCLU` and `PCCHOLESKY`, but it may also be used when a single iteration of the
   preconditioner is needed for smoothing in multigrid, `PCMG` or `PCGAMG` or within some other nested linear solve such as `PCFIELDSPLIT` or `PCBJACOBI`.

   There is an alias of this with the name `KSPNONE`.

   Options Database Key:
.   -ksp_type preonly - use a single application of the preconditioner only

   Level: beginner

   Notes:
   Since this does not involve an iteration the basic `KSP` parameters such as tolerances and maximum iteration counts
   do not apply

   To apply the preconditioner multiple times in a simple iteration use `KSPRICHARDSON`

   This `KSPType` cannot be used with the flag `-ksp_initial_guess_nonzero` or the call `KSPSetInitialGuessNonzero()` since it simply applies
   the preconditioner to the given right-hand side during `KSPSolve()`. Except when the
   `PCType` is `PCREDISTRIBUTE`; in that situation pass the nonzero initial guess flag with `-ksp_initial_guess_nonzero` or `KSPSetInitialGuessNonzero()`
   both to the outer `KSP` (which is `KSPPREONLY`) and the inner `KSP` object obtained with `KSPGetPC()` followed by `PCRedistributedGetKSP()`
   followed by `KSPSetInitialGuessNonzero()` or the option  `-redistribute_ksp_initial_guess_nonzero`.

   Developer Note:
   Even though this method does not use any norms, the user is allowed to set the `KSPNormType` to any value.
   This is so the users does not have to change `KSPNormType` options when they switch from other `KSP` methods to this one.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPRICHARDSON`, `KSPCHEBYSHEV`, `KSPGetPC()`, `KSPSetInitialGuessNonzero()`,
          `PCREDISTRIBUTE`, `PCRedistributeGetKSP()`, `KSPNONE`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NATURAL, PC_LEFT, 2));

  ksp->data                = NULL;
  ksp->ops->setup          = KSPSetUp_PREONLY;
  ksp->ops->solve          = KSPSolve_PREONLY;
  ksp->ops->matsolve       = KSPMatSolve_PREONLY;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  ksp->guess_not_read      = PETSC_TRUE; // A KSP of preonly never needs to zero the input x since PC do not use an initial guess
  PetscFunctionReturn(PETSC_SUCCESS);
}
