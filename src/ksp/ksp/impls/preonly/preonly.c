
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_PREONLY(KSP ksp)
{
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  PetscCheck(ksp->guess_zero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  ksp->its = 0;
  PetscCall(KSP_PCApply(ksp, ksp->vec_rhs, ksp->vec_sol));
  PetscCall(PCGetFailedReasonRank(ksp->pc, &pcreason));
  /* Note: only some ranks may have this set; this may lead to problems if the caller assumes ksp->reason is set on all processes or just uses the result */
  if (pcreason) {
    PetscCall(VecSetInf(ksp->vec_sol));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
  }
  if (ksp->numbermonitors) {
    Vec       v;
    PetscReal norm;
    Mat       A;

    PetscCall(VecNorm(ksp->vec_rhs, NORM_2, &norm));
    PetscCall(KSPMonitor(ksp, 0, norm));
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
  PetscCheck(ksp->guess_zero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_USER, "Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
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
     KSPPREONLY - This implements a method that applies ONLY the preconditioner exactly once.
                  This may be used in inner iterations, where it is desired to
                  allow multiple iterations as well as the "0-iteration" case. It is
                  commonly used with the direct solver preconditioners like `PCLU` and `PCCHOLESKY`.
                  There is an alias of `KSPNONE`.

   Options Database Key:
.   -ksp_type preonly - use preconditioner only

   Level: beginner

   Notes:
   Since this does not involve an iteration the basic `KSP` parameters such as tolerances and iteration counts
   do not apply

   To apply multiple preconditioners in a simple iteration use `KSPRICHARDSON`

   Developer Note:
   Even though this method does not use any norms, the user is allowed to set the `KSPNormType` to any value.
   This is so the users does not have to change `KSPNormType` options when they switch from other `KSP` methods to this one.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPRICHARDSON`, `KSPCHEBYSHEV`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
