#include <../src/ksp/ksp/impls/cg/cgimpl.h> /*I "petscksp.h" I*/

/*@
  KSPCGSetType - Sets the variant of the conjugate gradient method to
  use for solving a linear system with a complex coefficient matrix.
  This option is irrelevant when solving a real system.

  Logically Collective

  Input Parameters:
+ ksp  - the iterative context
- type - the variant of CG to use, one of
.vb
      KSP_CG_HERMITIAN - complex, Hermitian matrix (default)
      KSP_CG_SYMMETRIC - complex, symmetric matrix
.ve

  Options Database Keys:
+ -ksp_cg_type hermitian - Indicates Hermitian matrix
- -ksp_cg_type symmetric - Indicates symmetric matrix

  Level: intermediate

  Note:
  By default, the matrix is assumed to be complex, Hermitian.

.seealso: [](ch_ksp), `KSP`, `KSPCG`
@*/
PetscErrorCode KSPCGSetType(KSP ksp, KSPCGType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPCGSetType_C", (KSP, KSPCGType), (ksp, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGUseSingleReduction - Merge the two inner products needed in `KSPCG` into a single `MPI_Allreduce()` call.

  Logically Collective

  Input Parameters:
+ ksp - the iterative context
- flg - turn on or off the single reduction

  Options Database Key:
. -ksp_cg_single_reduction (true|false) - Merge inner products into single `MPI_Allreduce()`

  Level: intermediate

  Notes:
  The algorithm used in this case is described as Method 1 in {cite}`d1993conjugate`. V. Eijkhout credits the algorithm initially to Chronopoulos and Gear.

  It requires two extra work vectors than the conventional implementation in PETSc.

  An active trust-region radius, a preconditioned trust-region norm, a negative objective target, or convergence at negative curvature
  is unsupported and raises an error during `KSPSolve()` when this variant is enabled.
  This variant does not compute the objective returned by `KSPCGGetObjFcn()` or the solution norm returned by `KSPCGGetTRSolutionNorm()`.

  See also `KSPPIPECG`, `KSPPIPECR`, and `KSPGROPPCG` that use non-blocking reductions. [](sec_pipelineksp),

.seealso: [](ch_ksp), [](sec_pipelineksp), `KSP`, `KSPCG`, `KSPGMRES`, `KSPPIPECG`, `KSPPIPECR`, `KSPGROPPCG`
@*/
PetscErrorCode KSPCGUseSingleReduction(KSP ksp, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, flg, 2);
  PetscTryMethod(ksp, "KSPCGUseSingleReduction_C", (KSP, PetscBool), (ksp, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGSetTRNormType - Sets the norm used for the trust-region constraint

  Logically Collective

  Input Parameters:
+ ksp      - the iterative context
- normtype - the trust-region norm type

  Options Database Key:
. -ksp_cg_tr_norm_type (preconditioned|unpreconditioned) - select the trust-region norm

  Level: advanced

  Notes:
  Supported by `KSPCG`, `KSPSTCG`, `KSPNASH`, and `KSPGLTR`. The default is `KSP_CG_TR_NORM_UNPRECONDITIONED`.
  This choice does not change the residual norm selected by `KSPSetNormType()`.

  If the preconditioner applies $M^{-1}$, `KSP_CG_TR_NORM_PRECONDITIONED` uses $\sqrt{x^H M x}$ for the solution vector $x$.
  The preconditioner must be symmetric positive definite. `KSP_CG_TR_NORM_UNPRECONDITIONED` uses
  the Euclidean norm.

  `KSPCG` and `KSPSTCG` support nonzero initial guesses only with the Euclidean norm.

.seealso: [](ch_ksp), `KSP`, `KSPCGTRNormType`, `KSPCGGetTRNormType()`, `KSPCGSetRadius()`, `KSPSetNormType()`
@*/
PetscErrorCode KSPCGSetTRNormType(KSP ksp, KSPCGTRNormType normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, normtype, 2);
  PetscTryMethod(ksp, "KSPCGSetTRNormType_C", (KSP, KSPCGTRNormType), (ksp, normtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGGetTRNormType - Gets the norm used for the trust-region constraint

  Not Collective

  Input Parameter:
. ksp - the iterative context

  Output Parameter:
. normtype - the trust-region norm type

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `KSPCGTRNormType`, `KSPCGSetTRNormType()`, `KSPCGSetRadius()`
@*/
PetscErrorCode KSPCGGetTRNormType(KSP ksp, KSPCGTRNormType *normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(normtype, 2);
  PetscUseMethod(ksp, "KSPCGGetTRNormType_C", (KSP, KSPCGTRNormType *), (ksp, normtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGSetRadius - Sets the trust-region radius for a conjugate gradient solver

  Logically Collective

  Input Parameters:
+ ksp    - the iterative context
- radius - the nonnegative trust-region radius (0, the default, disables the constraint)

  Options Database Key:
. -ksp_cg_radius radius - trust-region radius

  Level: advanced

  Notes:
  Supported by `KSPCG`, `KSPSTCG`, `KSPNASH`, and `KSPGLTR`.
  `KSPCG` and `KSPSTCG` truncate steps at the boundary using the norm selected by `KSPCGSetTRNormType()`.
  An active radius also causes them to take a boundary step at the first nonpositive curvature direction.
  The convergence reason is `KSP_CONVERGED_STEP_LENGTH`, or `KSP_CONVERGED_NEG_CURVE` if `KSPSetConvergedNegativeCurvature()` is enabled.
  A nonzero initial guess must satisfy the constraint.

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `SNESNEWTONTR`
@*/
PetscErrorCode KSPCGSetRadius(KSP ksp, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, radius, 2);
  PetscTryMethod(ksp, "KSPCGSetRadius_C", (KSP, PetscReal), (ksp, radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGSetObjectiveTarget - Sets the target value for the CG quadratic model

  Logically Collective

  Input Parameters:
+ ksp - the iterative context
- obj - the objective value (0 is the default)

  Level: advanced

  Notes:
  The `KSPSolve()` will stop when the current value of the objective function $\frac{1}{2} x^H_k A x_k - b^H x_k$
  is smaller than `obj` if `obj` is negative. Otherwise the test is ignored.
  This is used for example inside `SNESNEWTONTR`.

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `SNESNEWTONTR`
@*/
PetscErrorCode KSPCGSetObjectiveTarget(KSP ksp, PetscReal obj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, obj, 2);
  PetscTryMethod(ksp, "KSPCGSetObjectiveTarget_C", (KSP, PetscReal), (ksp, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGGetTRSolutionNorm - Gets the solution norm used for the trust-region constraint

  Not Collective

  Input Parameter:
. ksp - the iterative context

  Output Parameter:
. norm - the solution norm

  Level: advanced

  Notes:
  The solution norm is computed using the norm type selected by `KSPCGSetTRNormType()` and is available after `KSPSolve()`.
  For `KSPCG` and `KSPSTCG` with an active radius, it is also updated before each monitor call.
  The single-reduction variant does not provide this quantity.

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `KSPCGSetTRNormType()`, `SNESNEWTONTR`
@*/
PetscErrorCode KSPCGGetTRSolutionNorm(KSP ksp, PetscReal *norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(norm, 2);
  PetscUseMethod(ksp, "KSPCGGetTRSolutionNorm_C", (KSP, PetscReal *), (ksp, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGGetObjFcn - Get the conjugate gradient objective function value

  Not collective

  Input Parameters:
+ ksp   - the iterative context
- o_fcn - the objective function value

  Level: advanced

  Note:
  This function will return the current objective function value $\frac{1}{2} x^H_k A x_k - b^H x_k$.
  if called during `KSPSolve()` (e.g. during a monitor call).
  When called outside of a `KSPSolve()`, it will return the last computed value inside the solver.

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `KSPMonitorSet`
@*/
PetscErrorCode KSPCGGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(o_fcn, 2);
  PetscUseMethod(ksp, "KSPCGGetObjFcn_C", (KSP, PetscReal *), (ksp, o_fcn));
  PetscFunctionReturn(PETSC_SUCCESS);
}
