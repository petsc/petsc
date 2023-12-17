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
. -ksp_cg_single_reduction <bool> - Merge inner products into single `MPI_Allreduce()`

  Level: intermediate

  Notes:
  The algorithm used in this case is described as Method 1 in {cite}`d1993conjugate`. V. Eijkhout credits the algorithm initially to Chronopoulos and Gear.

  It requires two extra work vectors than the conventional implementation in PETSc.

  See also `KSPPIPECG`, `KSPPIPECR`, and `KSPGROPPCG` that use non-blocking reductions. [](sec_pipelineksp),

.seealso: [](ch_ksp), [](sec_pipelineksp), `KSP`, `KSPCG`, `KSPGMRES`, `KSPPIPECG`, `KSPPIPECR`, `and KSPGROPPCG`
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
  KSPCGSetRadius - Sets the radius of the trust region used by the `KSPCG` when the solver is used inside `SNESNEWTONTR`

  Logically Collective

  Input Parameters:
+ ksp    - the iterative context
- radius - the trust region radius (0 is the default that disable the use of the radius)

  Level: advanced

  Note:
  When radius is greater then 0, the Steihaugh-Toint trick is used

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
  KSPCGSetObjectiveTarget - Sets the target value for the quadratic model reduction when the solver is used inside `SNESNEWTONTR`

  Logically Collective

  Input Parameters:
+ ksp - the iterative context
- obj - the objective value (0 is the default)

  Level: advanced

  Note:
  The `KSPSolve()` will stop when the current objective function
  $ 1/2 x_k * A * x_k - b * x_k $ is smaller than `obj` if `obj` is negative.
  Otherwise the test is ignored.

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
  KSPCGGetNormD - Got norm of the direction when the solver is used inside `SNESNEWTONTR`

  Collective

  Input Parameters:
+ ksp    - the iterative context
- norm_d - the norm of the direction

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `SNESNEWTONTR`
@*/
PetscErrorCode KSPCGGetNormD(KSP ksp, PetscReal *norm_d)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPCGGetNormD_C", (KSP, PetscReal *), (ksp, norm_d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCGGetObjFcn - Get objective function value when the solver is used inside `SNESNEWTONTR`

  Collective

  Input Parameters:
+ ksp   - the iterative context
- o_fcn - the objective function value

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `KSPCG`, `KSPNASH`, `KSPSTCG`, `KSPGLTR`, `SNESNEWTONTR`
@*/
PetscErrorCode KSPCGGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPCGGetObjFcn_C", (KSP, PetscReal *), (ksp, o_fcn));
  PetscFunctionReturn(PETSC_SUCCESS);
}
