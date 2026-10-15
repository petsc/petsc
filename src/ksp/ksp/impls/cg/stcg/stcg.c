#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP);

/*MC
   KSPSTCG -   Code to run conjugate gradient method subject to a constraint on the solution norm for use in a trust region method
   {cite}`steihaug:83`, {cite}`toint1981towards`

   Options Database Keys:
+  -ksp_cg_radius radius                                  - Trust Region Radius
-  -ksp_cg_tr_norm_type (preconditioned|unpreconditioned) - Norm used for the trust-region constraint

   Level: developer

   Notes:
   This specializes `KSPCG` by enabling `KSPSetConvergedNegativeCurvature()` and selecting the unpreconditioned residual norm by default.
   All other options and APIs are shared with `KSPCG`. It is used in trust-region methods such as `SNESNEWTONTR`.

   Use preconditioned conjugate gradient to compute an approximate minimizer of the quadratic function

   $$
   q(s) = g^T s + \frac{1}{2} s^T H s
   $$

   subject to the trust region constraint

   $$
            || s || \le \delta,
   $$

   where
.vb
     delta is the trust region radius,
     g is the gradient vector,
     H is the Hessian approximation, and
.ve

   `KSPConvergedReason` may include
+  `KSP_CONVERGED_NEG_CURVE` - if convergence is reached along a negative curvature direction,
-  `KSP_CONVERGED_STEP_LENGTH` - if convergence is reached along a constrained step,

  The preconditioner supplied should be symmetric and positive definite.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPCG`, `KSPCGSetType()`, `KSPType`, `KSP`, `KSPCGSetRadius()`, `KSPCGSetTRNormType()`, `KSPCGGetTRSolutionNorm()`, `KSPCGGetObjFcn()`, `KSPNASH`, `KSPGLTR`, `KSPQCG`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_STCG(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPCreate_CG(ksp));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetConvergedNegativeCurvature(ksp, PETSC_TRUE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ksp, KSPSTCG));
  PetscFunctionReturn(PETSC_SUCCESS);
}
