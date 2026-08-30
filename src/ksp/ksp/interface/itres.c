#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@
  KSPInitialResidual - Computes the residual. Either b - A*C*u = b - A*x with right
  preconditioning or C*(b - A*x) with left preconditioning; the latter
  residual is often called the "preconditioned residual".

  Collective

  Input Parameters:
+ ksp   - the `KSP` solver object
. vsoln - solution to use in computing residual
. vt1   - temporary work vector
. vt2   - temporary work vector
- vb    - right-hand-side vector

  Output Parameter:
. vres - calculated residual

  Level: developer

  Note:
  This routine assumes that an iterative method, designed for $ A x = b $
  will be used with a preconditioner, C, such that the actual problem is either
.vb
  AC u = b (right preconditioning) or
  CA x = Cb (left preconditioning).
.ve
  This means that the calculated residual will be preconditioned;
  the true residual $ b-Ax $
  is returned in the `vt2` temporary work vector.

.seealso: [](ch_ksp), `KSP`, `KSPSolve()`, `KSPMonitor()`
@*/
PetscErrorCode KSPInitialResidual(KSP ksp, Vec vsoln, Vec vt1, Vec vt2, Vec vres, Vec vb)
{
  Mat Amat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(vsoln, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(vt1, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(vt2, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(vres, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(vb, VEC_CLASSID, 6);
  PetscCall(PCGetOperators(ksp->pc, &Amat, NULL));
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, vsoln, vt1));
    PetscCall(VecWAXPY(vt2, -1.0, vt1, vb));
  } else PetscCall(VecCopy(vb, vt2));
  if (ksp->pc_side == PC_RIGHT) {
    PetscCall(VecCopy(vt2, vres));
  } else if (ksp->pc_side == PC_LEFT) {
    PetscCall(KSP_PCApply(ksp, vt2, vres));
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    PetscCall(PCApplySymmetricLeft(ksp->pc, vt2, vres));
  } else SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Invalid preconditioning side %d", (int)ksp->pc_side);
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by the first norm computation in the Krylov method */
  PetscCall(VecFlag(vres, ksp->reason == KSP_DIVERGED_PC_FAILED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPUnwindPreconditioner - Unwinds the preconditioning in the solution. That is,
  takes solution to the preconditioned problem and gets the solution to the
  original problem from it.

  Collective

  Input Parameters:
+ ksp   - iterative context
. vsoln - solution vector
- vt1   - temporary work vector

  Output Parameter:
. vsoln - contains solution on output

  Level: advanced

  Note:
  If preconditioning either symmetrically or on the right, this routine solves
  for the correction to the unpreconditioned problem.  If preconditioning on
  the left, nothing is done.

.seealso: [](ch_ksp), `KSP`, `KSPSetPCSide()`
@*/
PetscErrorCode KSPUnwindPreconditioner(KSP ksp, Vec vsoln, Vec vt1)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(vsoln, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(vt1, VEC_CLASSID, 3);
  if (ksp->pc_side == PC_RIGHT) {
    PetscCall(KSP_PCApply(ksp, vsoln, vt1));
    PetscCall(VecCopy(vt1, vsoln));
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    PetscCall(PCApplySymmetricRight(ksp->pc, vsoln, vt1));
    PetscCall(VecCopy(vt1, vsoln));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
