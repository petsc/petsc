
#include <petsc/private/kspimpl.h>   /*I "petscksp.h" I*/

/*@
   KSPInitialResidual - Computes the residual. Either b - A*C*u = b - A*x with right
     preconditioning or C*(b - A*x) with left preconditioning; the latter
     residual is often called the "preconditioned residual".

   Collective on ksp

   Input Parameters:
+  vsoln    - solution to use in computing residual
.  vt1, vt2 - temporary work vectors
-  vb       - right-hand-side vector

   Output Parameters:
.  vres     - calculated residual

   Notes:
   This routine assumes that an iterative method, designed for
$     A x = b
   will be used with a preconditioner, C, such that the actual problem is either
$     AC u = b (right preconditioning) or
$     CA x = Cb (left preconditioning).
   This means that the calculated residual will be scaled and/or preconditioned;
   the true residual
$     b-Ax
   is returned in the vt2 temporary.

   Level: developer

.seealso: `KSPMonitor()`
@*/

PetscErrorCode  KSPInitialResidual(KSP ksp,Vec vsoln,Vec vt1,Vec vt2,Vec vres,Vec vb)
{
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(vsoln,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vres,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vb,VEC_CLASSID,6);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));
  if (!ksp->guess_zero) {
    /* skip right scaling since current guess already has it */
    PetscCall(KSP_MatMult(ksp,Amat,vsoln,vt1));
    PetscCall(VecCopy(vb,vt2));
    PetscCall(VecAXPY(vt2,-1.0,vt1));
    if (ksp->pc_side == PC_RIGHT) {
      PetscCall(PCDiagonalScaleLeft(ksp->pc,vt2,vres));
    } else if (ksp->pc_side == PC_LEFT) {
      PetscCall(KSP_PCApply(ksp,vt2,vres));
      PetscCall(PCDiagonalScaleLeft(ksp->pc,vres,vres));
    } else if (ksp->pc_side == PC_SYMMETRIC) {
      PetscCall(PCApplySymmetricLeft(ksp->pc,vt2,vres));
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP, "Invalid preconditioning side %d", (int)ksp->pc_side);
  } else {
    PetscCall(VecCopy(vb,vt2));
    if (ksp->pc_side == PC_RIGHT) {
      PetscCall(PCDiagonalScaleLeft(ksp->pc,vb,vres));
    } else if (ksp->pc_side == PC_LEFT) {
      PetscCall(KSP_PCApply(ksp,vb,vres));
      PetscCall(PCDiagonalScaleLeft(ksp->pc,vres,vres));
    } else if (ksp->pc_side == PC_SYMMETRIC) {
      PetscCall(PCApplySymmetricLeft(ksp->pc, vb, vres));
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP, "Invalid preconditioning side %d", (int)ksp->pc_side);
  }
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by the first norm computaion in the Krylov method */
  if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
    PetscCall(VecSetInf(vres));
  }
  PetscFunctionReturn(0);
}

/*@
   KSPUnwindPreconditioner - Unwinds the preconditioning in the solution. That is,
     takes solution to the preconditioned problem and gets the solution to the
     original problem from it.

   Collective on ksp

   Input Parameters:
+  ksp  - iterative context
.  vsoln - solution vector
-  vt1   - temporary work vector

   Output Parameter:
.  vsoln - contains solution on output

   Notes:
   If preconditioning either symmetrically or on the right, this routine solves
   for the correction to the unpreconditioned problem.  If preconditioning on
   the left, nothing is done.

   Level: advanced

.seealso: `KSPSetPCSide()`
@*/
PetscErrorCode  KSPUnwindPreconditioner(KSP ksp,Vec vsoln,Vec vt1)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(vsoln,VEC_CLASSID,2);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  if (ksp->pc_side == PC_RIGHT) {
    PetscCall(KSP_PCApply(ksp,vsoln,vt1));
    PetscCall(PCDiagonalScaleRight(ksp->pc,vt1,vsoln));
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    PetscCall(PCApplySymmetricRight(ksp->pc,vsoln,vt1));
    PetscCall(VecCopy(vt1,vsoln));
  } else {
    PetscCall(PCDiagonalScaleRight(ksp->pc,vsoln,vsoln));
  }
  PetscFunctionReturn(0);
}
