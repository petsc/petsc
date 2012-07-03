
#include <petsc-private/kspimpl.h>   /*I "petscksp.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPInitialResidual"
/*@
   KSPInitialResidual - Computes the residual. Either b - A*C*u = b - A*x with right
     preconditioning or C*(b - A*x) with left preconditioning; that later
     residual is often called the "preconditioned residual".

   Collective on KSP

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

.keywords: KSP, residual

.seealso:  KSPMonitor()
@*/

PetscErrorCode  KSPInitialResidual(KSP ksp,Vec vsoln,Vec vt1,Vec vt2,Vec vres,Vec vb)
{
  MatStructure   pflag;
  Mat            Amat,Pmat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(vsoln,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vres,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vb,VEC_CLASSID,6);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    /* skip right scaling since current guess already has it */
    ierr = KSP_MatMult(ksp,Amat,vsoln,vt1);CHKERRQ(ierr);
    ierr = VecCopy(vb,vt2);CHKERRQ(ierr);
    ierr = VecAXPY(vt2,-1.0,vt1);CHKERRQ(ierr);
    ierr = (ksp->pc_side == PC_RIGHT)?(VecCopy(vt2,vres)):(KSP_PCApply(ksp,vt2,vres));CHKERRQ(ierr);
    ierr = PCDiagonalScaleLeft(ksp->pc,vres,vres);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(vb,vt2);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT) {
      ierr = PCDiagonalScaleLeft(ksp->pc,vb,vres);CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_LEFT) {
      ierr = KSP_PCApply(ksp,vb,vres);CHKERRQ(ierr);
      ierr = PCDiagonalScaleLeft(ksp->pc,vres,vres);CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_SYMMETRIC) {
      ierr = PCApplySymmetricLeft(ksp->pc, vb, vres);CHKERRQ(ierr);
    } else SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "Invalid preconditioning side %d", (int)ksp->pc_side);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPUnwindPreconditioner"
/*@
   KSPUnwindPreconditioner - Unwinds the preconditioning in the solution. That is,
     takes solution to the preconditioned problem and gets the solution to the 
     original problem from it.

   Collective on KSP

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

.keywords: KSP, unwind, preconditioner

.seealso: KSPSetPCSide()
@*/
PetscErrorCode  KSPUnwindPreconditioner(KSP ksp,Vec vsoln,Vec vt1)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(vsoln,VEC_CLASSID,2);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  if (ksp->pc_side == PC_RIGHT) {
    ierr = KSP_PCApply(ksp,vsoln,vt1);CHKERRQ(ierr);
    ierr = PCDiagonalScaleRight(ksp->pc,vt1,vsoln);CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    ierr = PCApplySymmetricRight(ksp->pc,vsoln,vt1);CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln);CHKERRQ(ierr);
  } else {
    ierr = PCDiagonalScaleRight(ksp->pc,vsoln,vsoln);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
