#ifndef lint
static char vcid[] = "$Id: itres.c,v 1.15 1995/11/04 23:28:36 bsmith Exp curfman $";
#endif

#include "kspimpl.h"   /*I "ksp.h" I*/
/*@
   KSPResidual - Computes the residual without making any assumptions
   about the solution.

   Input Parameters:
.  vsoln    - solution to use in computing residual
.  vt1, vt2 - temporary work vectors
.  vres     - calculated residual
.  vbinvf   - the result of binv^{-1} b.  If null, don't do it.
.  vb       - right-hand-side vector

   Notes:
   This routine assumes that an iterative method, designed for
$     A x = b
   will be used with a preconditioner, C, such that the actual problem is
$     M u = f    
   where M = AC (right preconditioning) or CA (left preconditioning).

.keywords: KSP, residual
@*/
int KSPResidual(KSP itP,Vec vsoln,Vec vt1,Vec vt2,Vec vres, Vec vbinvf,Vec vb)
{
  Scalar        one = -1.0;
  MatStructure  pflag;
  Mat           Amat, Pmat;
  int           ierr;

  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  PCGetOperators(itP->B,&Amat,&Pmat,&pflag);
  if (itP->pc_side == KSP_SYMMETRIC_PC)
    SETERRQ(1,"KSPResidual: KSP_SYMMETRIC_PC not yet supported.");
  if (itP->pc_side == KSP_RIGHT_PC) {
    if (vbinvf) {ierr = VecCopy(vb,vbinvf); CHKERRQ(ierr);}
    vbinvf = vb;
  }
  else {
    ierr = PCApply(itP->B,vb,vbinvf); CHKERRQ(ierr);
  }
  if (!itP->guess_zero) {
    /* compute initial residual: f - M*x */
    /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
    if (itP->pc_side == KSP_RIGHT_PC) {
      /* we want a * binv * b * x, or just a * x for the first step */
      /* a*x into temp */
      ierr = MatMult(Amat,vsoln,vt1); CHKERRQ(ierr);
    }
    else {
      /* else we do binv * a * x */
      ierr = PCApplyBAorAB(itP->B,itP->pc_side, vsoln, vt1, vt2 ); CHKERRQ(ierr);
    }
    /* This is an extra copy for the right-inverse case */
    ierr = VecCopy( vbinvf, vres ); CHKERRQ(ierr);
    ierr = VecAXPY(&one, vt1, vres ); CHKERRQ(ierr);
          /* inv(b)(f - a*x) into dest */
  }
  else {
    ierr = VecCopy( vbinvf, vres ); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   KSPUnwindPre - Unwinds the preconditioning in the solution.

   Input Parameters:
.  itP  - iterative context
.  vsoln - solution vector 
.  vt1   - temporary work vector

   Output Parameter:
.  vsoln - contains solution on output  

   Notes:
   If preconditioning either symmetrically or on the right, this routine solves
   for the correction to the unpreconditioned problem.  If preconditioning on
   the left, nothing is done.

.keywords: KSP, unwind, preconditioner

.seealso: KSPSetPreconditionerSide()
@*/
int KSPUnwindPre(KSP itP,Vec vsoln,Vec vt1)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (itP->pc_side == KSP_RIGHT_PC) {
    ierr = PCApply(itP->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  }
  else if (itP->pc_side == KSP_SYMMETRIC_PC) {
    ierr = PCApplySymmRight(itP->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  }
  return 0;
}

