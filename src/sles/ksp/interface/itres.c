#ifndef lint
static char vcid[] = "$Id: itres.c,v 1.11 1995/05/12 21:28:28 bsmith Exp bsmith $";
#endif

#include "kspimpl.h"   /*I "ksp.h" I*/
/*@
   KSPResidual - Computes the residual without making any assumptions
   about the solution.  Uses the general iterative structure.

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

  VALIDHEADER(itP,KSP_COOKIE);
  PCGetOperators(itP->B,&Amat,&Pmat,&pflag);
  if (itP->right_pre) {
    if (vbinvf) VecCopy(vb,vbinvf);
    vbinvf = vb;
  }
  else {
    PCApply(itP->B,vb,vbinvf);
  }
  if (!itP->guess_zero) {
    /* compute initial residual: f - M*x */
    /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
    if (itP->right_pre) {
        /* we want a * binv * b * x, or just a * x for the first step */
        /* a*x into temp */
        MatMult(Amat, vsoln, vt1 );
    }
    else {
        /* else we do binv * a * x */
        PCApplyBAorAB(itP->B,itP->right_pre, vsoln, vt1, vt2 );
    }
    /* This is an extra copy for the right-inverse case */
    VecCopy( vbinvf, vres ); 
    VecAXPY(&one, vt1, vres );
          /* inv(b)(f - a*x) into dest */
  }
  else {
    VecCopy( vbinvf, vres );
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
   If preconditioning on the right, this routine solves for the 
   correction to the unpreconditioned problem.  If preconditioning
   on the left, nothing is done.

.keywords: KSP, unwind, preconditioner

.seealso: KSPSetRightPreconditioner()
@*/
int KSPUnwindPre( KSP itP, Vec vsoln, Vec vt1 )
{
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->right_pre) {
    PCApply(itP->B, vsoln, vt1 );
    VecCopy( vt1, vsoln );
  }
  return 0;
}

