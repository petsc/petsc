#ifndef lint
static char vcid[] = "$Id: $";
#endif


#include "petsc.h"
#include "kspimpl.h"   /*I "ksp.h" I*/
/*@
    KSPResidual - Computes the residual without making any assumptions
    about the solution.  Uses the general iterative structure.

    Input parameters:
.    vsoln - solution to use in computing residual
.    vt1,2 - temps that are needed
.    vres  - calculated residual
.    vbinvf- the result of binv^{-1} b.  If null, don't do it
.    vb    - right-hand-side to use

    Notes:
    This routine assumes that an iterative method, designed for
$            A x = b
    will be used with a preconditioner, C, such that the actual problem is
$            M u = f    
    where M = AC (right preconditioning) or CA (left preconditioning).
 @*/
int KSPResidual(KSP itP,Vec vsoln,Vec vt1,Vec vt2,Vec vres, Vec vbinvf,Vec vb)
{
  Scalar one = -1.0;
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->right_pre) {
    if (vbinvf) VecCopy(vb,vbinvf);
    vbinvf = vb;
  }
  else {
    PCApply(itP->B,vb,vbinvf);
    itP->nbinv++;
  }
  if (!itP->guess_zero) {
    /* compute initial residual: f - M*x */
    /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
    if (itP->right_pre) {
        /* we want a * binv * b * x, or just a * x for the first step */
        /* a*x into temp */
        MatMult(PCGetMat(itP->B), vsoln, vt1 );
	itP->namult++;
    }
    else {
        /* else we do binv * a * x */
        PCApplyBAorAB(itP->B,itP->right_pre, vsoln, vt1, vt2 );
	itP->nmatop++;
    }
    /* This is an extra copy for the right-inverse case */
    VecCopy( vbinvf, vres ); 
    VecAXPY(&one, vt1, vres );
          /* inv(b)(f - a*x) into dest */
    itP->nvectors++;
  }
  else {
    VecCopy( vbinvf, vres );
  }
  return 0;
}

/*@
  KSPUnwindPre - Unwinds the preconditioning in the solution.

  Input Parameters:
. itP  - iterative context
. vsoln - solution vector 
. vt1   - temporary vector

  Output Parameter:
. vsoln - contains solution on output  
@*/
int KSPUnwindPre( KSP itP, Vec vsoln, Vec vt1 )
{
  VALIDHEADER(itP,KSP_COOKIE);
/* If we preconditioned on the right, we need to solve for the correction to
   the unpreconditioned problem.  Nothing needs to be done on the left. */
  if (itP->right_pre) {
    PCApply(itP->B, vsoln, vt1 );
    VecCopy( vt1, vsoln );
    itP->nbinv++;
  }
  return 0;
}

