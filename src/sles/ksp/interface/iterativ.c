#ifndef lint
static char vcid[] = "$Id: iterativ.c,v 1.37 1996/01/09 15:10:56 curfman Exp bsmith $";
#endif

/*
   This file contains some simple default routines.  
   These routines should be SHORT, since they will be included in every
   executable image that uses the iterative routines (note that, through
   the registry system, we provide a way to load only the truely necessary
   files) 
 */
#include "kspimpl.h"   /*I "ksp.h" I*/

/*
  KSPiDefaultFreeWork - Free work vectors

  Input Parameters:
. itP  - iterative context
 */
int KSPiDefaultFreeWork( KSP itP )
{
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (itP->work)  return VecDestroyVecs(itP->work,itP->nwork);
  return 0;
}

/*
   KSPCheckDef - Checks the definition of the KSP quantities 
   necessary for most of the solvers.

   Input Parameter:
.  itP - iterative context

   Returns:
   the number of errors encountered.

.keywords: KSP, errors, check, definition
 */
int KSPCheckDef( KSP itP )
{
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (!itP->vec_sol) {
    SETERRQ(1,"KSPCheckDef:Solution vector not specified"); 
  }
  if (!itP->vec_rhs) {
    SETERRQ(2,"KSPCheckDef:RHS vector not specified"); 
  }
  if (!itP->B)   {
    SETERRQ(4,"KSPCheckDef:Preconditioner routine not specified"); 
  }
  return 0;
}

/*ARGSUSED*/
/*@C
   KSPDefaultMonitor - Default code to print the residual norm at each 
   iteration of the iterative solvers.

   Input Parameters:
.  itP   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated).  
.  dummy - unused monitor context 

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPLGMonitorCreate()
@*/
int KSPDefaultMonitor(KSP itP,int n,double rnorm,void *dummy)
{
  MPIU_printf(itP->comm,"%d KSP Residual norm %14.12e \n",n,rnorm); return 0;
}

int KSPDefaultSMonitor(KSP ksp,int its, double fnorm,void *dummy)
{
  if (fnorm > 1.e-9 || fnorm == 0.0) {
    MPIU_printf(ksp->comm, "iter = %d, Residual norm %g \n",its,fnorm);
  }
  else if (fnorm > 1.e-11){
    MPIU_printf(ksp->comm, "iter = %d, Residual norm %5.3e \n",its,fnorm);
  }
  else {
    MPIU_printf(ksp->comm, "iter = %d, Residual norm < 1.e-11\n",its);
  }
  return 0;
}

/*ARGSUSED*/
/*@C
   KSPDefaultConverged - Default code to determine convergence of
   the iterative solvers.

   Input Parameters:
.  itP   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
.  dummy - unused convergence context 

   Returns:
   1 if the iteration has converged
  -1 if residual norm exceeds divergence threshold;
   0 otherwise.

   Notes:
   KSPDefaultConverged() reaches convergence when
$        rnorm < MAX ( rtol * rnorm_0, atol );
$  Divergence is detected if
$        rnorm > dtol * rnorm_0,
$  where rtol = relative tolerance,
$        atol = absolute tolerance.
$        dtol = divergence tolerance,
$        rnorm_0 = initial residual norm

   Use KSPSetTolerances() to alter the defaults for 
   rtol, atol, dtol.

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances()
@*/
int KSPDefaultConverged(KSP itP,int n,double rnorm,void *dummy)
{
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if ( n == 0 ) {
    itP->ttol   = PetscMax(itP->rtol*rnorm,itP->atol);
    itP->rnorm0 = rnorm;
  }
  if ( rnorm <= itP->ttol )      return 1;
  if ( rnorm >= itP->divtol*itP->rnorm0 || rnorm != rnorm) return -1;
  return(0);
}

/*
   KSPDefaultBuildSolution - Default code to create/move the solution.

   Input Parameters:
.  itP - iterative context
.  v   - pointer to the user's vector  

   Output Parameter:
.  V - pointer to a vector containing the solution

.keywords:  KSP, build, solution, default

.seealso: KSPGetSolution(), KSPDefaultBuildResidual()
*/
int KSPDefaultBuildSolution(KSP itP,Vec v,Vec *V)
{
  int ierr;
  if (itP->pc_side == PC_RIGHT) {
    if (itP->B) {
      if (v) {ierr = PCApply(itP->B,itP->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,"KSPDefaultBuildSolution:Not working with right preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(itP->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = itP->vec_sol;}
    }
  }
  else if (itP->pc_side == PC_SYMMETRIC) {
    if (itP->B) {
      if (v) {ierr = PCApplySymmRight(itP->B,itP->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,"KSPDefaultBuildSolution:Not working with symmetric preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(itP->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = itP->vec_sol;}
    }
  }
  else {
    if (v) {ierr = VecCopy(itP->vec_sol,v); CHKERRQ(ierr); *V = v;}
    else { *V = itP->vec_sol; }
  }
  return 0;
}

/*
   KSPDefaultBuildResidual - Default code to compute the residual.

   Input Parameters:
.  itP - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector  

   Output Parameter:
.  V - pointer to a vector containing the residual

.keywords:  KSP, build, residual, default

.seealso: KSPDefaultBuildSolution()
*/
int KSPDefaultBuildResidual(KSP itP,Vec t,Vec v,Vec *V)
{
  int          ierr;
  MatStructure pflag;
  Vec          T;
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;

  PCGetOperators(itP->B,&Amat,&Pmat,&pflag);
  ierr = KSPBuildSolution(itP,t,&T); CHKERRQ(ierr);
  ierr = MatMult(Amat, t, v ); CHKERRQ(ierr);
  ierr = VecAYPX(&mone, itP->vec_rhs, v ); CHKERRQ(ierr);
  *V = v; return 0;
}

/*
  KSPiDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. itP  - iterative context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated 
 */
int  KSPiDefaultGetWork( KSP itP, int nw )
{
  int ierr;
  if (itP->work) KSPiDefaultFreeWork( itP );
  itP->nwork = nw;
  ierr = VecDuplicateVecs(itP->vec_rhs,nw,&itP->work); CHKERRQ(ierr);
  PLogObjectParents(itP,nw,itP->work);
  return 0;
}

/*
  KSPiDefaultAdjustWork - Adjusts work vectors.

  Input Parameters:
. itP  - iterative context
 */
int KSPiDefaultAdjustWork( KSP itP )
{
  if ( itP->adjust_work_vectors ) {
    return (itP->adjust_work_vectors)(itP, itP->work,itP->nwork); 
  }
  return 0;
}

/*
KSPiDefaultDestroy - Destroys a iterative context variable for methods with
no separate context.  Preferred calling sequence KSPDestroy().

Input Parameters: 
.   itP - the iterative context
*/
int KSPiDefaultDestroy(PetscObject obj)
{
  KSP itP = (KSP) obj;
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (itP->data) PetscFree(itP->data);

  /* free work vectors */
  KSPiDefaultFreeWork( itP );
  return 0;
}

