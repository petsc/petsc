#ifndef lint
static char vcid[] = "$Id: iterativ.c,v 1.39 1996/03/10 17:26:57 bsmith Exp bsmith $";
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
. ksp  - iterative context
 */
int KSPiDefaultFreeWork( KSP ksp )
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->work)  return VecDestroyVecs(ksp->work,ksp->nwork);
  return 0;
}

/*
   KSPCheckDef - Checks the definition of the KSP quantities 
   necessary for most of the solvers.

   Input Parameter:
.  ksp - iterative context

   Returns:
   the number of errors encountered.

.keywords: KSP, errors, check, definition
 */
int KSPCheckDef( KSP ksp )
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!ksp->vec_sol) {
    SETERRQ(1,"KSPCheckDef:Solution vector not specified"); 
  }
  if (!ksp->vec_rhs) {
    SETERRQ(2,"KSPCheckDef:RHS vector not specified"); 
  }
  if (!ksp->B)   {
    SETERRQ(4,"KSPCheckDef:Preconditioner routine not specified"); 
  }
  return 0;
}

/*ARGSUSED*/
/*@C
   KSPDefaultMonitor - Default code to print the residual norm at each 
   iteration of the iterative solvers.

   Input Parameters:
.  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated).  
.  dummy - unused monitor context 

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPLGMonitorCreate()
@*/
int KSPDefaultMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e \n",n,rnorm); return 0;
}

int KSPDefaultSMonitor(KSP ksp,int its, double fnorm,void *dummy)
{
  if (fnorm > 1.e-9 || fnorm == 0.0) {
    PetscPrintf(ksp->comm, "iter = %d, Residual norm %g \n",its,fnorm);
  }
  else if (fnorm > 1.e-11){
    PetscPrintf(ksp->comm, "iter = %d, Residual norm %5.3e \n",its,fnorm);
  }
  else {
    PetscPrintf(ksp->comm, "iter = %d, Residual norm < 1.e-11\n",its);
  }
  return 0;
}

/*ARGSUSED*/
/*@C
   KSPDefaultConverged - Default code to determine convergence of
   the iterative solvers.

   Input Parameters:
.  ksp   - iterative context
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
int KSPDefaultConverged(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if ( n == 0 ) {
    ksp->ttol   = PetscMax(ksp->rtol*rnorm,ksp->atol);
    ksp->rnorm0 = rnorm;
  }
  if ( rnorm <= ksp->ttol )      return 1;
  if ( rnorm >= ksp->divtol*ksp->rnorm0 || rnorm != rnorm) return -1;
  return(0);
}

/*
   KSPDefaultBuildSolution - Default code to create/move the solution.

   Input Parameters:
.  ksp - iterative context
.  v   - pointer to the user's vector  

   Output Parameter:
.  V - pointer to a vector containing the solution

.keywords:  KSP, build, solution, default

.seealso: KSPGetSolution(), KSPDefaultBuildResidual()
*/
int KSPDefaultBuildSolution(KSP ksp,Vec v,Vec *V)
{
  int ierr;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->B) {
      if (v) {ierr = PCApply(ksp->B,ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,"KSPDefaultBuildSolution:Not working with right preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->B) {
      if (v) {ierr = PCApplySymmetricRight(ksp->B,ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,"KSPDefaultBuildSolution:Not working with symmetric preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else {
    if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
    else { *V = ksp->vec_sol; }
  }
  return 0;
}

/*
   KSPDefaultBuildResidual - Default code to compute the residual.

   Input Parameters:
.  ksp - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector  

   Output Parameter:
.  V - pointer to a vector containing the residual

.keywords:  KSP, build, residual, default

.seealso: KSPDefaultBuildSolution()
*/
int KSPDefaultBuildResidual(KSP ksp,Vec t,Vec v,Vec *V)
{
  int          ierr;
  MatStructure pflag;
  Vec          T;
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;

  PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);
  ierr = KSPBuildSolution(ksp,t,&T); CHKERRQ(ierr);
  ierr = MatMult(Amat, t, v ); CHKERRQ(ierr);
  ierr = VecAYPX(&mone, ksp->vec_rhs, v ); CHKERRQ(ierr);
  *V = v; return 0;
}

/*
  KSPiDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. ksp  - iterative context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated 
 */
int  KSPiDefaultGetWork( KSP ksp, int nw )
{
  int ierr;
  if (ksp->work) KSPiDefaultFreeWork( ksp );
  ksp->nwork = nw;
  ierr = VecDuplicateVecs(ksp->vec_rhs,nw,&ksp->work); CHKERRQ(ierr);
  PLogObjectParents(ksp,nw,ksp->work);
  return 0;
}

/*
  KSPiDefaultAdjustWork - Adjusts work vectors.

  Input Parameters:
. ksp  - iterative context
 */
int KSPiDefaultAdjustWork( KSP ksp )
{
  if ( ksp->adjust_work_vectors ) {
    return (ksp->adjust_work_vectors)(ksp, ksp->work,ksp->nwork); 
  }
  return 0;
}

/*
KSPiDefaultDestroy - Destroys a iterative context variable for methods with
no separate context.  Preferred calling sequence KSPDestroy().

Input Parameters: 
.   ksp - the iterative context
*/
int KSPiDefaultDestroy(PetscObject obj)
{
  KSP ksp = (KSP) obj;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->data) PetscFree(ksp->data);

  /* free work vectors */
  KSPiDefaultFreeWork( ksp );
  return 0;
}

