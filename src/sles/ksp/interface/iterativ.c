#ifndef lint
static char vcid[] = "$Id: iterativ.c,v 1.1 1994/10/01 20:02:39 bsmith Exp $";
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
  VALIDHEADER(itP,KSP_COOKIE);
  return VecFreeVecs(itP->work,itP->nwork);
}

/*@
   KSPCheckDef - Checks the definition of the KSP quantities 
   necessary for most of the solvers.

  Input Parameter:
. itP - iterative context

   Returns:
   the number of errors encountered.
 @*/
int KSPCheckDef( KSP itP )
{
  int err = 0;
  VALIDHEADER(itP,KSP_COOKIE);
  if (!itP->vec_sol) {
    SETERR(1,"Solution vector not specified for iterative method"); 
  }
  if (!itP->vec_rhs) {
    SETERR(2,"RHS vector not specified for iterative method"); 
  }
  if (!itP->amult)   {
    SETERR(3,"Matrix-vector product routine not specified"); 
  }
  return 0;
}

/*ARGSUSED*/
/*@C
  KSPDefaultMonitor - Default code to print residual at each iteration 
  in the iterative solvers.

  Input Parameters:
. itP   - iterative context
. n     - iteration number
. rnorm - 2-norm residual value (may be estimated).  
. dummy - unused monitor context 
 @*/
int KSPDefaultMonitor(KSP itP,int n,double rnorm,void *dummy)
{
  printf("%d %14.12e \n",n,rnorm); return 0;
}

/*ARGSUSED*/
/*@
  KSPDefaultConverged - Default code to determine convergence in
  the iterative solvers.

  Input Parameters:
. itP   - iterative context
. n     - iteration number
. rnorm - 2-norm residual value (may be estimated).  
. dummy - unused converged context 

  Returns:
  1 if the iteration has converged or exceeds divergence threshold, 
  0 otherwise.
  
 @*/
int KSPDefaultConverged(KSP itP,int n,double rnorm,void *dummy)
{
  VALIDHEADER(itP,KSP_COOKIE);
  if ( n == 0 ) {
    itP->ttol   = MAX(itP->rtol*rnorm,itP->atol);
    itP->rnorm0 = rnorm;
  }
  if ( rnorm <= itP->ttol )      return 1;
  if ( rnorm >= itP->divtol*itP->rnorm0 || rnorm != rnorm) return -1;
  return(0);
}

/*@
  KSPDefaultBuildSolution - Default code to create/move the solution.

  Input Parameters:
. itP  - iterative context
. v    - pointer to user vector  

  Returns:
  pointer to a vector containing the solution.
 @*/
int KSPDefaultBuildSolution(KSP itP,Vec v,Vec *V)
{
  int ierr;
  if (itP->right_pre) {
    if (itP->binv) { PRE(itP, itP->vec_sol, v );}
    else           {ierr = VecCopy(itP->vec_sol, v ); CHKERR(ierr);}
  }
  else {ierr = VecCopy(itP->vec_sol, v ); CHKERR(ierr);}
  *V = v; return 0;
}

/*@
  KSPDefaultBuildResidual - Default code to compute the residual.

  Input Parameters:
. itP  - iterative context
. t    - pointer to temporay vector
. v    - pointer to user vector  

  Returns:
  pointer to a vector containing the residual.
 @*/
int KSPDefaultBuildResidual(KSP itP,Vec t,Vec v,Vec *V)
{
  int    ierr;
  Vec    T;
  Scalar mone = -1.0;
  ierr = KSPBuildSolution(itP,t,&T); CHKERR(ierr);
  ierr = MM(itP, t, v ); CHKERR(ierr);
  ierr = VecAYPX(&mone, itP->vec_rhs, v ); CHKERR(ierr);
  *V = v; return 0;
}

/*
  KSPiDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. itP  - iterative context
. nw   - number of work vectors to allocate

  Note:
  Call this only if no work vectors have been allocated 
 */
int  KSPiDefaultGetWork( KSP itP, int nw )
{
  if (itP->work) KSPiDefaultFreeWork( itP );
  itP->nwork = nw;
  return VecGetVecs(itP->vec_rhs,nw,&itP->work);
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
}

/*
KSPiDefaultDestroy - Destroys a iterative context variable for methods with
no separate context.  Preferred calling sequence KSPDestroy().

Input Parameters: 
.   itP - the iterative context
*/
int KSPiDefaultDestroy(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->MethodPrivate) FREE(itP->MethodPrivate);

  /* free work vectors */
  KSPiDefaultFreeWork( itP );

  /* free the context variables */
  FREE(itP);
  return 0;
}

/*@
  KSPGetWorkCounts - Gets the counts of the number of operations
  performed by the iterative routines.

  Input Parameter:
. itP - Iterative context

  Output Parameters:
. matop - number of applications of matrix-vector product AND preconditioner
. amult - number of applications of matrix-vector product not counted in matop
. binv  - number of applications of preconditioner not counted in matop
. vecs  - number of operations on vectors
. scalars - number of operations on scalars

  Note:
  The counts provided by this routine depend on correctly counting the
  number of operations in the iterative methods; those are not currently
  very accurate.  Users are encouraged to look at the source codes and
  send fixes to gropp@mcs.anl.gov .

  This routine does NOT clear the values; use KSPClearWorkCounts to do that.
@*/
int KSPGetWorkCounts( KSP itP, int *matop, int *amult, int *binv, int *vecs,
                      int * scalars )
{
  VALIDHEADER(itP,KSP_COOKIE);
  *matop   = itP->nmatop;
  *amult   = itP->namult;
  *binv    = itP->nbinv;
  *vecs    = itP->nvectors;
  *scalars = itP->nscalar;
  return 0;
}
/*@
  KSPClearWorkCounts - Clears the work counts that are maintained for the
  iterative solvers.

  Input Parameter:
. itP - Iterative context
@*/
int KSPClearWorkCounts( KSP itP )
{
  VALIDHEADER(itP,KSP_COOKIE);
  itP->nmatop        = 0;
  itP->namult        = 0;
  itP->nbinv         = 0;
  itP->nvectors      = 0;
  itP->nscalar       = 0;
  return 0;
}
