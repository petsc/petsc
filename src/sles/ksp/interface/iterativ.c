#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iterativ.c,v 1.84 1999/04/19 22:14:29 bsmith Exp balay $";
#endif

/*
   This file contains some simple default routines.  
   These routines should be SHORT, since they will be included in every
   executable image that uses the iterative routines (note that, through
   the registry system, we provide a way to load only the truely necessary
   files) 
 */
#include "src/sles/ksp/kspimpl.h"   /*I "ksp.h" I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultFreeWork"
/*
  KSPDefaultFreeWork - Free work vectors

  Input Parameters:
. ksp  - iterative context
 */
int KSPDefaultFreeWork( KSP ksp )
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->work)  {
    ierr = VecDestroyVecs(ksp->work,ksp->nwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetResidualNorm"
/*@C
   KSPGetResidualNorm - Gets the last (approximate preconditioned)
   residual norm that has been computed.
 
   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  rnorm - residual norm

   Level: intermediate

.keywords: KSP, get, residual norm

.seealso: KSPComputeResidual()
@*/
int KSPGetResidualNorm(KSP ksp,double *rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *rnorm = ksp->rnorm;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetIterationNumber"
/*@C
   KSPGetIterationNumber - Gets the current iteration number (if the 
         KSPSolve() (SLESSolve()) is complete, returns the number of iterations
         used.
 
   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the ith iteration this returns i-1
.keywords: KSP, get, residual norm

.seealso: KSPComputeResidual(), KSPGetResidualNorm()
@*/
int KSPGetIterationNumber(KSP ksp,int *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *its = ksp->its;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSingularValueMonitor"
/*@C
    KSPSingularValueMonitor - Prints the two norm of the true residual and
    estimation of the extreme singular values of the preconditioned problem
    at each iteration.
 
    Collective on KSP

    Input Parameters:
+   ksp - the iterative context
.   n  - the iteration
-   rnorm - the two norm of the residual

    Options Database Key:
.   -ksp_singmonitor - Activates KSPSingularValueMonitor()

    Notes:
    The CG solver uses the Lanczos technique for eigenvalue computation, 
    while GMRES uses the Arnoldi technique; other iterative methods do
    not currently compute singular values.

    Level: intermediate

.keywords: KSP, CG, default, monitor, extreme, singular values, Lanczos, Arnoldi

.seealso: KSPComputeExtremeSingularValues()
@*/
int KSPSingularValueMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  double emin,emax,c;
  int    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!ksp->calc_sings) {
    ierr = PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e \n",n,rnorm);CHKERRQ(ierr);
  } else {
    ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);
    c = emax/emin;
    ierr = PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e %% max %g min %g max/min %g\n",n,rnorm,emax,emin,c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPVecViewMonitor"
/*@C
   KSPVecViewMonitor - Monitors progress of the KSP solvers by calling 
   VecView() for the approximate solution at each iteration.

   Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

   Notes:
    For some Krylov methods such as GMRES constructing the solution at
  each iteration is expensive, hence using this will slow the code.

.keywords: KSP, nonlinear, vector, monitor, view

.seealso: KSPSetMonitor(), KSPDefaultMonitor(), VecView()
@*/
int KSPVecViewMonitor(KSP ksp,int its,double fgnorm,void *dummy)
{
  int    ierr;
  Vec    x;
  Viewer viewer = (Viewer) dummy;

  PetscFunctionBegin;
  ierr = KSPBuildSolution(ksp,PETSC_NULL,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    viewer = VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultMonitor"
/*@C
   KSPDefaultMonitor - Print the residual norm at each iteration of an
   iterative solver.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).  
-  dummy - unused monitor context 

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPTrueMonitor(), KSPLGMonitorCreate()
@*/
int KSPDefaultMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e \n",n,rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPTrueMonitor"
/*@C
   KSPTrueMonitor - Prints the true residual norm as well as the preconditioned
   residual norm at each iteration of an iterative solver.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).  
-  dummy - unused monitor context 

   Options Database Key:
.  -ksp_truemonitor - Activates KSPTrueMonitor()

   Notes:
   When using right preconditioning, these values are equivalent.

   When using either ICC or ILU preconditioners in BlockSolve95 
   (via MATMPIROWBS matrix format), then use this monitor will
   print both the residual norm associated with the original
   (unscaled) matrix.

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPTrueMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  int          ierr;
  Vec          resid,work;
  double       scnorm;
  PC           pc;
  Mat          A, B;
  
  PetscFunctionBegin;
  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);

  /*
     Unscale the residual if the matrix is, for example, a BlockSolve matrix
    but only if both matrices are the same matrix, since only then would 
    they be scaled.
  */
  ierr = VecCopy(resid,work);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,&B,PETSC_NULL);CHKERRQ(ierr);
  if (A == B) {
    ierr = MatUnScaleSystem(A,PETSC_NULL,work);CHKERRQ(ierr);
    ierr = VecNorm(work,NORM_2,&scnorm);CHKERRQ(ierr);
    ierr = VecDestroy(work);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(ksp->comm,"%d KSP preconditioned resid norm %14.12e true resid norm %14.12e\n",n,rnorm,scnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultSMonitor"
/*
  Default (short) KSP Monitor, same as KSPDefaultMonitor() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often 
  different on different machines; by using this routine different 
  machines will usually generate the same output.
*/
int KSPDefaultSMonitor(KSP ksp,int its, double fnorm,void *dummy)
{
  int ierr;

  PetscFunctionBegin;
  if (fnorm > 1.e-9) {
    ierr = PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm %g \n",its,fnorm);CHKERRQ(ierr);
  } else if (fnorm > 1.e-11){
    ierr = PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm %5.3e \n",its,fnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm < 1.e-11\n",its);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSkipConverged"
/*@C
   KSPSkipConverged - Convergence test that NEVER returns as converged.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  dummy - unused convergence context 

   Returns:
.  0 - always

   Notes:
   This is used as the convergence test with the option KSPSetAvoidNorms(),
   since norms of the residual are not computed. Convergence is then declared 
   after a fixed number of iterations have been used. Useful when one is 
   using CG or Bi-CG-stab as a smoother.
                    
   Level: advanced

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSetAvoidNorms()
@*/
int KSPSkipConverged(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged"
/*@C
   KSPDefaultConverged - Determines convergence of
   the iterative solvers (default code).

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  dummy - unused convergence context 

   Returns:
+   1 - if the iteration has converged;
.  -1 - if residual norm exceeds divergence threshold;
-   0 - otherwise.

   Notes:
   KSPDefaultConverged() reaches convergence when
$      rnorm < MAX ( rtol * rnorm_0, atol );
   Divergence is detected if
$      rnorm > dtol * rnorm_0,

   where 
+     rtol = relative tolerance,
.     atol = absolute tolerance.
.     dtol = divergence tolerance,
-     rnorm_0 = initial residual norm

   Use KSPSetTolerances() to alter the defaults for rtol, atol, dtol.

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged()
@*/
int KSPDefaultConverged(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if ( n == 0 ) {
    ksp->ttol   = PetscMax(ksp->rtol*rnorm,ksp->atol);
    ksp->rnorm0 = rnorm;
  }
  if ( rnorm <= ksp->ttol )                                PetscFunctionReturn(1);
  if ( rnorm >= ksp->divtol*ksp->rnorm0 || rnorm != rnorm) PetscFunctionReturn(-1);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultBuildSolution"
/*
   KSPDefaultBuildSolution - Default code to create/move the solution.

   Input Parameters:
+  ksp - iterative context
-  v   - pointer to the user's vector  

   Output Parameter:
.  V - pointer to a vector containing the solution

   Level: advanced

.keywords:  KSP, build, solution, default

.seealso: KSPGetSolution(), KSPDefaultBuildResidual()
*/
int KSPDefaultBuildSolution(KSP ksp,Vec v,Vec *V)
{
  int ierr;
  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->B) {
      if (v) {ierr = PCApply(ksp->B,ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else {SETERRQ(PETSC_ERR_SUP,0,"Not working with right preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->B) {
      if (v) {ierr = PCApplySymmetricRight(ksp->B,ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else {SETERRQ(PETSC_ERR_SUP,0,"Not working with symmetric preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else {
    if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
    else { *V = ksp->vec_sol; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultBuildResidual"
/*
   KSPDefaultBuildResidual - Default code to compute the residual.

   Input Parameters:
.  ksp - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector  

   Output Parameter:
.  V - pointer to a vector containing the residual

   Level: advanced

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

  PetscFunctionBegin;
  PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);
  ierr = KSPBuildSolution(ksp,t,&T);CHKERRQ(ierr);
  ierr = MatMult(Amat, t, v );CHKERRQ(ierr);
  ierr = VecAYPX(&mone, ksp->vec_rhs, v );CHKERRQ(ierr);
  *V = v; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultGetWork"
/*
  KSPDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. ksp  - iterative context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated 
 */
int  KSPDefaultGetWork( KSP ksp, int nw )
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->work) {ierr = KSPDefaultFreeWork( ksp );CHKERRQ(ierr);}
  ksp->nwork = nw;
  ierr = VecDuplicateVecs(ksp->vec_rhs,nw,&ksp->work);CHKERRQ(ierr);
  PLogObjectParents(ksp,nw,ksp->work);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultDestroy"
/*
  KSPDefaultDestroy - Destroys a iterative context variable for methods with
  no separate context.  Preferred calling sequence KSPDestroy().

  Input Parameter: 
. ksp - the iterative context
*/
int KSPDefaultDestroy(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->data) PetscFree(ksp->data);

  /* free work vectors */
  KSPDefaultFreeWork( ksp );
  PetscFunctionReturn(0);
}

