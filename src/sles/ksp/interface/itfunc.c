#ifndef lint
static char vcid[] = "$Id: itfunc.c,v 1.12 1995/04/13 15:18:02 curfman Exp bsmith $";
#endif

#include "petsc.h"
#include "kspimpl.h"   /*I "ksp.h" I*/

/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Input Parameter:
.  itP   - iterative context obtained from KSPCreate()

   Keywords:  KSP, setup
@*/
int KSPSetUp(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->setupcalled) return 0;
  if (itP->type == -1) {
    SETERR(1,"Method must be set before calling KSPSetUp");
  }
  itP->setupcalled = 1;
  return (*(itP)->setup)(itP);
}
/*@
   KSPSolve - Solves linear system; call it after calling 
   KSPCreate(), KSPSetup(), and KSPSet*().

   Input Parameter:
.  itP - Iterative context obtained from KSPCreate()

   Output Parameter:
.  its - number of iterations required

   Notes:
   If the number of iterations (its) is negative, the iterations were 
   aborted by the convergence tester.  If the default convergence test 
   is used, this happens when the residual grows to more than 10000 
   times the initial residual.

   Keywords:  KSP, solve, linear system
@*/
int KSPSolve(KSP itP, int *its) 
{
  int    ierr;
  Scalar zero = 0.0;
  VALIDHEADER(itP,KSP_COOKIE);
  if (!itP->setupcalled){ ierr = KSPSetUp(itP); CHKERR(ierr);}
  if (itP->guess_zero) { VecSet(&zero,itP->vec_sol);}
  return (*(itP)->solver)(itP,its);
}

/*@
   KSPDestroy - Destroys KSP context created with KSPCreate().

   Input Parameter:
.  itP   - iterative context obtained from KSPCreate()

   Keywords:  KSP, destroy
@*/
int KSPDestroy(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  return (*(itP)->destroy)((PetscObject)itP);
}

/*@
   KSPSetIterations - Sets the maximum number of iterations to use.

   Input Parameters:
.  itP  - iterative context obtained from KSPCreate()
.  maxits - maximum number of iterations to use

   Options Database Key:
$  -kspmax_it  maxits

   Keywords:  KSP, maximum, iterations
@*/
int KSPSetIterations(KSP itP, int maxits)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->max_it = maxits;
  return 0;
}

/*@
    KSPSetRightPreconditioner - Sets a flag so that right preconditioning
    is used.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

    Note:
    Left preconditioning is used by default.  Symmetric preconditioning is
    not currently available.   Note, however, that symmetric preconditioning 
    can be emulated by using either right or left preconditioning and a pre 
    or post processing step.

    Keywords:  KSP, right, preconditioner, flag
@*/
int KSPSetRightPreconditioner(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->right_pre  = 1;
  return 0;
}

/*@
    KSPGetPreconditionerSide - Gets the preconditioning side.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where
$
$      side = 1:  right preconditioning
$      side = 0:  left preconditioning (default)

    Keywords:  KSP, right, left, side, preconditioner, flag
@*/
int KSPGetPreconditionerSide(KSP itP, int *side) 
{
  VALIDHEADER(itP,KSP_COOKIE);
  *side = (itP)->right_pre;
  return 0;
}

/*@
    KSPGetMethodFromContext - Returns the chosen method type.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

    Output Parameter:
    method - the method type

    Note:
    KSPGetMethod() gets the method from the command line.

    Keywords:  KSP, method, type
@*/
int KSPGetMethodFromContext( KSP itP, KSPMETHOD *method )
{
  VALIDHEADER(itP,KSP_COOKIE);
  *method = itP->type;
  return 0;
}

/*@
   KSPSetRelativeTolerance - Sets the convergence tolerance as a relative 
   decrease in the residual of tol. 

   Input Parameters:
.  itP - Iterative context obtained from KSPCreate()
.  tol - tolerance

   Options Database Key:
$  -ksprtol  tol

   Notes:
   Use KSPSetAbsoluteTolerance() to set the absolute tolerance. The
   first of the two tolerances (absolute and relative) reached
   will terminate the iteration. See also KSPSetConvergenceTest() for 
   setting user-defined stopping criteria.

   Keywords:  KSP, relative, tolerance, convergence, norm, residual
@*/
int KSPSetRelativeTolerance(KSP itP, double tol)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->rtol       = tol;
  return 0;
}
/*@
   KSPSetDivergenceTolerance - Sets the amount that the norm or the 
   residual can increase before KSPDefaultConverged() concludes 
   that the method is diverging.

   Input Parameters:
.  itP - Iterative context obtained from KSPCreate()
.  tol - tolerance

   Options Database Key:
$  -kspdivtol  tol

   Keywords:  KSP, convergence, divergence, tolerance, residual, norm
@*/
int KSPSetDivergenceTolerance(KSP itP, double tol)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->divtol       = tol;
  return 0;
}

/*@
   KSPSetAbsoluteTolerance - Sets the convergence tolerance as an absolute 
   size of the norm of the residual. 

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  tol - tolerance

   Options Database Key:
$  -kspatol  tol

   Notes:
   Use KSPSetRelativeTolerance() to set the relative tolerance. The
   first of the two tolerances (absolute and relative) reached
   will terminate the iterations. See also KSPSetConvergenceTest() 
   for setting user-defined stopping criteria.

   Keywords:  KSP, convergence, tolerance, absolute, residual, norm
@*/
int KSPSetAbsoluteTolerance(KSP itP, double tol) 
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->atol       = tol;
  return 0;
}

/*@
   KSPSetCalculateResidual - Sets a flag so that the two norm of the 
   residual is calculated at each iteration.

   Input Parameter:
.  itP - iterative context obtained from KSPCreate()

   Keywords:  KSP, residual, norm, flag
@*/
int KSPSetCalculateResidual(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->calc_res   = 1;
  return 0;
}

/*@
   KSPSetDoNotCalculateResidual - Sets a flag so that the two norm of the 
   residual is not calculated at each iteration.

   Input Parameter:
.  itP - iterative context obtained from KSPCreate()

   Notes:
   Most Krylov methods do not yet take advantage of this flag.

   Keywords:  KSP, residual, calculate, flag
@*/
int KSPSetDoNotCalculateResidual(KSP itP)
{      
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->calc_res   = 0;
  return 0;
}

/*@
   KSPSetUsePreconditionedResidual - Sets a flag so that the two norm of the 
   preconditioned residual is used rather then the true residual, in the 
   default convergence tests.

   Input Parameter:
.  itP  - iterative context obtained from KSPCreate()

   Notes:
   Currently only CG, CHEBYCHEV, and RICHARDSON use this with left
   preconditioning.  All other methods always used the preconditioned
   residual.  With right preconditioning this flag is ignored.

   Options Database Key:
$  -ksppreres

   Keywords:  KSP, residual, preconditioned, flag
@*/
int KSPSetUsePreconditionedResidual(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->use_pres   = 1;
  return 0;
}

/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the 
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Keywords:  KSP, initial guess, nonzero
@*/
int KSPSetInitialGuessNonzero(KSP itP)
{
  (itP)->guess_zero   = 0;
  return 0;
}

/*@
   KSPSetCalculateEigenvalues - Sets a flag so that the the method will
   calculate the extreme eigenvalues via a Lanczos or Arnoldi process
   as it solves the linear system.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Options Database Key:
$  -kspeigen

   Note:
   Currently this option is not valid for all iterative methods.

   Keywords:  KSP, eigenvalues, calculate, flag
@*/
int KSPSetCalculateEigenvalues(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->calc_eigs  = 1;
  return 0;
}

/*@
   KSPSetRhs - Sets the right-hand-side for the linear system to
   be solved.

   Input Parameters:
.  itP - Iterative context obtained from KSPCreate()
.  b   - right-hand-side vector

   Keywords:  KSP, right-hand-side, rhs, set
@*/
int KSPSetRhs(KSP itP,Vec b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->vec_rhs    = (b);
  return 0;
}

/*@
   KSPGetRhs - Gets the right-hand-side for the linear system to
   be solved.

   Input Parameter:
.  itP - Iterative context obtained from KSPCreate()

   Output Parameter:
.  r - right-hand-side vector

   Keywords:  KSP, right-hand-side, rhs, get
@*/
int KSPGetRhs(KSP itP,Vec *r)
{   
  VALIDHEADER(itP,KSP_COOKIE);
  *r = (itP)->vec_rhs; return 0;
} 

/*@
   KSPSetSolution - Sets the location of the solution for the 
   linear system to be solved.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  x   - solution vector

   Keywords:  KSP, solution, set
@*/
int KSPSetSolution(KSP itP, Vec x)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->vec_sol    = (x);
  return 0;
}

/*@
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved. Note that this may not be were the solution
   is stored during the iterative process, see KSPBuildSolution().

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector

   Keywords:  KSP, solution, get
@*/
int KSPGetSolution(KSP itP, Vec *v)
{
  VALIDHEADER(itP,KSP_COOKIE);  *v = (itP)->vec_sol; return 0;
}

/*@
   KSPSetBinv - Sets the preconditioner to be used to calculate the 
   application of the preconditioner on a vector. 

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  B   - the preconditioner object

   Notes:
   Use KSPGetBinv() to retrieve the preconditioner context (for example,
   to free it at the end of the computations).

   Keywords:  KSP, preconditioner, Binv, set
@*/
int KSPSetBinv(KSP itP,PC B)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->B = B;
  return 0;
}

/*@
   KSPGetBinv - Returns a pointer to the preconditioner context
   set with KSPSetBinv().

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Output Parameter:
.  B - preconditioner context

   Keywords:  KSP, preconditioner, Binv, get
@*/
int KSPGetBinv(KSP itP, PC *B)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *B = (itP)->B; return 0;
}

/*@
   KSPSetMonitor - Sets the function to be used at every
   iteration of the iterative solution. 

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  monitor - pointer to int function
.  mctx    - context for private data for the monitor routine (may be null)

   Calling sequence of monitor:
.  monitor (KSP itP, int it, double rnorm, void *mctx)

   Input Parameters of monitor:
.  itP - iterative context obtained from KSPCreate()
.  it - iteration number
.  mctx  - optional monitoring context, as set by KSPSetMonitor()

   Output Parameter of monitor:
.  rnorm - (estimated) 2-norm of (preconditioned) residual

   Notes:
   The default is to do nothing.  To print the residual, or preconditioned 
   residual if KSPSetUsePreconditionedResidual() was called, use 
   KSPDefaultMonitor() as the monitor routine, with a null monitoring 
   context.
  
   Options Database Key:
$  -kspmonitor   : key for setting KSPDefaultMonitor()
$  -kspxmonitor  : key for setting KSPLGDefaultMonitor()

   Keywords:  KSP, monitor, set
@*/
int KSPSetMonitor(KSP itP, int (*monitor)(KSP,int,double,void*), void *mctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->usr_monitor = monitor;(itP)->monP = (void*)mctx;
  return 0;
}

/*@
   KSPGetMonitorContext - Gets the monitoring context, as set by 
   KSPSetMonitor().

   Input Parameter:
.  itP - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Keywords:  KSP, monitor, context
@*/
int KSPGetMonitorContext(KSP itP, void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx =      ((itP)->monP);
  return 0;
}

/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.
   If set, this array will contain the residual norms computed at each
   iteration of the solver.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  a   - array to hold history
.  na  - size of a

   Keywords:  KSP, residual, history, norm
@*/
int KSPSetResidualHistory(KSP itP, double *a, int na)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->residual_history = a; (itP)->res_hist_size    = na;
  return 0;
}

/*@
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.  

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  converge - pointer to int function
.  cctx    - context for private data for the convergence routine (may be 
              null)

   Calling sequence of converge:
.  converge (KSP itP, int it, double rnorm, void *mctx)

   Input Parameters of converge:
.  itP - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  cctx  - optional convergence context, as set by KSPSetConvergenceTest()

   Return value of converge:
   The convergence test should return 0 for not converged, 1 for 
   converged, and -1 for abort or failure to converge.  

   Notes:
   The default convergence test, KSPDefaultConverged(), aborts if the 
   residual grows to more than 10000 times the initial residual.

   The default is a combination of relative and absolute tolerances.  
   The residual value that is tested may be an approximation; routines 
   that need exact values should compute them.

   Keywords:  KSP, convergence, test, context
@*/
int KSPSetConvergenceTest(KSP itP, int (*converge)(KSP,int,double,void*), 
                          void *cctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->converged = converge;	(itP)->cnvP = (void*)cctx;
  return 0;
}

/*@
   KSPGetConvergenceContext - Gets the convergence context set with 
   KSPSetConvergenceTest().  

   Input Parameter:
.  itP - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Keywords:  KSP, convergence, test, context
@*/
int KSPGetConvergenceContext(KSP itP, void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = ((itP)->cnvP);
  return 0;
}

/*@
   KSPBuildSolution - Builds the approximate solution in a vector provided.

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameter:
.  v - optional location to stash solution.  If v is not provided,
       then a location is generated.
.  V - the solution

   Note:
   Regardless of whether or not v is provided, the solution is 
   returned in V.

   Keywords:  KSP, build, solution
@*/
int KSPBuildSolution(KSP ctx, Vec v, Vec *V)
{
  Vec w = v;
  int ierr;
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!w) {
    ierr = VecCreate(ctx->vec_rhs,&w); CHKERR(ierr);
    PLogObjectParent((PetscObject)ctx,w);
  }
  return (*ctx->BuildSolution)(ctx,w,V);
}

/*@
   KSPBuildResidual - Builds the residual in a vector provided.

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameters:
.  v   - optional location to stash residual.  If v is not provided,
         then a location is generated.
.  t   - work vector.  If not provided then one is generated.
.  V   - the residual

   Note:
   Regardless of whether or not v is provided, the residual is 
   returned in V.

   Keywords:  KSP, build, residual
@*/
int KSPBuildResidual(KSP ctx, Vec t, Vec v, Vec *V)
{
  int flag = 0, ierr;
  Vec w = v, tt = t;
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!w) {
    ierr = VecCreate(ctx->vec_rhs,&w); CHKERR(ierr);
    PLogObjectParent((PetscObject)ctx,w);
  }
  if (!tt) {
    ierr = VecCreate(ctx->vec_rhs,&tt); CHKERR(ierr); flag = 1;
    PLogObjectParent((PetscObject)ctx,tt);
  }
  ierr = (*ctx->BuildResidual)(ctx,tt,w,V); CHKERR(ierr);
  if (flag) ierr = VecDestroy(tt); CHKERR(ierr);
  return ierr;
}

