#include "petsc.h"
#include "kspimpl.h"   /*I "ksp.h" I*/

/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Input Parameter:
.  itP   - iterative context obtained from KSPCreate()
@*/
int KSPSetUp(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->setupcalled) return 0;
  if (itP->method == -1) {
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
@*/
int KSPSolve(KSP itP, int *its) 
{
  int ierr;
  VALIDHEADER(itP,KSP_COOKIE);
  if (!itP->setupcalled){ ierr = KSPSetUp(itP); CHKERR(ierr);}
  return (*(itP)->solver)(itP,its);
}

/*@
   KSPDestroy - Destroys KSPCntx created with KSPCreate().

   Input Parameter:
.  itP   - iterative context obtained from KSPCreate
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
.  maxits - maximum iterations to use
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
@*/
int KSPGetMethodFromContext( KSP itP, KSPMETHOD *method )
{
  VALIDHEADER(itP,KSP_COOKIE);
  *method = itP->method;
  return 0;
}

/*@
   KSPSetRelativeTolerance - Sets the convergence tolerance as a relative 
   decrease in the residual of tol. Use KSPSetAbsoluteTolerance()
   to set an absolute tolerance for convergence. The first of the 
   two tolerances reached will terminate the iteration.

   Input Parameters:
.  itP - Iterative context obtained from KSPCreate()
.  tol - tolerance
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
@*/
int KSPSetDivergenceTolerance(KSP itP, double tol)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->divtol       = tol;
  return 0;
}

/*@
   KSPSetAbsoluteTolerance - Sets the convergence tolerance as an absolute 
   size of the norm of the residual. Use KSPSetRelativeTolerance() for 
   relative tolerance. The first of the two tolerances reached determines
   when the iterations are stopped. See also KSPSetConvergenceTest()
   for how you may set your own, more sophisticated stopping criteria.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  tol - tolerance
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
@*/
int KSPSetDoNotCalculateResidual(KSP itP)
{      
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->calc_res   = 0;
  return 0;
}

/*@
   KSPSetUsePreconditionedResidual - Sets a flag so that the two norm of the 
   preconditioned residual is used rather then the true residual.

   Input Parameter:
.  itP  - iterative context obtained from KSPCreate()

   Notes:
   Currently only CG, CHEBYCHEV, and RICHARDSON use this with left
   preconditioning.  All other methods always used the preconditioned
   residual.  With right preconditioning this flag is ignored.
@*/
int KSPSetUsePreconditionedResidual(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->use_pres   = 1;
  return 0;
}

/*@
   KSPSetInitialGuessZero - Tells the iterative solver
   that the initial guess is zero; otherwise it assumes it is 
   non-zero. If the initial guess is zero, this saves one matrix 
   multiply in the calculation of the initial residual.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
@*/
int KSPSetInitialGuessZero(KSP itP)
{
  (itP)->guess_zero   = 1;
  return 0;
}

/*@
   KSPSetCalculateEigenvalues - Sets a flag so that the the method will
   calculate the extreme eigenvalues via a Lanczo or Arnoldi process
   as it solves the linear system.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
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
@*/
int KSPSetSolution(KSP itP, Vec x)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->vec_sol    = (x);
  return 0;
}

/*@
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector
@*/
int KSPGetSolution(KSP itP, Vec *v)
{
  VALIDHEADER(itP,KSP_COOKIE);  *v = (itP)->vec_sol; return 0;
}

/*@
   KSPSetAmult - Sets the function to be used to calculate the 
   matrix vector product.  

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  amult - pointer to int function
.  amultP - pointer to amult context

   Calling sequence of amult:
.  amult (void *amultP, Vec x, Vec y)

   Input Parameters of amult:
.  amultP - pointer to the amult context, as set by KSPSetAmult()
.  x - input vector

   Output Parameter of amult:
.  y - output vector

   Note:
   Use KSPGetAmultContext() to retrieve the multiplication context, 
   say to free it at the end of the computations.
@*/
int KSPSetAmult(KSP itP, int (*amult)(void *,Vec,Vec), void *amultP)
{
  VALIDHEADER(itP,KSP_COOKIE);  (itP)->amult = amult; (itP)->amultP = amultP;
  return 0;
}

/*@
   KSPGetAmultContext - Returns a pointer to the operator context
   set with KSPSetAmult().

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - matrix multipler context
@*/
int KSPGetAmultContext(KSP itP, void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = (itP)->amultP; return 0;
}

/*@
   KSPSetAmultTranspose - Sets the function to be used to 
   calculate the transpose of the  matrix vector product.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  tamult - pointer to int function

   Calling sequence of tamult:
.  tamult (void *amultP, Vec x, Vec y)

   Input Parameters of tamult:
.  amultP - pointer to the amult context, as set by KSPSetAmult()
.  x - input vector

   Output Parameter of tamult:
.  y - ouput vector
@*/
int KSPSetAmultTranspose(KSP itP, int (*tamult)(void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tamult = tamult;
  return 0;
}

/*@
   KSPSetBinv - Sets the function to be used to calculate the 
   application of the preconditioner on a vector. 

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   binv - pointer to void function
.   binvP - pointer to preconditioner context
 
   Calling sequence of binv:
.  binv (void *binvP, Vec x, Vec y)

   Input Parameters of binv:
.  binvP - pointer to the binv context
.  x - input vector

   Output Parameter of binv:
.  y - ouput vector

   Notes:
   Use KSPGetBinvContext() to retrive the preconditioner context,
   say to free it at the end of the computations.
@*/
int KSPSetBinv(KSP itP, int (*binv)(void *,Vec,Vec), void *binvP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->binv  = binv; (itP)->binvP = (void*)binvP;
  return 0;
}

/*@
   KSPGetBinvContext - Returns a pointer to the preconditioner context
   set with KSPSetBinv().

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - preconditioner context
@*/
int KSPGetBinvContext(KSP itP, void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = (itP)->binv; return 0;
}

/*@
   KSPSetBinvTranspose - Sets the function to be used to calculate the 
   application of the transpose of the preconditioner on a vector. 

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   tbinv - pointer to void function

   Calling sequence of tbinv:
.   tbinv (void *binvP, Vec x, Vec y)

   Input Parameters of tbinv:
.  binvP - pointer to the amult context, as set by KSPSetBinv()
.  x - input vector

   Output Parameter of tbinv:
.  y - ouput vector
@*/
int KSPSetBinvTranspose(KSP itP, int (*tbinv)(void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tbinv  = tbinv;
  return 0;
}

/*@
   KSPSetMatop - Sets the function to be used to calculate the 
   application of the preconditioner followed by the application of the 
   matrix multiplier on a vector.  For left preconditioner the order 
   is reversed.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  matop - pointer to int function

   Calling sequence of matop:
.  matop (void *amultP, void *binvP, Vec x, Vec y)

   Input Parameters of matop:
.  amultP - pointer to the amult context, as set by KSPSetAmult()
.  binvP - pointer to the binv context, as set by KSPSetBinv()
.  x - input vector

   Output Parameter of matop:
.  y - output vector
@*/
int KSPSetMatop(KSP itP, int (*matop)(void *,void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->matop      = matop;
  return 0;
}

/*@
   KSPSetMatopTranspose - Sets the function to be used to calculate
   the action of the transpose of the Matop.

   Input Parameters:
.  itP - iterative context obtained from KSPCreate()
.  tmatop - pointer to void function

   Calling sequence of tmatop:
.  tmatop (void *amultP, void *binvP, Vec x, Vec y)

   Input Parameters of tmatop:
.  amultP - pointer to the amult context, as set by KSPSetAmult()
.  binvP - pointer to the binv context, as set by KSPSetBinv()
.  x - input vector

   Output Parameter of tmatop:
.  y - output vector
@*/
int KSPSetMatopTranspose(KSP itP, int (*tmatop)(void *,void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tmatop      = tmatop;
  return 0;
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
@*/
int KSPGetConvergenceContext(KSP itP, void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = ((itP)->cnvP);
  return 0;
}

/*@
  KSPBuildSolution - builds the solution in a vector provided

  Input Parameter:
. ctx - iterative context obtained from KSPCreate()

  Output Parameter:
. v   - optional location to stash solution.  If v is not provided,
        then a location is generated.
. V   - the solution

  Note:
  Regardless of whether or not v is provided, the solution is 
  returned in V.
@*/
int KSPBuildSolution(KSP ctx, Vec v, Vec *V)
{
  Vec w = v;
  int ierr;
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!w) {ierr = VecCreate(ctx->vec_rhs,&w); CHKERR(ierr);}
  return (*ctx->BuildSolution)(ctx,w,V);
}

/*@
  KSPBuildResidual - Builds the residual in a vector provided.

  Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

  Output Parameters:
.  v   - optional location to stash solution.  If v is not provided,
         then a location is generated.
.  t   - work vector.  If not provided then one is generated.
.  V   - the solution

  Note:
  Regardless of whether or not v is provided, the solution is 
  returned in V.
@*/
int KSPBuildResidual(KSP ctx, Vec t, Vec v, Vec *V)
{
  int flag = 0, ierr;
  Vec w = v, tt = t;
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!w) {ierr = VecCreate(ctx->vec_rhs,&w); CHKERR(ierr);}
  if (!tt) {ierr = VecCreate(ctx->vec_rhs,&tt); CHKERR(ierr); flag = 1;}
  ierr = (*ctx->BuildResidual)(ctx,tt,w,V); CHKERR(ierr);
  if (flag) ierr = VecDestroy(tt); CHKERR(ierr);
  return ierr;
}

