#include "petsc.h"
#include "kspimpl.h"   /*I "ksp.h" I*/

/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Input Parameters:
.   itP   - iterative context obtained from KSPCreate()
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

   Input Parameters:
.   itP - Iterative context obtained from KSPCreate()

   Returns:
   The number of iterations required.  If the return is negative, the 
   iterations were aborted by the convergence tester (if the default
   convergence test is used, this happens when the residual grows to more
   than 10000 times the initial residual).
@*/
int KSPSolve(KSP itP,int *its) 
{
  int ierr;
  VALIDHEADER(itP,KSP_COOKIE);
  if (!itP->setupcalled){ ierr = KSPSetUp(itP); CHKERR(ierr);}
  return (*(itP)->solver)(itP,its);
}

/*@
   KSPDestroy Destroys KSPCntx created with KSPCreate().

   Input Parameters:
.   itP   - iterative context obtained from KSPCreate
@*/
int KSPDestroy(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  return (*(itP)->destroy)((PetscObject)itP);
}

/*@
   KSPSetIterations - Sets the maximum number of iterations to use.

   Input Parameters:
.   itP  - iterative context obtained from KSPCreate()
.   maxits - maximum iterations to use
@*/
int KSPSetIterations(KSP itP, int its)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->max_it = its;
  return 0;
}

/*@
    KSPSetRightPreconditioner - Sets right preconditioning.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

    Note:
    Left preconditioning is used by default.  Symmetric preconditioning is
    not currently available (note that it can be emulated by using either
    right or left preconditioning and a pre or post processing step).
@*/
int KSPSetRightPreconditioner(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->right_pre  = 1;
  return 0;
}

/*@
    KSPGetPreconditionerSide - Gets preconditioning side.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

    Returns:
    One for right preconditioning and zero for left preconditioning.
@*/
int KSPGetPreconditionerSide(KSP itP,int *side) 
{
  VALIDHEADER(itP,KSP_COOKIE);
  *side = (itP)->right_pre;
  return 0;
}

/*@
    KSPGetMethodFromContext - Returns the chosen method type.

    Input Parameter:
.   itP - Iterative context obtained from KSPCreate()


    Note:
    KSPGetMethod gets the method from the command line.
@*/
int KSPGetMethodFromContext( KSP itP,KSPMETHOD *method )
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
.   itP - Iterative context obtained from KSPCreate()
.   tol - tolerance

@*/
int KSPSetRelativeTolerance(KSP itP, double r)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->rtol       = r;
  return 0;
}
/*@
   KSPSetDivergenceTolerance - Sets the amount that the norm or the 
      residual can increase before KSPDefaultConverged() concludes 
      that the method is diverging.

   Input Parameters:
.   itP - Iterative context obtained from KSPCreate()
.   tol - tolerance

@*/
int KSPSetDivergenceTolerance(KSP itP, double r)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->divtol       = r;
  return 0;
}

/*@
   KSPSetAbsoluteTolerance - Sets the convergence tolerance as an absolute 
   size of the norm of the residual. Use KSPSetRelativeTolerance() for 
   relative tolerance. The first of the two tolerances reached determines
   when the iterations are stopped. See also KSPSetConvergenceTest()
   for how you may set your own, more sophisticated stopping criteria.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   tol - tolerance

@*/
int KSPSetAbsoluteTolerance(KSP itP, double a) 
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->atol       = a;
  return 0;
}

/*@
   KSPSetCalculateResidual - Sets a flag so that the two norm of the 
   residual is calculated at each iteration.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

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

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

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

   Input Parameters:
.   itP  - iterative context obtained from KSPCreate()

   Notes:
     Currently only CG, CHEBYCHEV, and RICHARDSON use this with left
   preconditioning. All other methods always used the preconditioned
   residual. With right preconditioning this flag is ignored.
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
.   itP - iterative context obtained from KSPCreate()

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
.   itP - iterative context obtained from KSPCreate()

@*/
int KSPSetCalculateEigenvalues(KSP itP)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->calc_eigs  = 1;
  return 0;
}

/*@
   KSPSetRhs - Sets the right hand side for the linear system to
   be solved.

   Input Parameters:
.   itP - Iterative context obtained from KSPCreate()
.   x   - the right hand side vector

@*/
int KSPSetRhs(KSP itP,Vec b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->vec_rhs    = (b);
  return 0;
}

/*@
   KSPGetRhs - Gets the right-hand side for the linear system to
   be solved.

   Input Parameter:
.   itP - Iterative context obtained from KSPCreate()

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
.   itP - iterative context obtained from KSPCreate()
.   x   - the solution vector

@*/
int KSPSetSolution(KSP itP,Vec b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->vec_sol    = (b);
  return 0;
}

/*@
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

@*/
int KSPGetSolution(KSP itP,Vec *v)
{
  VALIDHEADER(itP,KSP_COOKIE);  *v = (itP)->vec_sol; return 0;
}

/*@
   KSPSetAmult - Sets the function to be used to calculate the 
   matrix vector product. Use KSPGetAmultContext() to retrive the 
   multiply context, say at the end of the computations.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   amult - pointer to int function
.   amultP - pointer to amult context

@*/
int KSPSetAmult(KSP itP,int (*a)(void *,Vec,Vec), void *b)
{
  VALIDHEADER(itP,KSP_COOKIE);  (itP)->amult = a;(itP)->amultP = b;
  return 0;
}

/*@
   KSPGetAmultContext - Returns a pointer to the operator context
   set with KSPSetAmult().

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

   Output Parameter:
.   returns the matrix multipler context
 
@*/
int KSPGetAmultContext(KSP itP,void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = (itP)->amultP; return 0;
}

/*@
   KSPSetAmultTranspose - Sets the function to be used to 
   calculate the transpose of the  matrix vector product.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   tamult - pointer to void function

@*/
int KSPSetAmultTranspose(KSP itP,int   (*a)(void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tamult = a;
  return 0;
}

/*@
   KSPSetBinv - Sets the function to be used to calculate the 
   application of the preconditioner on a vector. Use 
   KSPGetBinvContext() to retrive the preconditioner context,
   say to free it at the end of the conputations.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   binv - pointer to void function
.   binvP - pointer to preconditioner context
 
@*/
int KSPSetBinv(KSP itP,int   (*a)(void *,Vec,Vec),void *b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->binv  = a; (itP)->binvP = (void*)b;
  return 0;
}

/*@
   KSPGetBinvContext - Returns a pointer to the preconditioner context
   set with KSPSetBinv().

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

   Output Parameter:
.   returns the preconditioner context
 
@*/
int KSPGetBinvContext(KSP itP,void **ctx)
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
 
@*/
int KSPSetBinvTranspose(KSP itP,int (*a)(void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tbinv  = a;
  return 0;
}

/*@
   KSPSetMatop - Sets the function to be used to calculate the 
   application of the preconditioner followed by the application of the 
   matrix multiplier on a vector. For left preconditioner the order 
   is reversed.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   matop - pointer to void function

@*/
int KSPSetMatop(KSP itP,int (*a)(void *,void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->matop      = a;
  return 0;
}

/*@
   KSPSetMatopTranspose - Sets the function to be used to calculate
   the action of the transpose of the Matop.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   tmatop - pointer to void function

@*/
int KSPSetMatopTranspose(KSP itP,int (*a)(void *,void *,Vec,Vec))
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->tmatop      = a;
  return 0;
}

/*@
   KSPSetMonitor - Sets the function to be used at every
   iteration of the iterative solution. 

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   monitor - pointer to void function
.   mctx    - context for private data for the monitor routine (may be null)

   Notes:
   The default is to do nothing.  To print the residual, or preconditioned 
   residual if KSPSetUsePreconditionedResidual was called, use 
   KSPDefaultMonitor as the monitor routine, with a null context.

   The function has the format
$        void fcn(itP,it,rnorm)
$        KSP itP;
$        int    it;              Iteration number
$        double rnorm;           (Estimated) 2-norm of residual

@*/
int KSPSetMonitor(KSP itP,int   (*a)(KSP,int,double,void*),void  *b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->usr_monitor = a;(itP)->monP = (void*)b;
  return 0;
}

/*@
   KSPGetMonitorContext - Gets the context provided by KSPSetMonitor.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

@*/
int KSPGetMonitorContext(KSP itP,void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx =      ((itP)->monP);
  return 0;
}

/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   a   - array to hold history
.   na  - size of a

@*/
int KSPSetResidualHistory(KSP itP,double *a,int    na)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->residual_history = a; (itP)->res_hist_size    = na;
  return 0;
}

/*@
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.  

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()
.   converge - pointer to int function
.   cctx    - context for private data for the convergence routine (may be 
              null)

   Notes:
   The test should return 0 for not converged, 1 for converged, and -1 for
   abort on failure to converge.  The default convergence tester aborts 
   if the residual grows to more than 10000 times the initial residual.

   The default is a combination of relative and absolute
   tolerances.  The residual value that is tested my be an approximation;
   routines that need exact values should compute them.

   The function has the format
$        int fcn(itP,it,rnorm)
$        KSP *itP;
$        int    it;              Iteration number
$        double rnorm;           (Estimated) 2-norm of residual
@*/
int KSPSetConvergenceTest(KSP itP,int (*a)(KSP,int,double,void*),void *b)
{
  VALIDHEADER(itP,KSP_COOKIE);
  (itP)->converged = a;(itP)->cnvP = (void*)b;
  return 0;
}

/*@
  KSPGetConvergenceContext - Gets the convergence context set with 
  KSPSetConvergenceTest().  

   Input Parameters:
.   itP - iterative context obtained from KSPCreate()

@*/
int KSPGetConvergenceContext(KSP itP,void **ctx)
{
  VALIDHEADER(itP,KSP_COOKIE);
  *ctx = ((itP)->cnvP);
  return 0;
}

/*@
  KSPBuildSolution - builds the solution in a vector provided

  Input Parameters:
.  ctx - the KSP context

  OutPut Parameters:
.  v   - location to stash solution. If not provided then generates one.
@*/
int KSPBuildSolution(KSP ctx,Vec v,Vec *V)
{
  Vec w = v;
  int ierr;
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!w) {ierr = VecCreate(ctx->vec_rhs,&w); CHKERR(ierr);}
  return (*ctx->BuildSolution)(ctx,w,V);
}

/*@
  KSPBuildResidual - builds the residual in a vector provided

  Input Parameters:
.  ctx - the KSP context

  OutPut Parameters:
.  v   - location to stash solution. If not provided then generates one.
.  t   - work vector.  If not provided then generates one.
@*/
int KSPBuildResidual(KSP ctx,Vec t,Vec v,Vec *V)
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



