#ifndef lint
static char vcid[] = "$Id: itfunc.c,v 1.67 1996/09/28 16:25:10 curfman Exp balay $";
#endif
/*
      Interface KSP routines that the user calls.
*/
#include "petsc.h"
#include "draw.h"
#include "src/ksp/kspimpl.h"   /*I "ksp.h" I*/

/*@
   KSPComputeExtremeSingularValues - Computes the extreme singular values
   for the preconditioned operator. Called after or during KSPSolve()
   (SLESSolve()).

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  emin, emax - extreme singular values

   Notes:
   One must call KSPSetComputeSingularValues() before calling KSPSetUp() 
   (or use the option -ksp_eigen) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeEigenvalues()
@*/
int KSPComputeExtremeSingularValues(KSP ksp,double *emax,double *emin)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(emax);
  PetscValidScalarPointer(emin);
  if (!ksp->calc_sings) {
    SETERRQ(4,"KSPComputeExtremeSingularValues:SingularValues not requested before KSPSetUp");
  }

  if (ksp->computeextremesingularvalues) {
    return (*ksp->computeextremesingularvalues)(ksp,emax,emin);
  }
  return 0;
}

/*@
   KSPComputeEigenvalues - Computes the extreme eigenvalue
          for the preconditioned operator. Called after or during KSPSolve() (SLESSolve()).
          This does not usually provide accurate estimates; it is only for helping 
          people understand the convergence of iterative methods, not for eigenanalysis. 

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()
.  n - size of arrays r and c

   Output Parameters:
.  r - real part of computed eigenvalues
.  c - complex part of computed eigenvalues

   Notes:
   One must call KSPSetComputeEigenvalues() before calling KSPSetUp() 
   (or use the option -ksp_eigen) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeExtremeSingularValues()
@*/
int KSPComputeEigenvalues(KSP ksp,int n,double *r,double *c)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(r);
  PetscValidScalarPointer(c);
  if (!ksp->calc_sings) {
    SETERRQ(4,"KSPComputeEigenvalues:Eigenvalues not requested before KSPSetUp");
  }

  if (ksp->computeeigenvalues) {
    return (*ksp->computeeigenvalues)(ksp,n,r,c);
  }
  return 0;
}

/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Input Parameter:
.  ksp   - iterative context obtained from KSPCreate()

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy()
@*/
int KSPSetUp(KSP ksp)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->setupcalled) return 0;
  if (ksp->type == -1) {
    SETERRQ(1,"KSPSetUp:Type must be set first");
  }
  ksp->setupcalled = 1;
  return (*ksp->setup)(ksp);
}


/*@
   KSPSolve - Solves linear system; call it after calling 
   KSPCreate(), KSPSetup(), and KSPSet*().

   Input Parameter:
.  ksp - Iterative context obtained from KSPCreate()

   Output Parameter:
.  its - number of iterations required

   Notes:
   On return, the parameter "its" contains
$   - the iteration number at which convergence
$     was successfully reached, 
$   - or the negative of the iteration at which
$      divergence or breakdown was detected.

   If using a direct method (e.g., via the KSP solver
   KSPPREONLY and a preconditioner such as PCLU/PCILU),
   then its=1.  See KSPSetTolerances() and KSPDefaultConverged()
   for more details.

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPDefaultConverged()
@*/
int KSPSolve(KSP ksp, int *its) 
{
  int    ierr,flag1,flag2,rank;
  Scalar zero = 0.0;

  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidIntPointer(its);

  if (!ksp->setupcalled){ ierr = KSPSetUp(ksp); CHKERRQ(ierr);}
  if (ksp->guess_zero) { VecSet(&zero,ksp->vec_sol);}
  ierr = (*ksp->solver)(ksp,its); CHKERRQ(ierr);

  MPI_Comm_rank(ksp->comm,&rank);

  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues",&flag1);CHKERRQ(ierr);
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues",&flag2);CHKERRQ(ierr);
  if (flag1 || flag2) {
    int    n = *its, i;
    double *r,*c;
    r = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(r);
    c = r + n;
    ierr = KSPComputeEigenvalues(ksp,n,r,c); CHKERRQ(ierr);
    if (flag1) {
      PetscPrintf(ksp->comm,"Iteratively computed Eigenvalues\n");
      for ( i=0; i<n; i++ ) {
        PetscPrintf(ksp->comm,"%g %g\n",r[i],c[i]);
      }
    }
    if (flag2 && !rank) {
      Viewer    viewer;
      Draw      draw;
      DrawSP    drawsp;

      ierr = ViewerDrawOpenX(MPI_COMM_SELF,0,"Iteratively Computed Eigenvalues",0,0,300,300,&viewer);
             CHKERRQ(ierr);
      ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
      ierr = DrawSPCreate(draw,1,&drawsp); CHKERRQ(ierr);
      for ( i=0; i<n; i++ ) {
        ierr = DrawSPAddPoint(drawsp,r+i,c+i); CHKERRQ(ierr);
      }
      ierr = DrawSPDraw(drawsp); CHKERRQ(ierr);
      ierr = DrawSPDestroy(drawsp); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    PetscFree(r);
  }

  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues_explicitly",&flag1); CHKERRQ(ierr);
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues_explicitly",&flag2); CHKERRQ(ierr);
  if (flag1 || flag2) {
    int    n, i;
    double *r,*c;
    ierr = VecGetSize(ksp->vec_sol,&n); CHKERRQ(ierr);
    r = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(r);
    c = r + n;
    ierr = KSPComputeEigenvaluesExplicitly(ksp,n,r,c); CHKERRQ(ierr);
    if (flag1) {
      PetscPrintf(ksp->comm,"Explicitly computed Eigenvalues\n");
      for ( i=0; i<n; i++ ) {
        PetscPrintf(ksp->comm,"%g %g\n",r[i],c[i]);
      }
    }
    if (flag2 && !rank) {
      Viewer    viewer;
      Draw      draw;
      DrawSP    drawsp;

      ierr = ViewerDrawOpenX(MPI_COMM_SELF,0,"Explicitly Computed Eigenvalues",0,320,300,300,&viewer); 
             CHKERRQ(ierr);
      ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
      ierr = DrawSPCreate(draw,1,&drawsp); CHKERRQ(ierr);
      for ( i=0; i<n; i++ ) {
        ierr = DrawSPAddPoint(drawsp,r+i,c+i); CHKERRQ(ierr);
      }
      ierr = DrawSPDraw(drawsp); CHKERRQ(ierr);
      ierr = DrawSPDestroy(drawsp); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    PetscFree(r);
  }
  
  return 0;
}

/*@C
   KSPDestroy - Destroys KSP context.

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

.keywords: KSP, destroy

.seealso: KSPCreate(), KSPSetUp(), KSPSolve()
@*/
int KSPDestroy(KSP ksp)
{
  int ierr;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = (*ksp->destroy)((PetscObject)ksp); CHKERRQ(ierr);
  if (ksp->xmonitor) KSPLGMonitorDestroy(ksp->xmonitor);
  PLogObjectDestroy(ksp);
  PetscHeaderDestroy(ksp);
  return 0;
}

/*@
    KSPSetPreconditionerSide - Sets the preconditioning side.

    Input Parameter:
.   ksp - Iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
$
$      PC_LEFT - left preconditioning (default)
$      PC_RIGHT - right preconditioning
$      PC_SYMMETRIC - symmetric preconditioning

   Options Database Keys:
$  -ksp_left_pc, -ksp_right_pc, -ksp_symmetric_pc,

    Notes:
    Left preconditioning is used by default.  Symmetric preconditioning is
    currently available only for the KSPQCG method. Note, however, that
    symmetric preconditioning can be emulated by using either right or left
    preconditioning and a pre or post processing step.

.keywords: KSP, set, right, left, symmetric, side, preconditioner, flag

.seealso: KSPGetPreconditionerSide()
@*/
int KSPSetPreconditionerSide(KSP ksp,PCSide side)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->pc_side = side;
  return 0;
}

/*@C
    KSPGetPreconditionerSide - Gets the preconditioning side.

    Input Parameter:
.   ksp - Iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
$
$      PC_LEFT - left preconditioning (default)
$      PC_RIGHT - right preconditioning
$      PC_SYMMETRIC - symmetric preconditioning

.keywords: KSP, get, right, left, symmetric, side, preconditioner, flag

.seealso: KSPSetPreconditionerSide()
@*/
int KSPGetPreconditionerSide(KSP ksp, PCSide *side) 
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *side = ksp->pc_side;
  return 0;
}

/*@
   KSPGetTolerances - Gets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence tests. 

   Input Parameter:
.  ksp - the Krylov subspace context
  
   Output Parameters:
.  rtol - the relative convergence tolerance
.  atol - the absolute convergence tolerance
.  dtol - the divergence tolerance
.  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

.keywords: KSP, get, tolerance, absolute, relative, divergence, convergence,
maximum, iterations

.seealso: KSPSetTolerances()
@*/
int KSPGetTolerances(KSP ksp,double *rtol,double *atol,double *dtol,
                     int *maxits)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol)   *atol   = ksp->atol;
  if (rtol)   *rtol   = ksp->rtol;
  if (dtol)   *dtol   = ksp->divtol;
  if (maxits) *maxits = ksp->max_it;
  return 0;
}
/*@
    KSPSetTolerances - Sets the relative, absolute, divergence, and maximum
    iteration tolerances used by the default KSP convergence testers. 

   Input Parameters:
.  ksp - the Krylov subspace context
.  rtol - the relative convergence tolerance
   (relative decrease in the residual norm)
.  atol - the absolute convergence tolerance 
   (absolute size of the residual norm)
.  dtol - the divergence tolerance
   (amount residual can increase before KSPDefaultConverged
   concludes that the method is diverging)
.  maxits - maximum number of iterations to use

   Notes:
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   See KSPDefaultConverged() for details on the use of these parameters
   in the default convergence test.  See also KSPSetConvergenceTest() 
   for setting user-defined stopping criteria.

   Options Database Keys:
$  -ksp_atol  tol  (absolute tolerance)
$  -ksp_rtol  tol  (relative tolerance)
$  -ksp_divtol  tol  (divergence tolerance)
$  -ksp_max_it  maxits  (maximum iterations)

.keywords: KSP, set, tolerance, absolute, relative, divergence, 
           convergence, maximum, iterations

.seealso: KSPGetTolerances(), KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPSetTolerances(KSP ksp,double rtol,double atol,double dtol,int maxits)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol != PETSC_DEFAULT)   ksp->atol   = atol;
  if (rtol != PETSC_DEFAULT)   ksp->rtol   = rtol;
  if (dtol != PETSC_DEFAULT)   ksp->divtol = dtol;
  if (maxits != PETSC_DEFAULT) ksp->max_it = maxits;
  return 0;
}

/*@
   KSPSetComputeResidual - Sets a flag to indicate whether the two norm 
   of the residual is calculated at each iteration.

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  flag - PETSC_TRUE or PETSC_FALSE

   Notes:
   Most Krylov methods do not yet take advantage of flag = PETSC_FALSE.

.keywords: KSP, set, residual, norm, calculate, flag
@*/
int KSPSetComputeResidual(KSP ksp,PetscTruth flag)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_res   = flag;
  return 0;
}

/*@
   KSPSetUsePreconditionedResidual - Sets a flag so that the two norm of the 
   preconditioned residual is used rather than the true residual, in the 
   default convergence tests.

   Input Parameter:
.  ksp  - iterative context obtained from KSPCreate()

   Notes:
   Currently only CG, CHEBYCHEV, and RICHARDSON use this with left
   preconditioning.  All other methods always used the preconditioned
   residual.  With right preconditioning this flag is ignored, since 
   the preconditioned residual and true residual are the same.

   Options Database Key:
$  -ksp_preres

.keywords: KSP, set, residual, precondition, flag
@*/
int KSPSetUsePreconditionedResidual(KSP ksp)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->use_pres   = 1;
  return 0;
}

/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the 
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

.keywords: KSP, set, initial guess, nonzero
@*/
int KSPSetInitialGuessNonzero(KSP ksp)
{
  ksp->guess_zero   = 0;
  return 0;
}

/*@
   KSPSetComputeSingularValues - Sets a flag so that the extreme singular 
   values will be calculated via a Lanczos or Arnoldi process as the linear 
   system is solved.

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
$  -ksp_singmonitor

   Notes:
   Currently this option is not valid for all iterative methods.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, set, singular values, calculate

.seealso: KSPComputeExtremeSingularValues(), KSPSingularValueMonitor()
@*/
int KSPSetComputeSingularValues(KSP ksp)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_sings  = 1;
  return 0;
}

/*@
   KSPSetRhs - Sets the right-hand-side vector for the linear system to
   be solved.

   Input Parameters:
.  ksp - Iterative context obtained from KSPCreate()
.  b   - right-hand-side vector

.keywords: KSP, set, right-hand-side, rhs

.seealso: KSPGetRhs(), KSPSetSolution()
@*/
int KSPSetRhs(KSP ksp,Vec b)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  ksp->vec_rhs    = (b);
  return 0;
}

/*@C
   KSPGetRhs - Gets the right-hand-side vector for the linear system to
   be solved.

   Input Parameter:
.  ksp - Iterative context obtained from KSPCreate()

   Output Parameter:
.  r - right-hand-side vector

.keywords: KSP, get, right-hand-side, rhs

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPGetRhs(KSP ksp,Vec *r)
{   
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *r = ksp->vec_rhs; return 0;
} 

/*@
   KSPSetSolution - Sets the location of the solution for the 
   linear system to be solved.

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  x   - solution vector

.keywords: KSP, set, solution

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPSetSolution(KSP ksp, Vec x)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  ksp->vec_sol    = (x);
  return 0;
}

/*@C
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved.  Note that this may not be where the solution
   is stored during the iterative process; see KSPBuildSolution().

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector

.keywords: KSP, get, solution

.seealso: KSPGetRhs(), KSPSetSolution()
@*/
int KSPGetSolution(KSP ksp, Vec *v)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);  *v = ksp->vec_sol; return 0;
}

/*@
   KSPSetPC - Sets the preconditioner to be used to calculate the 
   application of the preconditioner on a vector. 

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  B   - the preconditioner object

   Notes:
   Use KSPGetPC() to retrieve the preconditioner context (for example,
   to free it at the end of the computations).

.keywords: KSP, set, precondition, Binv

.seealso: KSPGetPC()
@*/
int KSPSetPC(KSP ksp,PC B)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(B,PC_COOKIE);
  ksp->B = B;
  return 0;
}

/*@C
   KSPGetPC - Returns a pointer to the preconditioner context
   set with KSPSetPC().

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  B - preconditioner context

.keywords: KSP, get, preconditioner, Binv

.seealso: KSPSetPC()
@*/
int KSPGetPC(KSP ksp, PC *B)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *B = ksp->B; return 0;
}

/*@C
   KSPSetMonitor - Sets the function to be used at every
   iteration of the iterative solution. 

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  monitor - pointer to int function
.  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context
             is desired)

   Calling sequence of monitor:
.  monitor (KSP ksp, int it, double rnorm, void *mctx)

   Input parameters of monitor:
.  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  mctx  - optional monitoring context, as set by KSPSetMonitor()

   Options Database Keys:
$    -ksp_monitor      : sets KSPDefaultMonitor()
$    -ksp_truemonitor  : sets KSPTrueMonitor()
$    -ksp_xmonitor     : sets line graph monitor,
$                        uses KSPLGMonitorCreate()
$    -ksp_xtruemonitor : sets line graph monitor,
$                        uses KSPLGMonitorCreate()
$    -ksp_singmonitor  : sets KSPSingularValueMonitor()
$    -ksp_bsmonitor    : sets BlockSolve95 monitor that
$                        prints both scaled and unscaled
$                        residual norms

   Notes:  
   The default is to do nothing.  To print the residual, or preconditioned 
   residual if KSPSetUsePreconditionedResidual() was called, use 
   KSPDefaultMonitor() as the monitor routine, with a null monitoring 
   context.

.keywords: KSP, set, monitor

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPSetMonitor(KSP ksp, int (*monitor)(KSP,int,double,void*), void *mctx)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->monitor = monitor;ksp->monP = (void*)mctx;
  return 0;
}

/*@C
   KSPGetMonitorContext - Gets the monitoring context, as set by 
   KSPSetMonitor().

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

.keywords: KSP, get, monitor, context

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPGetMonitorContext(KSP ksp, void **ctx)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx =      (ksp->monP);
  return 0;
}

/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.
   If set, this array will contain the residual norms computed at each
   iteration of the solver.

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  a   - array to hold history
.  na  - size of a

.keywords: KSP, set, residual, history, norm
@*/
int KSPSetResidualHistory(KSP ksp, double *a, int na)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (na) PetscValidScalarPointer(a);
  ksp->residual_history = a;
  ksp->res_hist_size    = na;
  return 0;
}

/*@C
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.  

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()
.  converge - pointer to int function
.  cctx    - context for private data for the convergence routine (may be 
              null)

   Calling sequence of converge:
.  converge (KSP ksp, int it, double rnorm, void *mctx)

   Input Parameters of converge:
.  ksp - iterative context obtained from KSPCreate()
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

.keywords: KSP, set, convergence, test, context

.seealso: KSPDefaultConverged(), KSPGetConvergenceContext()
@*/
int KSPSetConvergenceTest(KSP ksp,int (*converge)(KSP,int,double,void*),void *cctx)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->converged = converge;	ksp->cnvP = (void*)cctx;
  return 0;
}

/*@C
   KSPGetConvergenceContext - Gets the convergence context set with 
   KSPSetConvergenceTest().  

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

.keywords: KSP, get, convergence, test, context

.seealso: KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPGetConvergenceContext(KSP ksp, void **ctx)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx = ksp->cnvP;
  return 0;
}

/*@C
   KSPBuildSolution - Builds the approximate solution in a vector provided.
   This routine is NOT commonly needed (see SLESSolve()).

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameter: Provide exactly one of
.  v - location to stash solution.   
.  V - the solution is returned in this location. This vector is created 
       internally. This vector should NOT be destroyed by the user with
       VecDestroy().

   Notes:
   This routine must be called after SLESSolve().
   This routine can be used in one of two ways
$    KSPBuildSolution(ctx,PETSC_NULL,&V) or
$    KSPBuildSolution(ctx,v,PETSC_NULL); 
   In the first case an internal vector is allocated to store the solution
   (the user cannot destroy this vector). In the second case the solution
   is generated in the vector that the user provides. Note that for certain 
   methods, such as KSPCG, the second case requires a copy of the solution,
   while in the first case the call is essentially free since it simply 
   returns the vector where the solution already is stored.

.keywords: KSP, build, solution

.seealso: KSPGetSolution(), KSPBuildResidual()
@*/
int KSPBuildSolution(KSP ctx, Vec v, Vec *V)
{
  PetscValidHeaderSpecific(ctx,KSP_COOKIE);
  if (!V && !v) SETERRQ(1,"KSPBuildSolution:Must provide either v or V");
  if (!V) V = &v;
  return (*ctx->buildsolution)(ctx,v,V);
}

/*@C
   KSPBuildResidual - Builds the residual in a vector provided.

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameters:
.  v   - optional location to stash residual.  If v is not provided,
         then a location is generated.
.  t   - work vector.  If not provided then one is generated.
.  V   - the residual

   Notes:
   Regardless of whether or not v is provided, the residual is 
   returned in V.

.keywords: KSP, build, residual

.seealso: KSPBuildSolution()
@*/
int KSPBuildResidual(KSP ctx, Vec t, Vec v, Vec *V)
{
  int flag = 0, ierr;
  Vec w = v, tt = t;
  PetscValidHeaderSpecific(ctx,KSP_COOKIE);
  if (!w) {
    ierr = VecDuplicate(ctx->vec_rhs,&w); CHKERRQ(ierr);
    PLogObjectParent((PetscObject)ctx,w);
  }
  if (!tt) {
    ierr = VecDuplicate(ctx->vec_rhs,&tt); CHKERRQ(ierr); flag = 1;
    PLogObjectParent((PetscObject)ctx,tt);
  }
  ierr = (*ctx->buildresidual)(ctx,tt,w,V); CHKERRQ(ierr);
  if (flag) ierr = VecDestroy(tt); CHKERRQ(ierr);
  return ierr;
}

