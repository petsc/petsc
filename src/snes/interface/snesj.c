
#ifndef lint
static char vcid[] = "$Id: snes.c,v 1.11 1995/04/27 20:17:07 bsmith Exp bsmith $";
#endif

#include "draw.h"
#include "snesimpl.h"
#include "options.h"

/*@
   SNESSetFromOptions - Sets various SLES parameters from user options.

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, set, options, database

.seealso: SNESPrintHelp()
@*/
int SNESSetFromOptions(SNES snes)
{
  SNESMETHOD method;
  double tmp;
  SLES   sles;
  VALIDHEADER(snes,SNES_COOKIE);
  if (SNESGetMethodFromOptions(snes,&method)) {
    SNESSetMethod(snes,method);
  }
  if (OptionsHasName(0,0,"-help"))  SNESPrintHelp(snes);
  if (OptionsGetDouble(0,snes->prefix,"-snes_stol",&tmp)) {
    SNESSetSolutionTolerance(snes,tmp);
  }
  if (OptionsGetDouble(0,snes->prefix,"-snes_ttol",&tmp)) {
    SNESSetTruncationTolerance(snes,tmp);
  }
  if (OptionsGetDouble(0,snes->prefix,"-snes_atol",&tmp)) {
    SNESSetAbsoluteTolerance(snes,tmp);
  }
  if (OptionsGetDouble(0,snes->prefix,"-snes_rtol",&tmp)) {
    SNESSetRelativeTolerance(snes,tmp);
  }
  OptionsGetInt(0,snes->prefix,"-snes_max_it",&snes->max_its);
  OptionsGetInt(0,snes->prefix,"-snes_max_resid",&snes->max_resids);
  if (OptionsHasName(0,snes->prefix,"-snes_monitor")) {
    SNESSetMonitor(snes,SNESDefaultMonitor,0);
  }
  if (OptionsHasName(0,snes->prefix,"-snes_xmonitor")){
    int       ierr,mytid = 0;
    DrawLGCtx lg;
    MPI_Initialized(&mytid);
    if (mytid) MPI_Comm_rank(snes->comm,&mytid);
    if (!mytid) {
      ierr = SNESLGMonitorCreate(0,0,0,0,300,300,&lg); CHKERR(ierr);
      SNESSetMonitor(snes,SNESLGMonitor,(void *)lg);
    }
  }
  SNESGetSLES(snes,&sles);
  SLESSetFromOptions(sles);
  if (!snes->SetFromOptions) return 0;
  return (*snes->SetFromOptions)(snes);
}

/*@
   SNESPrintHelp - Prints all options for the SNES component.

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, help

.seealso: SLESSetFromOptions()
@*/
int SNESPrintHelp(SNES snes)
{
  int     rank;
  char    *prefix = "-";
  if (snes->prefix) prefix = snes->prefix;
  VALIDHEADER(snes,SNES_COOKIE);
  if (!snes->PrintHelp) return 0;
  MPI_Comm_rank(snes->comm,&rank); if (rank) return 0;
  fprintf(stderr,"SNES options ----------------------------\n");
  fprintf(stderr,"%ssnes_method [ls] \n",prefix);
  fprintf(stderr,"%ssnes_stol tol (default %g)\n",prefix,snes->xtol);
  fprintf(stderr,"%ssnes_atol tol (default %g)\n",prefix,snes->atol);
  fprintf(stderr,"%ssnes_rtol tol (default %g)\n",prefix,snes->rtol);
  fprintf(stderr,"%ssnes_ttol tol (default %g)\n",prefix,snes->trunctol);
  fprintf(stderr,"%ssnes_max_it its (default %d)\n",prefix,snes->max_its);
  fprintf(stderr,"%ssnes_monitor\n",prefix);
  return (*snes->PrintHelp)(snes);
}
/*@
   SNESSetApplicationContext - Sets the optional user-defined context for 
   the nonlinear solvers.  

   Input Parameters:
.  snes - the SNES context
.  usrP - optional user context

.keywords: SNES, nonlinear, set, application, context

.seealso: SNESGetApplicationContext()
@*/
int SNESSetApplicationContext(SNES snes,void *usrP)
{
   VALIDHEADER(snes,SNES_COOKIE);
   snes->user		= usrP;
   return 0;
}
/*@
   SNESGetApplicationContext - Gets the user-defined context for the 
   nonlinear solvers.  

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  usrP - user context

.keywords: SNES, nonlinear, get, application, context

.seealso: SNESSetApplicationContext()
@*/
int SNESGetApplicationContext( SNES snes,  void **usrP )
{
   VALIDHEADER(snes,SNES_COOKIE);
   *usrP = snes->user;
   return 0;
}

/*@
   SNESGetSLES - Returns the SLES context for a SNES solver.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  sles - the SLES context

   Notes:
   The user can then directly manipulate the SLES context to set various
   options, etc.  Likewise, the user can then manipulate the KSP and PC 
   contexts as well.

.keywords: SNES, nonlinear, get, SLES, context

.seealso: SLESGetPC(), SLESGetKSP()
@*/
int SNESGetSLES(SNES snes,SLES *sles)
{
   VALIDHEADER(snes,SNES_COOKIE);
  *sles = snes->sles;
  return 0;
}

/* -----------------------------------------------------------*/

/*@
   SNESCreate - Creates a nonlinear solver context.

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  outsnes - the new SNES context

.keywords: SNES, nonlinear, create, context

.seealso: SNESSetUp(), SNESSolve(), SNESDestroy()
@*/
int SNESCreate(MPI_Comm comm, SNES *outsnes)
{
  int  ierr;
  SNES snes;
  *outsnes = 0;
  PETSCHEADERCREATE(snes,_SNES,SNES_COOKIE,SNES_NLS,comm);
  PLogObjectCreate(snes);
  snes->max_its         = 50;
  snes->max_resids	= 1000;
  snes->max_funcs	= 1000;
  snes->norm		= 0.0;
  snes->rtol		= 1.e-8;
  snes->atol		= 1.e-10;
  snes->xtol		= 1.e-8;
  snes->trunctol	= 1.e-12;
  snes->nresids         = 0;
  snes->Monitor         = 0;
  ierr = SLESCreate(comm,&snes->sles); CHKERR(ierr);
  PLogObjectParent(snes,snes->sles)
  *outsnes = snes;
  return 0;
}

/* --------------------------------------------------------------- */
/*@C
   SNESSetFunction - Sets the residual evaluation routine and residual 
   vector for use by the SNES routines.

   Input Parameters:
.  snes - the SNES context
.  func - residual evaluation routine
.  resid_neg - indicator whether func evaluates f or -f. 
   If resid_neg is nonzero, then func evaluates -f; otherwise, 
   func evaluates f.
.  ctx - optional user-defined function context 
.  r - vector to store residual

   Calling sequence of func:
.  func (Vec x, Vec f, void *ctx);

.  x - input vector
.  f - residual vector or its negative
.  ctx - optional user-defined context for private data for the 
         residual evaluation routine (may be null)

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
$  where f'(x) denotes the Jacobian matrix and f(x) is the residual.
   By setting resid_neg = 1, the user can supply -f(x) directly.

.keywords: SNES, nonlinear, set, residual

.seealso: SNESGetFunction(), SNESSetJacobian(), SNESSetSolution()
@*/
int SNESSetFunction( SNES snes, Vec r, int (*func)(Vec,Vec,void*),
                     void *ctx,int rneg)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->ComputeResidual     = func; 
  snes->rsign               = rneg;
  snes->vec_res             = r;
  snes->resP                = ctx;
  return 0;
}

int SNESComputeFunction(SNES snes,Vec x, Vec y)
{
  int    ierr;
  Scalar mone = -1.0;
  ierr = (*snes->ComputeResidual)(x,y,snes->resP); CHKERR(ierr);
  snes->nresids++;
  if (!snes->rsign) {
    ierr = VecScale(&mone,y); CHKERR(ierr);
  }
  return 0;
}

/*@
   SNESSetJacobian - Sets the function to compute Jacobian as well as the
   location to store it.

   Input Parameters:
.  snes - the SNES context
.  A - Jacobian matrix
.  func - Jacobian evaluation routine
.  ctx - optional user-defined context for private data for the 
         Jacobian evaluation routine (may be null)

   Calling sequence of func:
.  func (Vec x, Mat *A, Mat *B, int *flag,void *ctx);

.  x - input vector
.  A - Jacobian matrix
.  B - preconditioner matrix, usually the same as A
.  flag - same as options to SLESSetOperators(). Usually 0 or 
$         MAT_SAME_NONZERO_PATTERN
.  ctx - optional user-defined Jacobian context

   Notes: 
   The function func() takes a Mat * as an argument rather than a Mat.
   This is to allow the Jacobian code to replace it with a new matrix 
   when appropriate, for instance, if the nonzero structure is changing.

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: SNESSetFunction(), SNESSetSolution()
@*/
int SNESSetJacobian(SNES snes,Mat A,Mat B,int (*func)(Vec,Mat*,Mat*,int*,void*),
                    void *ctx)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->ComputeJacobian = func;
  snes->jacP            = ctx;
  snes->jacobian        = A;
  snes->jacobian_pre    = B;
  return 0;
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */

/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.  Call SNESSetUp() after calling SNESCreate()
   and optional routines of the form SNESSetXXX(), but before calling 
   SNESSolve().  

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, setup

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
int SNESSetUp(SNES snes)
{
  VALIDHEADER(snes,SNES_COOKIE);
  return (*(snes)->Setup)( snes );
}

/*@
   SNESDestroy - Destroys the nonlinear solver context that was created
   with SNESCreate().

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, destroy

.seealso: SNESCreate(), SNESSetUp(), SNESSolve()
@*/
int SNESDestroy(SNES snes)
{
  VALIDHEADER(snes,SNES_COOKIE);
  return (*(snes)->destroy)( (PetscObject) snes );
}

/* ----------- Routines to set solver parameters ---------- */

/*@
   SNESSetMaxIterations - Sets the maximum number of global iterations to use.

   Input Parameters:
.  snes - the SNES context
.  maxits - maximum number of iterations to use

   Options Database Key:
$  -snes_max_it  maxits

   Note:
   The default maximum number of iterations is 50.

.keywords: SNES, nonlinear, set, maximum, iterations

.seealso: SNESSetMaxResidualEvaluations()
@*/
int SNESSetMaxIterations(SNES snes,int maxits)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->max_its = maxits;
  return 0;
}

/*@
   SNESSetMaxResidualEvaluations - Sets the maximum number of residual
   evaluations to use.

   Input Parameters:
.  snes - the SNES context
.  maxr - maximum number of residual evaluations

   Options Database Key:
$  -snes_max_resid maxr

   Note:
   The default maximum number of residual evaluations is 1000.

.keywords: SNES, nonlinear, set, maximum, residual, evaluations

.seealso: SNESSetMaxIterations()
@*/
int SNESSetMaxResidualEvaluations(SNES snes,int maxr)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->max_resids = maxr;
  return 0;
}

/*@
   SNESSetRelativeTolerance - Sets the relative convergence tolerance.  

   Input Parameters:
.  snes - the SNES context
.  rtol - tolerance
   
   Options Database Key: 
$    -snes_rtol tol

.keywords: SNES, nonlinear, set, relative, convergence, tolerance
 
.seealso: SNESSetAbsoluteTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetRelativeTolerance(SNES snes,double rtol)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->rtol = rtol;
  return 0;
}

/*@
   SNESSetAbsoluteTolerance - Sets the absolute convergence tolerance.  

   Input Parameters:
.  snes - the SNES context
.  atol - tolerance

   Options Database Key: 
$    -snes_atol tol

   Notes:
$  The following convergence monitoring routines use atol

.keywords: SNES, nonlinear, set, absolute, convergence, tolerance

.seealso: SNESSetRelativeTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetAbsoluteTolerance(SNES snes,double atol)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->atol = atol;
  return 0;
}

/*@
   SNESSetTruncationTolerance - Sets the tolerance that may be used by the
   step routines to control the accuracy of the step computation.

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance

   Options Database Key: 
$    -snes_ttol tol

   Notes:
   If the step computation involves an application of the inverse
   Jacobian (or Hessian), this parameter may be used to control the 
   accuracy of that application.  In particular, this tolerance is used 
   by SNESKSPDefaultConverged() and SNESKSPQuadraticConverged() to determine
   the minimum convergence tolerance for the iterative linear solvers.

.keywords: SNES, nonlinear, set, truncation, tolerance

.seealso: SNESSetRelativeTolerance(), SNESSetSolutionTolerance(),
          SNESSetAbsoluteTolerance()
@*/
int SNESSetTruncationTolerance(SNES snes,double tol)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->trunctol = tol;
  return 0;
}

/*@
   SNESSetSolutionTolerance - Sets the convergence tolerance in terms of 
   the norm of the change in the solution between steps.

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance

   Options Database Key: 
$    -snes_stol tol

.keywords: SNES, nonlinear, set, solution, tolerance

.seealso: SNESSetTruncationTolerance(), SNESSetRelativeTolerance(),
          SNESSetAbsoluteTolerance()
@*/
int SNESSetSolutionTolerance( SNES snes, double tol )
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->xtol = tol;
  return 0;
}

/* ---------- Routines to set various aspects of nonlinear solver --------- */

/*@
   SNESSetSolution - Sets the initial guess routine and solution vector
   for use by the SNES routines.

   Input Parameters:
.  snes - the SNES context
.  x - the solution vector
.  func - optional routine to compute an initial guess (may be null)
.  ctx - optional user-defined context for private data for the 
         initial guess routine (may be null)

   Calling sequence of func:
   int guess(Vec x, void *ctx)

.  x - input vector
.  ctx - optional user-defined initial guess context 

   Note:
   If no initial guess routine is indicated, an initial guess of zero 
   will be used.

.keywords: SNES, nonlinear, set, solution, initial guess

.seealso: SNESGetSolution(), SNESSetJacobian(), SNESSetFunction()
@*/
int SNESSetSolution(SNES snes,Vec x,int (*func)(Vec,void*),void *ctx)
{
  VALIDHEADER(snes,SNES_COOKIE);
  snes->vec_sol             = x;
  snes->ComputeInitialGuess = func;
  snes->gusP                = ctx;
  return 0;
}

/* ------------ Routines to set performance monitoring options ----------- */

/*@C
   SNESSetMonitor - Sets the function that is to be used at every
   iteration of the nonlinear solver to display the iteration's 
   progress.   

   Input Parameters:
.  snes - the SNES context
.  func - monitoring routine
.  mctx - optional user-defined context for private data for the 
          monitor routine (may be null)

   Calling sequence of func:
   int func((SNES snes,int its, Vec x,Vec f,double fnorm,void *mctx)

.  snes - the SNES context
.  its - iteration number
.  x - current iterate
.  f - current residual (+/-).  f is either the residual or its negative, 
       depending on the user's preference, as set with SNESSetFunction()
.  fnorm - 2-norm residual value (may be estimated)
.  mctx - optional monitoring context

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESDefaultMonitor()
@*/
int SNESSetMonitor( SNES snes, int (*func)(SNES,int,double,void*), 
                    void *mctx )
{
  snes->Monitor = func;
  snes->monP    = (void*)mctx;
  return 0;
}

/*@C
   SNESSetConvergenceTest - Sets the function that is to be used 
   to test for convergence of the nonlinear iterative solution.   

   Input Parameters:
.  snes - the SNES context
.  func - routine to test for convergence
.  cctx - optional context for private data for the convergence routine 
          (may be null)

   Calling sequence of func:
   int func (SNES snes,double xnorm,double pnorm,double fnorm,
             void *cctx)

.  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of residual
.  cctx - optional convergence context

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESDefaultConverged()
@*/
int SNESSetConvergenceTest(SNES nlP,
          int (*func)(SNES,double,double,double,void*),void *cctx)
{
  (nlP)->Converged = func;
  (nlP)->cnvP      = cctx;
  return 0;
}

/*@
   SNESScaleStep - Scales a step so that its length is less than the
   positive parameter delta.

    Input Parameters:
.   snes - the SNES context
.   y - approximate solution of linear system
.   fnorm - 2-norm of current residual
.   delta - trust region size

    Output Parameters:
.   gpnorm - predicted residual norm at the new point, assuming local 
    linearization.  The value is zero if the step lies within the trust 
    region, and exceeds zero otherwise.
.   ynorm - 2-norm of the step

    Note:
    For non-trust region methods such as SNES_NLS, the parameter delta 
    is set to be the maximum allowable step size.  

.keywords: SNES, nonlinear, scale, step
@*/
int SNESScaleStep(SNES snes,Vec y,double *fnorm,double *delta, 
                  double *gpnorm,double *ynorm)
{
  double norm;
  Scalar cnorm;
  VecNorm(y, &norm );
  if (norm > *delta) {
     norm = *delta/norm;
     *gpnorm = (1.0 - norm)*(*fnorm);
     cnorm = norm;
     VecScale( &cnorm, y );
     *ynorm = *delta;
  } else {
     *gpnorm = 0.0;
     *ynorm = norm;
  }
  return 0;
}

/*@
   SNESSolve - Solves a nonlinear system.  Call SNESSolve after calling 
   SNESCreate(), optional routines of the form SNESSetXXX(), and SNESSetUp().

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
   its - number of iterations until termination

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESSetUp(), SNESDestroy()
@*/
int SNESSolve(SNES snes,int *its)
{
  int ierr;
  PLogEventBegin(SNES_Solve,snes,0,0,0);
  ierr = (*(snes)->Solver)( snes,its ); CHKERR(ierr);
  PLogEventEnd(SNES_Solve,snes,0,0,0);
  return 0;
}

/* --------- Internal routines for SNES Package --------- */

/*
   SNESComputeInitialGuess - Manages computation of initial approximation.
 */
int SNESComputeInitialGuess( SNES nlP,Vec  x )
{
  int    ierr;
  Scalar zero = 0.0;
  if (nlP->ComputeInitialGuess) {
    ierr = (*nlP->ComputeInitialGuess)( x, nlP->gusP); CHKERR(ierr);
  }
  else VecSet(&zero, x );
  return 0;
}

/* ------------------------------------------------------------------ */


#include "sys/nreg.h"
NRList *__NLList;

/*@
   SNESSetMethod - Sets the method for the nonlinear solver.  

   Input Parameters:
.  snes - the SNES context
.  method - choose from 

  Possible methods:
$    SNES_NLS - Newton's method with line search
$    SNES_NTR - Newton's method with trust region
@*/
int SNESSetMethod( SNES snes, SNESMETHOD method)
{
  int (*r)(SNES);
  VALIDHEADER(snes,SNES_COOKIE);
  /* Get the function pointers for the iterative method requested */
  if (!__NLList) {SNESRegisterAll();}
  if (!__NLList) {SETERR(1,"Could not acquire list of SNES methods"); }
  r =  (int (*)(SNES))NRFindRoutine( __NLList, (int)method, (char *)0 );
  if (!r) {SETERR(1,"Unknown SNES method");}
  return (*r)(snes);
}

/* --------------------------------------------------------------------- */
/*@
   SNESRegister - Adds the method to the nonlinear solver package, given 
   a function pointer and a nonlinear solver name of the type SNESMETHOD.

   Input Parameters:
.  name - for instance SNES_NLS, SNES_NTR, ...
.  sname - corresponding string for name
.  create - routine to create method context

.keywords: SNES, nonlinear, register

.seealso: SNESRegisterAll(), SNESRegisterDestroy()
@*/
int SNESRegister(int name, char *sname, int (*create)(SNES))
{
  int ierr;
  if (!__NLList) {ierr = NRCreate(&__NLList); CHKERR(ierr);}
  NRRegister( __NLList, name, sname, (int (*)(void*))create );
  return 0;
}
/* --------------------------------------------------------------------- */
/*@
   SNESRegisterDestroy - Frees the list of nonlinear solvers that were
   registered by SNESRegister().

.keywords: SNES, nonlinear, register, destroy

.seealso: SNESRegisterAll(), SNESRegisterAll()
@*/
int SNESRegisterDestroy()
{
  if (__NLList) {
    NRDestroy( __NLList );
    __NLList = 0;
  }
  return 0;
}
#include "options.h"
/*@C
   SNESGetMethodFromOptions - Sets the selected method from the options
   database.

   Input parameters:
.  ctx - the SNES context

   Output Parameter:
.  method -  solver method

   Returns:
   Returns 1 if the method is found; 0 otherwise.

   Options Database Key:
$  -snes_method  method

.keywords: SNES, nonlinear, options, database, get, method

.seealso: SNESGetMethodName()
@*/
int SNESGetMethodFromOptions(SNES ctx,SNESMETHOD *method)
{
  char sbuf[50];
  if (OptionsGetString(0,ctx->prefix,"-snes_method", sbuf, 50 )) {
    if (!__NLList) SNESRegisterAll();
    *method = (SNESMETHOD)NRFindID( __NLList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   SNESGetMethodName - Gets the SNES method name (as a string) from
   the method type.

   Input Parameter:
.  method - SNES method

   Output Parameter:
.  name - name of SNES method

.keywords: SNES, nonlinear, get, method, name

.seealso: SNESGetMethodFromOptions()
@*/
int SNESGetMethodName(SNESMETHOD method,char **name)
{
  if (!__NLList) SNESRegisterAll();
  *name = NRFindName( __NLList, (int) method );
  return 0;
}

#include <stdio.h>
/*@C
   SNESPrintMethods - Prints the SNES methods available from the options 
   database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "snesmethod") 

.keywords: SNES, nonlinear, print, methods, options, database

.seealso: SNESPrintHelp()
@*/
int SNESPrintMethods(char* prefix,char *name)
{
  FuncList *entry;
  if (!__NLList) {SNESRegisterAll();}
  entry = __NLList->head;
  fprintf(stderr," %s%s (one of)",prefix,name);
  while (entry) {
    fprintf(stderr," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stderr,"\n");
  return 0;
}

/*@
   SNESGetSolution - Returns the vector where the approximate solution is
   stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

.keywords: SNES, nonlinear, get, solution

.seealso: SNESSetSolution(), SNESGetFunction()
@*/
int SNESGetSolution(SNES snes,Vec *x)
{
  VALIDHEADER(snes,SNES_COOKIE);
  *x = snes->vec_sol;
  return 0;
}  

/*@
   SNESGetFunction - Returns the vector where the residual is
   stored.  Actually usually returns the vector where the negative of 
   the residual is stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the residual (or its negative)

.keywords: SNES, nonlinear, get residual

.seealso: SNESSetFunction(), SNESGetSolution()
@*/
int SNESGetFunction(SNES snes,Vec *r)
{
  VALIDHEADER(snes,SNES_COOKIE);
  *r = snes->vec_res;
  return 0;
}  






