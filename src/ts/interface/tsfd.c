#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ts.c,v 1.21 1997/09/11 20:40:48 bsmith Exp $";
#endif

#include "src/ts/tsimpl.h"        /*I "ts.h"  I*/
#include "pinclude/pviewer.h"
#include <math.h>

extern int TSGetTypeFromOptions_Private(TS,TSType*,int*);
extern int TSPrintTypes_Private(MPI_Comm,char*,char*);

#undef __FUNC__  
#define __FUNC__ "TSComputeRHSFunction"
/*
   TSComputeRHSFunction - Evaluates the right-hand-side function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
int TSComputeRHSFunction(TS ts,double t,Vec x, Vec y)
{
  int ierr;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  PetscValidHeader(x);  PetscValidHeader(y);

  if (ts->rhsfunction) return (*ts->rhsfunction)(ts,t,x,y,ts->funP);

  if (ts->rhsmatrix) { /* assemble matrix for this timestep */
    MatStructure flg;
    ierr = (*ts->rhsmatrix)(ts,t,&ts->A,&ts->B,&flg,ts->jacP); CHKERRQ(ierr);
  }
  ierr = MatMult(ts->A,x,y); CHKERRQ(ierr);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSFunction"
/*@C
    TSSetRHSFunction - Sets the routine for evaluating the function,
    F(t,u), where U_t = F(t,u).

    Input Parameters:
.   ts - the TS context obtained from TSCreate()
.   f - routine for evaluating the right-hand-side function
.   ctx - [optional] user-defined context for private data for the 
          function evaluation routine (may be PETSC_NULL)

    Calling sequence of func:
.   func (TS ts,double t,Vec u,Vec F,void *ctx);

.   t - current timestep
.   u - input vector
.   F - function vector
.   ctx - [optional] user-defined function context 

    Important: 
    The user MUST call either this routine or TSSetRHSMatrix().

.keywords: TS, timestep, set, right-hand-side, function

.seealso: TSSetRHSMatrix()
@*/
int TSSetRHSFunction(TS ts,int (*f)(TS,double,Vec,Vec,void*),void *ctx)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type == TS_LINEAR) SETERRQ(1,0,"Cannot set function for linear problem");
  ts->rhsfunction = f;
  ts->funP        = ctx;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSMatrix"
/*@C
   TSSetRHSMatrix - Sets the function to compute the matrix A, where U_t = A(t) U.
   Also sets the location to store A.

   Input Parameters:
.  ts  - the TS context obtained from TSCreate()
.  A   - matrix
.  B   - preconditioner matrix (usually same as A)
.  f   - the matrix evaluation routine; use PETSC_NULL if A is not a function of t.
.  ctx - [optional] user-defined context for private data for the 
          matrix evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
.  func (TS ts,double t,Mat *A,Mat *B,int *flag,void *ctx);

.  t - current timestep
.  A - matrix A, where U_t = A(t) U
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in SLESSetOperators())
.  ctx - [optional] user-defined context for matrix evaluation routine

   Notes: 
   See SLESSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the matrix evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Important: 
   The user MUST call either this routine or TSSetRHSFunction().

.keywords: TS, timestep, set, right-hand-side, matrix

.seealso: TSSetRHSFunction()
@*/
int TSSetRHSMatrix(TS ts,Mat A, Mat B,int (*f)(TS,double,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->rhsmatrix = f;
  ts->jacP      = ctx;
  ts->A         = A;
  ts->B         = B;

  if (ts->problem_type == TS_NONLINEAR) SETERRQ(1,0,"Not for nonlinear problems");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSJacobian"
/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of F,
   where U_t = F(U,t), as well as the location to store the matrix.

   Input Parameters:
.  ts  - the TS context obtained from TSCreate()
.  A   - Jacobian matrix
.  B   - preconditioner matrix (usually same as A)
.  f   - the Jacobian evaluation routine
.  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
.  func (TS ts,double t,Vec u,Mat *A,Mat *B,int *flag,void *ctx);

.  t - current timestep
.  u - input vector
.  A - matrix A, where U_t = A(t)u
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in SLESSetOperators())
.  ctx - [optional] user-defined context for matrix evaluation routine

   Notes: 
   See SLESSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the matrix evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

.keywords: TS, timestep, set, right-hand-side, Jacobian
@*/
int TSSetRHSJacobian(TS ts,Mat A, Mat B,int (*f)(TS,double,Vec,Mat*,Mat*,
                     MatStructure*,void*),void *ctx)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->rhsjacobian = f;
  ts->jacP        = ctx;
  ts->A           = A;
  ts->B           = B;
  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(1,0,"Not for linear problems");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSView"
/*@ 
    TSView - Prints the TS data structure.

    Input Parameters:
.   ts - the TS context obtained from TSCreate()
.   viewer - visualization context

    Options Database Key:
$   -ts_view : calls TSView() at end of TSStep()

    Notes:
    The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

    The user can open alternative visualization contexts with
$     ViewerFileOpenASCII() - output to a specified file

.keywords: TS, timestep, view

.seealso: ViewerFileOpenASCII()
@*/
int TSView(TS ts,Viewer viewer)
{
  FILE                *fd;
  int                 ierr;
  char                *method;
  ViewerType          vtype;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(ts->comm,fd,"TS Object:\n");
    TSGetType(ts,PETSC_NULL,&method);
    PetscFPrintf(ts->comm,fd,"  method: %s\n",method);
    if (ts->view) (*ts->view)((PetscObject)ts,viewer);
    PetscFPrintf(ts->comm,fd,"  maximum steps=%d\n",ts->max_steps);
    PetscFPrintf(ts->comm,fd,"  maximum time=%g\n",ts->max_time);
    if (ts->problem_type == TS_NONLINEAR)
      PetscFPrintf(ts->comm,fd,"  total number of nonlinear solver iterations=%d\n",ts->nonlinear_its);
    PetscFPrintf(ts->comm,fd,"  total number of linear solver iterations=%d\n",ts->linear_its);
  } else if (vtype == STRING_VIEWER) {
    TSGetType(ts,PETSC_NULL,&method);
    ViewerStringSPrintf(viewer," %-7.7s",method);
  }
  if (ts->sles) SLESView(ts->sles,viewer);
  if (ts->snes) SNESView(ts->snes,viewer);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions"
/*@
   TSSetFromOptions - Sets various TS parameters from user options.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

.keywords: TS, timestep, set, options, database

.seealso: TSPrintHelp()
@*/
int TSSetFromOptions(TS ts)
{
  int    ierr,flg,loc[4],nmax;
  TSType method;

  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->setup_called)SETERRQ(1,0,"Call prior to TSSetUp!");
  ierr = TSGetTypeFromOptions_Private(ts,&method,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = TSSetType(ts,method); CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg)  TSPrintHelp(ts);
  ierr = OptionsGetInt(ts->prefix,"-ts_max_steps",&ts->max_steps,&flg);CHKERRQ(ierr);
  ierr = OptionsGetDouble(ts->prefix,"-ts_max_time",&ts->max_time,&flg);CHKERRQ(ierr);
  ierr = OptionsHasName(ts->prefix,"-ts_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    TSSetMonitor(ts,TSDefaultMonitor,0);
  }
  nmax = 4;
  ierr = OptionsGetIntArray(ts->prefix,"-ts_xmonitor",loc,&nmax,&flg); CHKERRQ(ierr);
  if (flg) {
    int    rank = 0;
    DrawLG lg;
    MPI_Comm_rank(ts->comm,&rank);
    if (!rank) {
      ierr = TSLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); CHKERRQ(ierr);
      PLogObjectParent(ts,(PetscObject) lg);
      TSSetMonitor(ts,TSLGMonitor,(void *)lg);
    }
  }
  if (!ts->setfromoptions) return 0;
  return (*ts->setfromoptions)(ts);
}

#undef __FUNC__  
#define __FUNC__ "TSPrintHelp"
/*@
   TSPrintHelp - Prints all options for the TS (timestepping) component.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
$  -help, -h

.keywords: TS, timestep, print, help

.seealso: TSSetFromOptions()
@*/
int TSPrintHelp(TS ts)
{
  char    *prefix = "-";

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->prefix) prefix = ts->prefix;
  PetscPrintf(ts->comm,"TS options --------------------------------------------------\n");
  TSPrintTypes_Private(ts->comm,prefix,"ts_type");  
  PetscPrintf(ts->comm," %sts_monitor: use default TS monitor\n",prefix);
  PetscPrintf(ts->comm," %sts_view: view TS info after each solve\n",prefix);

  PetscPrintf(ts->comm," %sts_max_steps <steps>: maximum steps, defaults to %d\n",prefix,ts->max_steps);
  PetscPrintf(ts->comm," %sts_max_time <steps>: maximum time, defaults to %g\n",prefix,ts->max_time);
  if (ts->printhelp) (*ts->printhelp)(ts,prefix);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetApplicationContext"
/*@C
   TSSetApplicationContext - Sets an optional user-defined context for 
   the timesteppers.

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  usrP - optional user context

.keywords: TS, timestep, set, application, context

.seealso: TSGetApplicationContext()
@*/
int TSSetApplicationContext(TS ts,void *usrP)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->user = usrP;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetApplicationContext"
/*@C
    TSGetApplicationContext - Gets the user-defined context for the 
    timestepper.

    Input Parameter:
.   ts - the TS context obtained from TSCreate()

    Output Parameter:
.   usrP - user context

.keywords: TS, timestep, get, application, context

.seealso: TSSetApplicationContext()
@*/
int TSGetApplicationContext( TS ts,  void **usrP )
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *usrP = ts->user;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetTimeStepNumber"
/*@
   TSGetTimeStepNumber - Gets the current number of timesteps.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  iter - number steps so far

.keywords: TS, timestep, get, iteration, number
@*/
int TSGetTimeStepNumber(TS ts,int* iter)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *iter = ts->steps;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetInitialTimeStep"
/*@
   TSSetInitialTimeStep - Sets the initial timestep to be used, 
   as well as the initial time.

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  initial_time - the initial time
.  time_step - the size of the timestep

.seealso: TSSetTimeStep(), TSGetTimeStep()

.keywords: TS, set, initial, timestep
@*/
int TSSetInitialTimeStep(TS ts,double initial_time,double time_step)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->time_step         = time_step;
  ts->initial_time_step = time_step;
  ts->ptime             = initial_time;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetTimeStep"
/*@
   TSSetTimeStep - Allows one to reset the timestep at any time,
   useful for simple pseudo-timestepping codes.

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  time_step - the size of the timestep

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, set, timestep
@*/
int TSSetTimeStep(TS ts,double time_step)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->time_step = time_step;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetTimeStep"
/*@
   TSGetTimeStep - Gets the current timestep size.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  dt - the current timestep size

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, get, timestep
@*/
int TSGetTimeStep(TS ts,double* dt)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *dt = ts->time_step;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetSolution"
/*@C
   TSGetSolution - Returns the solution at the present timestep. It
   is valid to call this routine inside the function that you are evaluating
   in order to move to the new timestep. This vector not changed until
   the solution at the next timestep has been calculated.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  v - the vector containing the solution

.seealso: TSGetTimeStep()

.keywords: TS, timestep, get, solution
@*/
int TSGetSolution(TS ts,Vec *v)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *v = ts->vec_sol_always;
  return 0;
}

/* -----------------------------------------------------------*/
extern int TSCreate_Euler(TS);

#undef __FUNC__  
#define __FUNC__ "TSCreate"
/*@C
   TSCreate - Creates a timestepper context.

   Input Parameter:
.  comm - MPI communicator
.  type - One of  TS_LINEAR,TS_NONLINEAR

$  The types refer to the problems
$         U_t = A U
$         U_t = A(t) U
$         U_t = F(t,U) 

   Output Parameter:
.  outts - the new TS context

  Options Database Command:
$ -ts_type  <method>
$    Use -help for a list of available methods
$    (for instance, Euler)

.keywords: TS, timestep, create, context

.seealso: TSSetUp(), TSStep(), TSDestroy()
@*/
int TSCreate(MPI_Comm comm,TSProblemType problemtype,TS *outts)
{
  int  ierr;
  TS   ts;

  *outts = 0;
  PetscHeaderCreate(ts,_p_TS,TS_COOKIE,TS_EULER,comm,TSDestroy,TSView);
  PLogObjectCreate(ts);
  ts->max_steps         = 5000;
  ts->max_time          = 5.0;
  ts->time_step         = .1;
  ts->initial_time_step = ts->time_step;
  ts->steps             = 0;
  ts->ptime             = 0.0;
  ts->data              = 0;
  ts->view              = 0;
  ts->setup_called      = 0;
  ts->problem_type      = problemtype;
  ts->numbermonitors    = 0;
  ts->linear_its        = 0;
  ts->nonlinear_its     = 0;

  ierr = TSCreate_Euler(ts); CHKERRQ(ierr);
  *outts = ts;
  return 0;
}

/* ----- Routines to initialize and destroy a timestepper ---- */

#undef __FUNC__  
#define __FUNC__ "TSSetUp"
/*@
   TSSetUp - Sets up the internal data structures for the later use
   of a timestepper.  Call TSSetUp() after calling TSCreate()
   and optional routines of the form TSSetXXX(), but before calling 
   TSStep().  

   Input Parameter:
.  ts - the TS context
.   ts - the TS context obtained from TSCreate()

.keywords: TS, timestep, setup

.seealso: TSCreate(), TSSolve(), TSDestroy()
@*/
int TSSetUp(TS ts)
{
  int ierr;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (!ts->vec_sol) SETERRQ(1,0,"Must call TSSetSolution() first");
  ierr = (*ts->setup)(ts); CHKERRQ(ierr);
  ts->setup_called = 1;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSDestroy"
/*@C
   TSDestroy - Destroys the timestepper context that was created
   with TSCreate().

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

.keywords: TS, timestepper, destroy

.seealso: TSCreate(), TSSetUp(), TSSolve()
@*/
int TSDestroy(TS ts)
{
  int ierr;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (--ts->refct > 0) return 0;

  if (ts->sles) {ierr = SLESDestroy(ts->sles); CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes); CHKERRQ(ierr);}
  ierr = (*(ts)->destroy)((PetscObject)ts); CHKERRQ(ierr);
  PLogObjectDestroy((PetscObject)ts);
  PetscHeaderDestroy((PetscObject)ts);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetSNES"
/*@C
   TSGetSNES - Returns the SNES (nonlinear solver) associated with 
   a TS (timestepper) context. Valid only for nonlinear problems.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  snes - the nonlinear solver context

.keywords: timestep, get, SNES
@*/
int TSGetSNES(TS ts,SNES *snes)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type == TS_LINEAR) SETERRQ(1,0,"Nonlinear only");
  *snes = ts->snes;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetSLES"
/*@C
   TSGetSLES - Returns the SLES (linear solver) associated with 
   a TS (timestepper) context.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  sles - the nonlinear solver context

.keywords: timestep, get, SLES
@*/
int TSGetSLES(TS ts,SLES *sles)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type != TS_LINEAR) SETERRQ(1,0,"Linear only");
  *sles = ts->sles;
  return 0;
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNC__  
#define __FUNC__ "TSSetDuration"
/*@
   TSSetDuration - Sets the maximum number of timesteps to use and 
   maximum time for iteration.

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  maxsteps - maximum number of iterations to use
.  maxtime - final time to iterate to

   Options Database Keys:
$   -ts_max_steps <maxsteps>
$   -ts_max_time <maxtime>

   Notes:
   The default maximum number of iterations is 5000. Default time is 5.0

.keywords: TS, timestep, set, maximum, iterations
@*/
int TSSetDuration(TS ts,int maxsteps,double maxtime)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->max_steps = maxsteps;
  ts->max_time  = maxtime;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSSetSolution"
/*@
   TSSetSolution - Sets the initial solution vector
   for use by the TS routines.

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  x - the solution vector

.keywords: TS, timestep, set, solution, initial conditions
@*/
int TSSetSolution(TS ts,Vec x)
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->vec_sol        = ts->vec_sol_always = x;
  return 0;
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNC__  
#define __FUNC__ "TSSetMonitor"
/*@C
   TSSetMonitor - Sets the function that is to be used at every
   timestep to display the iteration's  progress.   

   Input Parameters:
.  ts - the TS context obtained from TSCreate()
.  func - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
          monitor routine (may be PETSC_NULL)

   Calling sequence of func:
   int func(TS ts,int steps,double time,Vec x,void *mctx)

.    ts - the TS context
.    steps - iteration number
.    time - current timestep
.    x - current iterate
.    mctx - [optional] monitoring context

.keywords: TS, timestep, set, monitor

.seealso: TSDefaultMonitor()
@*/
int TSSetMonitor(TS ts, int (*monitor)(TS,int,double,Vec,void*), void *mctx )
{
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  if (!monitor) {
    ts->numbermonitors = 0;
    return 0;
  }
  if (ts->numbermonitors >= MAXTSMONITORS) {
    SETERRQ(1,0,"Too many monitors set");
  }
  ts->monitor[ts->numbermonitors]           = monitor;
  ts->monitorcontext[ts->numbermonitors++]  = (void*)mctx;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSDefaultMonitor"
int TSDefaultMonitor(TS ts, int step, double time,Vec v, void *ctx)
{
  PetscPrintf(ts->comm,"timestep %d dt %g time %g\n",step,ts->time_step,time);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSStep"
/*@
   TSStep - Steps the requested number of timesteps.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
.  steps - number of iterations until termination
.  time - time until termination

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetUp(), TSDestroy()
@*/
int TSStep(TS ts,int *steps,double *time)
{
  int ierr,flg;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (!ts->setup_called) {ierr = TSSetUp(ts); CHKERRQ(ierr);}
  PLogEventBegin(TS_Step,ts,0,0,0);
  ierr = (*(ts)->step)(ts,steps,time); CHKERRQ(ierr);
  PLogEventEnd(TS_Step,ts,0,0,0);
  ierr = OptionsHasName(PETSC_NULL,"-ts_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = TSView(ts,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSMonitor"
/*
     Runs the user provided monitor routines, if they exists.
*/
int TSMonitor(TS ts,int step,double time,Vec x)
{
  int i,ierr,n = ts->numbermonitors;
  for ( i=0; i<n; i++ ) {
    ierr = (*ts->monitor[i])(ts,step,time,x,ts->monitorcontext[i]);CHKERRQ(ierr);
  }
  return 0;
}

/* ------------------------------------------------------------------------*/

/*@C
   TSLGMonitorCreate - Creates a line graph context for use with 
   TS to monitor convergence of preconditioned residual norms.

   Input Parameters:
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
.  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
$    -ts_xmonitor : automatically sets line graph monitor

   Notes: 
   Use TSLGMonitorDestroy() to destroy this line graph, not DrawLGDestroy().

.keywords: TS, monitor, line graph, residual, create

.seealso: TSLGMonitorDestroy(), TSSetMonitor()
@*/
int TSLGMonitorCreate(char *host,char *label,int x,int y,int m,
                       int n, DrawLG *draw)
{
  Draw win;
  int  ierr;
  ierr = DrawOpenX(PETSC_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,1,draw); CHKERRQ(ierr);
  ierr = DrawLGIndicateDataPoints(*draw); CHKERRQ(ierr);

  PLogObjectParent(*draw,win);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSLGMonitor"
int TSLGMonitor(TS ts,int n,double time,Vec v,void *monctx)
{
  DrawLG lg = (DrawLG) monctx;
  double x,y = time;

  if (!n) DrawLGReset(lg);
  x = (double) n;
  DrawLGAddPoint(lg,&x,&y);
  if (n < 20 || (n % 5)) {
    DrawLGDraw(lg);
  }
  return 0;
} 

#undef __FUNC__  
#define __FUNC__ "TSLGMonitorDestroy" 
/*@C
   TSLGMonitorDestroy - Destroys a line graph context that was created 
   with TSLGMonitorCreate().

   Input Parameter:
.  draw - the drawing context

.keywords: TS, monitor, line graph, destroy

.seealso: TSLGMonitorCreate(),  TSSetMonitor(), TSLGMonitor();
@*/
int TSLGMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  DrawLGGetDraw(drawlg,&draw);
  DrawDestroy(draw);
  DrawLGDestroy(drawlg);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "TSGetTime"
/*@
   TSGetTime - Gets the current time.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  t  - the current time

   Contributed by: Matthew Knepley

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, get, time
@*/
int TSGetTime(TS ts, double* t)
{
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  *t = ts->ptime;
  return(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetProblemType" 
/*@C
   TSGetProblemType - Returns the problem type of
   a TS (timestepper) context.

   Input Parameter:
.  ts   - The TS context obtained from TSCreate()

   Output Parameter:
.  type - The problem type, TS_LINEAR or TS_NONLINEAR

   Contributed by: Matthew Knepley

.keywords: ts, get, type

@*/
int TSGetProblemType(TS ts, TSProblemType *type)
{
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  *type = ts->problem_type;
  return(0);
}

#undef __FUNC__
#define __FUNC__ "TSSetOptionsPrefix" 
/*@C
   TSSetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Input Parameter:
.  ts     - The TS context
.  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Contributed by: Matthew Knepley

.keywords: TS, set, options, prefix, database

.seealso: TSSetFromOptions()

@*/
int TSSetOptionsPrefix(TS ts, char *prefix)
{
  int ierr;

  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ts, prefix); CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESSetOptionsPrefix(ts->snes, prefix);              CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = SLESSetOptionsPrefix(ts->sles, prefix);              CHKERRQ(ierr);
      break;
  }
  return(0);
}


#undef __FUNC__
#define __FUNC__ "TSAppendOptionsPrefix" 
/*@C
   TSAppendOptionsPrefix - Appends to the prefix used for searching for all
   TS options in the database.

   Input Parameter:
.  ts     - The TS context
.  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Contributed by: Matthew Knepley

.keywords: TS, append, options, prefix, database

.seealso: TSGetOptionsPrefix()

@*/
int TSAppendOptionsPrefix(TS ts, char *prefix)
{
  int ierr;

  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject) ts, prefix); CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESAppendOptionsPrefix(ts->snes, prefix);              CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = SLESAppendOptionsPrefix(ts->sles, prefix);              CHKERRQ(ierr);
      break;
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "TSGetOptionsPrefix"
/*@
   TSGetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Input Parameter:
.  ts - The TS context

   Output Parameter:
.  prefix - A pointer to the prefix string used

   Contributed by: Matthew Knepley

.keywords: TS, get, options, prefix, database

.seealso: TSAppendOptionsPrefix()

@*/
int TSGetOptionsPrefix(TS ts, char **prefix)
{
  int ierr;

  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ts, prefix); CHKERRQ(ierr);
  return(0);
}

#undef __FUNC__
#define __FUNC__ "TSGetRHSMatrix"
/*@C
   TSGetRHSMatrix - Returns the matrix A at the present timestep.

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
.  A   - The matrix A, where U_t = A(t) U
.  M   - The preconditioner matrix, usually the same as A
.  ctx - User-defined context for matrix evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Contributed by: Matthew Knepley

.seealso: TSGetTimeStep(), TSGetTime(), TSGetTimeStepNumber(), TSGetRHSJacobian()

.keywords: TS, timestep, get, matrix

@*/
int TSGetRHSMatrix(TS ts, Mat *A, Mat *M, void **ctx)
{
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  if (A)   *A = ts->A;
  if (M)   *M = ts->B;
  if (ctx) *ctx = ts->jacP;
  return 0;
}

#undef __FUNC__
#define __FUNC__ "TSGetRHSJacobian"
/*@C
   TSGetRHSJacobian - Returns the Jacobian J at the present timestep.

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
.  J   - The Jacobian J of F, where U_t = F(U,t)
.  M   - The preconditioner matrix, usually the same as J
.  ctx - User-defined context for Jacobian evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Contributed by: Matthew Knepley

.seealso: TSGetTimeStep(), TSGetRHSMatrix(), TSGetTime(), TSGetTimeStepNumber()

.keywords: TS, timestep, get, matrix, Jacobian
@*/
int TSGetRHSJacobian(TS ts, Mat *J, Mat *M, void **ctx)
{
  return TSGetRHSMatrix(ts, J, M, ctx);
}
