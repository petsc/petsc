#ifdef PETSC_RCS_HEADER

#endif

#include "src/ts/tsimpl.h"        /*I "ts.h"  I*/

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  PetscValidHeader(x);  PetscValidHeader(y);

  if (ts->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
    PetscFunctionReturn(0);
  }

  if (ts->rhsmatrix) { /* assemble matrix for this timestep */
    MatStructure flg;
    PetscStackPush("TS user right-hand-side matrix function");
    ierr = (*ts->rhsmatrix)(ts,t,&ts->A,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
  }
  ierr = MatMult(ts->A,x,y);CHKERRQ(ierr);

  /* apply user-provided boundary conditions (only needed if these are time dependent) */
  ierr = TSComputeRHSBoundaryConditions(ts,t,y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSFunction"
/*@C
    TSSetRHSFunction - Sets the routine for evaluating the function,
    F(t,u), where U_t = F(t,u).

    Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   f - routine for evaluating the right-hand-side function
-   ctx - [optional] user-defined context for private data for the 
          function evaluation routine (may be PETSC_NULL)

    Calling sequence of func:
$     func (TS ts,double t,Vec u,Vec F,void *ctx);

+   t - current timestep
.   u - input vector
.   F - function vector
-   ctx - [optional] user-defined function context 

    Important: 
    The user MUST call either this routine or TSSetRHSMatrix().

    Level: beginner

.keywords: TS, timestep, set, right-hand-side, function

.seealso: TSSetRHSMatrix()
@*/
int TSSetRHSFunction(TS ts,int (*f)(TS,double,Vec,Vec,void*),void *ctx)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Cannot set function for linear problem");
  }
  ts->rhsfunction = f;
  ts->funP        = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSMatrix"
/*@C
   TSSetRHSMatrix - Sets the function to compute the matrix A, where U_t = A(t) U.
   Also sets the location to store A.

   Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  A   - matrix
.  B   - preconditioner matrix (usually same as A)
.  f   - the matrix evaluation routine; use PETSC_NULL (PETSC_NULL_FUNCTION in fortran)
         if A is not a function of t.
-  ctx - [optional] user-defined context for private data for the 
          matrix evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$     func (TS ts,double t,Mat *A,Mat *B,int *flag,void *ctx);

+  t - current timestep
.  A - matrix A, where U_t = A(t) U
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in SLESSetOperators())
-  ctx - [optional] user-defined context for matrix evaluation routine

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

   Level: beginner

.keywords: TS, timestep, set, right-hand-side, matrix

.seealso: TSSetRHSFunction()
@*/
int TSSetRHSMatrix(TS ts,Mat A, Mat B,int (*f)(TS,double,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type == TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not for nonlinear problems; use TSSetRHSJacobian()");
  }

  ts->rhsmatrix = f;
  ts->jacP      = ctx;
  ts->A         = A;
  ts->B         = B;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSJacobian"
/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of F,
   where U_t = F(U,t), as well as the location to store the matrix.

   Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  A   - Jacobian matrix
.  B   - preconditioner matrix (usually same as A)
.  f   - the Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$     func (TS ts,double t,Vec u,Mat *A,Mat *B,int *flag,void *ctx);

+  t - current timestep
.  u - input vector
.  A - matrix A, where U_t = A(t)u
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in SLESSetOperators())
-  ctx - [optional] user-defined context for matrix evaluation routine

   Notes: 
   See SLESSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the matrix evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner
   
.keywords: TS, timestep, set, right-hand-side, Jacobian

.seealso: TSDefaultComputeJacobianColor(),
          SNESDefaultComputeJacobianColor()

@*/
int TSSetRHSJacobian(TS ts,Mat A, Mat B,int (*f)(TS,double,Vec,Mat*,Mat*,
                     MatStructure*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not for linear problems; use TSSetRHSMatrix()");
  }

  ts->rhsjacobian = f;
  ts->jacP        = ctx;
  ts->A           = A;
  ts->B           = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSComputeRHSBoundaryConditions"
/*
   TSComputeRHSBoundaryConditions - Evaluates the boundary condition function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
int TSComputeRHSBoundaryConditions(TS ts,double t,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  PetscValidHeader(x);

  if (ts->rhsbc) {
    PetscStackPush("TS user boundary condition function");
    ierr = (*ts->rhsbc)(ts,t,x,ts->bcP);CHKERRQ(ierr);
    PetscStackPop;
    PetscFunctionReturn(0);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetRHSBoundaryConditions"
/*@C
    TSSetRHSBoundaryConditions - Sets the routine for evaluating the function,
    boundary conditions for the function F.

    Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   f - routine for evaluating the boundary condition function
-   ctx - [optional] user-defined context for private data for the 
          function evaluation routine (may be PETSC_NULL)

    Calling sequence of func:
$     func (TS ts,double t,Vec F,void *ctx);

+   t - current timestep
.   F - function vector
-   ctx - [optional] user-defined function context 

    Level: intermediate

.keywords: TS, timestep, set, boundary conditions, function
@*/
int TSSetRHSBoundaryConditions(TS ts,int (*f)(TS,double,Vec,void*),void *ctx)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type != TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For linear problems only");
  }
  ts->rhsbc = f;
  ts->bcP   = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSView"
/*@ 
    TSView - Prints the TS data structure.

    Collective on TS, unless Viewer is VIEWER_STDOUT_SELF

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_view - calls TSView() at end of TSStep()

    Notes:
    The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

    The user can open an alternative visualization context with
    ViewerASCIIOpen() - output to a specified file.

    Level: beginner

.keywords: TS, timestep, view

.seealso: ViewerASCIIOpen()
@*/
int TSView(TS ts,Viewer viewer)
{
  int                 ierr;
  char                *method;
  ViewerType          vtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = ViewerASCIIPrintf(viewer,"TS Object:\n");CHKERRQ(ierr);
    ierr = TSGetType(ts,(TSType *)&method);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  method: %s\n",method);CHKERRQ(ierr);
    if (ts->view) {
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ts->view)(ts,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = ViewerASCIIPrintf(viewer,"  maximum steps=%d\n",ts->max_steps);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  maximum time=%g\n",ts->max_time);CHKERRQ(ierr);
    if (ts->problem_type == TS_NONLINEAR) {
      ierr = ViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%d\n",ts->nonlinear_its);CHKERRQ(ierr);
    }
    ierr = ViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%d\n",ts->linear_its);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,STRING_VIEWER)) {
    ierr = TSGetType(ts,(TSType *)&method);CHKERRQ(ierr);
    ierr = ViewerStringSPrintf(viewer," %-7.7s",method);CHKERRQ(ierr);
  }
  ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (ts->sles) {ierr = SLESView(ts->sles,viewer);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
  ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "TSSetApplicationContext"
/*@C
   TSSetApplicationContext - Sets an optional user-defined context for 
   the timesteppers.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  usrP - optional user context

   Level: intermediate

.keywords: TS, timestep, set, application, context

.seealso: TSGetApplicationContext()
@*/
int TSSetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->user = usrP;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetApplicationContext"
/*@C
    TSGetApplicationContext - Gets the user-defined context for the 
    timestepper.

    Not Collective

    Input Parameter:
.   ts - the TS context obtained from TSCreate()

    Output Parameter:
.   usrP - user context

    Level: intermediate

.keywords: TS, timestep, get, application, context

.seealso: TSSetApplicationContext()
@*/
int TSGetApplicationContext( TS ts,  void **usrP )
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *usrP = ts->user;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetTimeStepNumber"
/*@
   TSGetTimeStepNumber - Gets the current number of timesteps.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  iter - number steps so far

   Level: intermediate

.keywords: TS, timestep, get, iteration, number
@*/
int TSGetTimeStepNumber(TS ts,int* iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *iter = ts->steps;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetInitialTimeStep"
/*@
   TSSetInitialTimeStep - Sets the initial timestep to be used, 
   as well as the initial time.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  initial_time - the initial time
-  time_step - the size of the timestep

   Level: intermediate

.seealso: TSSetTimeStep(), TSGetTimeStep()

.keywords: TS, set, initial, timestep
@*/
int TSSetInitialTimeStep(TS ts,double initial_time,double time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->time_step         = time_step;
  ts->initial_time_step = time_step;
  ts->ptime             = initial_time;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetTimeStep"
/*@
   TSSetTimeStep - Allows one to reset the timestep at any time,
   useful for simple pseudo-timestepping codes.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time_step - the size of the timestep

   Level: intermediate

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, set, timestep
@*/
int TSSetTimeStep(TS ts,double time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->time_step = time_step;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetTimeStep"
/*@
   TSGetTimeStep - Gets the current timestep size.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  dt - the current timestep size

   Level: intermediate

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, get, timestep
@*/
int TSGetTimeStep(TS ts,double* dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *dt = ts->time_step;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetSolution"
/*@C
   TSGetSolution - Returns the solution at the present timestep. It
   is valid to call this routine inside the function that you are evaluating
   in order to move to the new timestep. This vector not changed until
   the solution at the next timestep has been calculated.

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  v - the vector containing the solution

   Level: intermediate

.seealso: TSGetTimeStep()

.keywords: TS, timestep, get, solution
@*/
int TSGetSolution(TS ts,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  *v = ts->vec_sol_always;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPublish_Petsc"
static int TSPublish_Petsc(PetscObject object)
{
#if defined(PETSC_HAVE_AMS)
  TS   v = (TS) object;
  int  ierr;
  
  PetscFunctionBegin;

  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Step",&v->steps,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Time",&v->ptime,1,AMS_DOUBLE,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"CurrentTimeStep",&v->time_step,1,
                               AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSCreate"
/*@C
   TSCreate - Creates a timestepper context.

   Collective on MPI_Comm

   Input Parameter:
+  comm - MPI communicator
-  type - One of  TS_LINEAR,TS_NONLINEAR
   where these types refer to problems of the forms
.vb
         U_t = A U    
         U_t = A(t) U 
         U_t = F(t,U) 
.ve

   Output Parameter:
.  outts - the new TS context

   Level: beginner

.keywords: TS, timestep, create, context

.seealso: TSSetUp(), TSStep(), TSDestroy()
@*/
int TSCreate(MPI_Comm comm,TSProblemType problemtype,TS *outts)
{
  TS   ts;

  PetscFunctionBegin;
  *outts = 0;
  PetscHeaderCreate(ts,_p_TS,int,TS_COOKIE,-1,"TS",comm,TSDestroy,TSView);
  PLogObjectCreate(ts);
  ts->bops->publish     = TSPublish_Petsc;
  ts->max_steps         = 5000;
  ts->max_time          = 5.0;
  ts->time_step         = .1;
  ts->initial_time_step = ts->time_step;
  ts->steps             = 0;
  ts->ptime             = 0.0;
  ts->data              = 0;
  ts->view              = 0;
  ts->setupcalled       = 0;
  ts->problem_type      = problemtype;
  ts->numbermonitors    = 0;
  ts->linear_its        = 0;
  ts->nonlinear_its     = 0;

  *outts = ts;
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a timestepper ---- */

#undef __FUNC__  
#define __FUNC__ "TSSetUp"
/*@
   TSSetUp - Sets up the internal data structures for the later use
   of a timestepper.  

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Notes:
   For basic use of the TS solvers the user need not explicitly call
   TSSetUp(), since these actions will automatically occur during
   the call to TSStep().  However, if one wishes to control this
   phase separately, TSSetUp() should be called after TSCreate()
   and optional routines of the form TSSetXXX(), but before TSStep().  

   Level: advanced

.keywords: TS, timestep, setup

.seealso: TSCreate(), TSStep(), TSDestroy()
@*/
int TSSetUp(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (!ts->vec_sol) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call TSSetSolution() first");
  if (!ts->type_name) {
    ierr = TSSetType(ts,TS_EULER);CHKERRQ(ierr);
  }
  ierr = (*ts->setup)(ts);CHKERRQ(ierr);
  ts->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSDestroy"
/*@C
   TSDestroy - Destroys the timestepper context that was created
   with TSCreate().

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: beginner

.keywords: TS, timestepper, destroy

.seealso: TSCreate(), TSSetUp(), TSSolve()
@*/
int TSDestroy(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (--ts->refct > 0) PetscFunctionReturn(0);

  if (ts->sles) {ierr = SLESDestroy(ts->sles);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes);CHKERRQ(ierr);}
  ierr = (*(ts)->destroy)(ts);CHKERRQ(ierr);
  PLogObjectDestroy((PetscObject)ts);
  PetscHeaderDestroy((PetscObject)ts);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetSNES"
/*@C
   TSGetSNES - Returns the SNES (nonlinear solver) associated with 
   a TS (timestepper) context. Valid only for nonlinear problems.

   Not Collective, but SNES is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  snes - the nonlinear solver context

   Notes:
   The user can then directly manipulate the SNES context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   SLES, KSP, and PC contexts as well.

   TSGetSNES() does not work for integrators that do not use SNES; in
   this case TSGetSNES() returns PETSC_NULL in snes.

   Level: beginner

.keywords: timestep, get, SNES
@*/
int TSGetSNES(TS ts,SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type == TS_LINEAR) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Nonlinear only; use TSGetSLES()");
  *snes = ts->snes;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetSLES"
/*@C
   TSGetSLES - Returns the SLES (linear solver) associated with 
   a TS (timestepper) context.

   Not Collective, but SLES is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  sles - the nonlinear solver context

   Notes:
   The user can then directly manipulate the SLES context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   KSP and PC contexts as well.

   TSGetSLES() does not work for integrators that do not use SLES;
   in this case TSGetSLES() returns PETSC_NULL in sles.

   Level: beginner

.keywords: timestep, get, SLES
@*/
int TSGetSLES(TS ts,SLES *sles)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->problem_type != TS_LINEAR) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Linear only; use TSGetSNES()");
  *sles = ts->sles;
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNC__  
#define __FUNC__ "TSSetDuration"
/*@
   TSSetDuration - Sets the maximum number of timesteps to use and 
   maximum time for iteration.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  maxsteps - maximum number of iterations to use
-  maxtime - final time to iterate to

   Options Database Keys:
.  -ts_max_steps <maxsteps> - Sets maxsteps
.  -ts_max_time <maxtime> - Sets maxtime

   Notes:
   The default maximum number of iterations is 5000. Default time is 5.0

   Level: intermediate

.keywords: TS, timestep, set, maximum, iterations
@*/
int TSSetDuration(TS ts,int maxsteps,double maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->max_steps = maxsteps;
  ts->max_time  = maxtime;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetSolution"
/*@
   TSSetSolution - Sets the initial solution vector
   for use by the TS routines.

   Collective on TS and Vec

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  x - the solution vector

   Level: beginner

.keywords: TS, timestep, set, solution, initial conditions
@*/
int TSSetSolution(TS ts,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->vec_sol        = ts->vec_sol_always = x;
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNC__  
#define __FUNC__ "TSSetMonitor"
/*@C
   TSSetMonitor - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.   

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  func - monitoring routine
-  mctx - [optional] user-defined context for private data for the 
          monitor routine (may be PETSC_NULL)

   Calling sequence of func:
$    int func(TS ts,int steps,double time,Vec x,void *mctx)

+    ts - the TS context
.    steps - iteration number
.    time - current timestep
.    x - current iterate
-    mctx - [optional] monitoring context

   Notes:
   This routine adds an additional monitor to the list of monitors that 
   already has been loaded.

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSDefaultMonitor(), TSClearMonitor()
@*/
int TSSetMonitor(TS ts, int (*monitor)(TS,int,double,Vec,void*), void *mctx )
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->numbermonitors >= MAXTSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many monitors set");
  }
  ts->monitor[ts->numbermonitors]           = monitor;
  ts->monitorcontext[ts->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSClearMonitor"
/*@C
   TSClearMonitor - Clears all the monitors that have been set on a time-step object.   

   Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSDefaultMonitor(), TSSetMonitor()
@*/
int TSClearMonitor(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ts->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSDefaultMonitor"
int TSDefaultMonitor(TS ts, int step, double time,Vec v, void *ctx)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(ts->comm,"timestep %d dt %g time %g\n",step,ts->time_step,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSStep"
/*@
   TSStep - Steps the requested number of timesteps.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  steps - number of iterations until termination
-  time - time until termination

   Level: beginner

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetUp(), TSDestroy()
@*/
int TSStep(TS ts,int *steps,double *time)
{
  int ierr,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (!ts->setupcalled) {ierr = TSSetUp(ts);CHKERRQ(ierr);}
  PLogEventBegin(TS_Step,ts,0,0,0);
  ierr = (*(ts)->step)(ts,steps,time);CHKERRQ(ierr);
  PLogEventEnd(TS_Step,ts,0,0,0);
  ierr = OptionsHasName(PETSC_NULL,"-ts_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TSView(ts,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSMonitor"
/*
     Runs the user provided monitor routines, if they exists.
*/
int TSMonitor(TS ts,int step,double time,Vec x)
{
  int i,ierr,n = ts->numbermonitors;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    ierr = (*ts->monitor[i])(ts,step,time,x,ts->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------*/

/*@C
   TSLGMonitorCreate - Creates a line graph context for use with 
   TS to monitor convergence of preconditioned residual norms.

   Collective on TS

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
.  -ts_xmonitor - automatically sets line graph monitor

   Notes: 
   Use TSLGMonitorDestroy() to destroy this line graph, not DrawLGDestroy().

   Level: intermediate

.keywords: TS, monitor, line graph, residual, create

.seealso: TSLGMonitorDestroy(), TSSetMonitor()
@*/
int TSLGMonitorCreate(char *host,char *label,int x,int y,int m,
                       int n, DrawLG *draw)
{
  Draw win;
  int  ierr;

  PetscFunctionBegin;
  ierr = DrawOpenX(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = DrawLGCreate(win,1,draw);CHKERRQ(ierr);
  ierr = DrawLGIndicateDataPoints(*draw);CHKERRQ(ierr);

  PLogObjectParent(*draw,win);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSLGMonitor"
int TSLGMonitor(TS ts,int n,double time,Vec v,void *monctx)
{
  DrawLG lg = (DrawLG) monctx;
  double x,y = time;
  int    ierr;

  PetscFunctionBegin;
  if (!n) {ierr = DrawLGReset(lg);CHKERRQ(ierr);}
  x = (double) n;
  ierr = DrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || (n % 5)) {
    ierr = DrawLGDraw(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "TSLGMonitorDestroy" 
/*@C
   TSLGMonitorDestroy - Destroys a line graph context that was created 
   with TSLGMonitorCreate().

   Collective on DrawLG

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: TS, monitor, line graph, destroy

.seealso: TSLGMonitorCreate(),  TSSetMonitor(), TSLGMonitor();
@*/
int TSLGMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  int  ierr;

  PetscFunctionBegin;
  ierr = DrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  ierr = DrawDestroy(draw);CHKERRQ(ierr);
  ierr = DrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetTime"
/*@
   TSGetTime - Gets the current time.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  t  - the current time

   Contributed by: Matthew Knepley

   Level: beginner

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, get, time
@*/
int TSGetTime(TS ts, double* t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  *t = ts->ptime;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetProblemType" 
/*@C
   TSGetProblemType - Returns the problem type of a TS (timestepper) context.

   Not Collective

   Input Parameter:
.  ts   - The TS context obtained from TSCreate()

   Output Parameter:
.  type - The problem type, TS_LINEAR or TS_NONLINEAR

   Level: intermediate

   Contributed by: Matthew Knepley

.keywords: ts, get, type

@*/
int TSGetProblemType(TS ts, TSProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  *type = ts->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "TSSetOptionsPrefix" 
/*@C
   TSSetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Collective on TS

   Input Parameter:
+  ts     - The TS context
-  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Contributed by: Matthew Knepley

   Level: advanced

.keywords: TS, set, options, prefix, database

.seealso: TSSetFromOptions()

@*/
int TSSetOptionsPrefix(TS ts, char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ts, prefix);CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESSetOptionsPrefix(ts->snes, prefix);CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = SLESSetOptionsPrefix(ts->sles, prefix);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "TSAppendOptionsPrefix" 
/*@C
   TSAppendOptionsPrefix - Appends to the prefix used for searching for all
   TS options in the database.

   Collective on TS

   Input Parameter:
+  ts     - The TS context
-  prefix - The prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Contributed by: Matthew Knepley

   Level: advanced

.keywords: TS, append, options, prefix, database

.seealso: TSGetOptionsPrefix()

@*/
int TSAppendOptionsPrefix(TS ts, char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject) ts, prefix);CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESAppendOptionsPrefix(ts->snes, prefix);CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = SLESAppendOptionsPrefix(ts->sles, prefix);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "TSGetOptionsPrefix"
/*@C
   TSGetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Not Collective

   Input Parameter:
.  ts - The TS context

   Output Parameter:
.  prefix - A pointer to the prefix string used

   Contributed by: Matthew Knepley

   Notes: On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: intermediate

.keywords: TS, get, options, prefix, database

.seealso: TSAppendOptionsPrefix()
@*/
int TSGetOptionsPrefix(TS ts, char **prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ts, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "TSGetRHSMatrix"
/*@C
   TSGetRHSMatrix - Returns the matrix A at the present timestep.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  A   - The matrix A, where U_t = A(t) U
.  M   - The preconditioner matrix, usually the same as A
-  ctx - User-defined context for matrix evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Contributed by: Matthew Knepley

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetTime(), TSGetTimeStepNumber(), TSGetRHSJacobian()

.keywords: TS, timestep, get, matrix

@*/
int TSGetRHSMatrix(TS ts, Mat *A, Mat *M, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE);
  if (A)   *A = ts->A;
  if (M)   *M = ts->B;
  if (ctx) *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "TSGetRHSJacobian"
/*@C
   TSGetRHSJacobian - Returns the Jacobian J at the present timestep.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  J   - The Jacobian J of F, where U_t = F(U,t)
.  M   - The preconditioner matrix, usually the same as J
- ctx - User-defined context for Jacobian evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Contributed by: Matthew Knepley

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetRHSMatrix(), TSGetTime(), TSGetTimeStepNumber()

.keywords: TS, timestep, get, matrix, Jacobian
@*/
int TSGetRHSJacobian(TS ts, Mat *J, Mat *M, void **ctx)
{
  int ierr;

  PetscFunctionBegin;
  ierr = TSGetRHSMatrix(ts, J, M, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   TSRegister - Adds a method to the timestepping solver package.

   Synopsis:

   TSRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(TS))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   TSRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   TSRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
              "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TSSetType(ts,"my_solver")
   or at runtime via the option
$     -ts_type my_solver

   Level: advanced

   $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.keywords: TS, register

.seealso: TSRegisterAll(), TSRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "TSRegister_Private"
int TSRegister_Private(char *sname,char *path,char *name,int (*function)(TS))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname,path);CHKERRQ(ierr);
  PetscStrcat(fullname,":"); PetscStrcat(fullname,name);
  FListAdd_Private(&TSList,sname,fullname,        (int (*)(void*))function);
  PetscFunctionReturn(0);
}
