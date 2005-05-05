#define PETSCTS_DLL

#include "src/ts/tsimpl.h"        /*I "petscts.h"  I*/

/* Logging support */
PetscCookie PETSCTS_DLLEXPORT TS_COOKIE = 0;
PetscEvent  TS_Step = 0, TS_PseudoComputeTimeStep = 0, TS_FunctionEval = 0, TS_JacobianEval = 0;

#undef __FUNCT__  
#define __FUNCT__ "TSSetTypeFromOptions"
/*
  TSSetTypeFromOptions - Sets the type of ts from user options.

  Collective on TS

  Input Parameter:
. ts - The ts

  Level: intermediate

.keywords: TS, set, options, database, type
.seealso: TSSetFromOptions(), TSSetType()
*/
static PetscErrorCode TSSetTypeFromOptions(TS ts)
{
  PetscTruth     opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->type_name) {
    defaultType = ts->type_name;
  } else {
    defaultType = TS_EULER;
  }

  if (!TSRegisterAllCalled) {
    ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-ts_type", "TS method"," TSSetType", TSList, defaultType, typeName, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSSetType(ts, typeName);CHKERRQ(ierr);
  } else {
    ierr = TSSetType(ts, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions"
/*@
   TSSetFromOptions - Sets various TS parameters from user options.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
+  -ts_type <type> - TS_EULER, TS_BEULER, TS_SUNDIALS, TS_PSEUDO, TS_CRANK_NICHOLSON
.  -ts_max_steps maxsteps - maximum number of time-steps to take
.  -ts_max_time time - maximum time to compute to
.  -ts_dt dt - initial time step
.  -ts_monitor - print information at each timestep
-  -ts_xmonitor - plot information at each timestep

   Level: beginner

.keywords: TS, timestep, set, options, database

.seealso: TSGetType
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetFromOptions(TS ts)
{
  PetscReal      dt;
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ierr = PetscOptionsBegin(ts->comm, ts->prefix, "Time step options", "TS");CHKERRQ(ierr);

  /* Handle generic TS options */
  ierr = PetscOptionsInt("-ts_max_steps","Maximum number of time steps","TSSetDuration",ts->max_steps,&ts->max_steps,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_max_time","Time to run to","TSSetDuration",ts->max_time,&ts->max_time,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_init_time","Initial time","TSSetInitialTime", ts->ptime, &ts->ptime, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_dt","Initial time step","TSSetInitialTimeStep",ts->initial_time_step,&dt,&opt);CHKERRQ(ierr);
  if (opt) {
    ts->initial_time_step = ts->time_step = dt;
  }

  /* Monitor options */
    ierr = PetscOptionsName("-ts_monitor","Monitor timestep size","TSDefaultMonitor",&opt);CHKERRQ(ierr);
    if (opt) {
      ierr = TSSetMonitor(ts,TSDefaultMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ts_xmonitor","Monitor timestep size graphically","TSLGMonitor",&opt);CHKERRQ(ierr);
    if (opt) {
      ierr = TSSetMonitor(ts,TSLGMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ts_vecmonitor","Monitor solution graphically","TSVecViewMonitor",&opt);CHKERRQ(ierr);
    if (opt) {
      ierr = TSSetMonitor(ts,TSVecViewMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }

  /* Handle TS type options */
  ierr = TSSetTypeFromOptions(ts);CHKERRQ(ierr);

  /* Handle specific TS options */
  if (ts->ops->setfromoptions) {
    ierr = (*ts->ops->setfromoptions)(ts);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Handle subobject options */
  switch(ts->problem_type) {
    /* Should check for implicit/explicit */
  case TS_LINEAR:
    if (ts->ksp) {
      ierr = KSPSetOperators(ts->ksp,ts->A,ts->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ts->ksp);CHKERRQ(ierr);
    }
    break;
  case TS_NONLINEAR:
    if (ts->snes) {
      /* this is a bit of a hack, but it gets the matrix information into SNES earlier
         so that SNES and KSP have more information to pick reasonable defaults
         before they allow users to set options */
      ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,0,ts);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid problem type: %d", (int)ts->problem_type);
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "TSViewFromOptions"
/*@
  TSViewFromOptions - This function visualizes the ts based upon user options.

  Collective on TS

  Input Parameter:
. ts - The ts

  Level: intermediate

.keywords: TS, view, options, database
.seealso: TSSetFromOptions(), TSView()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSViewFromOptions(TS ts,const char title[])
{
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscTruth     opt;
  char           typeName[1024];
  char           fileName[PETSC_MAX_PATH_LEN];
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(ts->prefix, "-ts_view", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscOptionsGetString(ts->prefix, "-ts_view", typeName, 1024, &opt);CHKERRQ(ierr);
    ierr = PetscStrlen(typeName, &len);CHKERRQ(ierr);
    if (len > 0) {
      ierr = PetscViewerCreate(ts->comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, typeName);CHKERRQ(ierr);
      ierr = PetscOptionsGetString(ts->prefix, "-ts_view_file", fileName, 1024, &opt);CHKERRQ(ierr);
      if (opt) {
        ierr = PetscViewerSetFilename(viewer, fileName);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerSetFilename(viewer, ts->name);CHKERRQ(ierr);
      }
      ierr = TSView(ts, viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    } else {
      ierr = TSView(ts, PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsHasName(ts->prefix, "-ts_view_draw", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscViewerDrawOpen(ts->comm, 0, 0, 0, 0, 300, 300, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
    if (title) {
      ierr = PetscDrawSetTitle(draw, (char *)title);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectName((PetscObject) ts);CHKERRQ(ierr);
      ierr = PetscDrawSetTitle(draw, ts->name);CHKERRQ(ierr);
    }
    ierr = TSView(ts, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSComputeRHSJacobian"
/*@
   TSComputeRHSJacobian - Computes the Jacobian matrix that has been
      set with TSSetRHSJacobian().

   Collective on TS and Vec

   Input Parameters:
+  ts - the SNES context
.  t - current timestep
-  x - input vector

   Output Parameters:
+  A - Jacobian matrix
.  B - optional preconditioning matrix
-  flag - flag indicating matrix structure

   Notes: 
   Most users should not need to explicitly call this routine, as it 
   is used internally within the nonlinear solvers. 

   See KSPSetOperators() for important information about setting the
   flag parameter.

   TSComputeJacobian() is valid only for TS_NONLINEAR

   Level: developer

.keywords: SNES, compute, Jacobian, matrix

.seealso:  TSSetRHSJacobian(), KSPSetOperators()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSComputeRHSJacobian(TS ts,PetscReal t,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(X,VEC_COOKIE,3);
  PetscCheckSameComm(ts,1,X,3);
  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"For TS_NONLINEAR only");
  }
  if (ts->ops->rhsjacobian) {
    ierr = PetscLogEventBegin(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
    *flg = DIFFERENT_NONZERO_PATTERN;
    PetscStackPush("TS user Jacobian function");
    ierr = (*ts->ops->rhsjacobian)(ts,t,X,A,B,flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
    ierr = PetscLogEventEnd(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
    /* make sure user returned a correct Jacobian and preconditioner */
    PetscValidHeaderSpecific(*A,MAT_COOKIE,4);
    PetscValidHeaderSpecific(*B,MAT_COOKIE,5);  
  } else {
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (*A != *B) {
      ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSComputeRHSFunction"
/*
   TSComputeRHSFunction - Evaluates the right-hand-side function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
PetscErrorCode TSComputeRHSFunction(TS ts,PetscReal t,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);
  if (ts->ops->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->ops->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    if (ts->ops->rhsmatrix) { /* assemble matrix for this timestep */
      MatStructure flg;
      PetscStackPush("TS user right-hand-side matrix function");
      ierr = (*ts->ops->rhsmatrix)(ts,t,&ts->A,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
      PetscStackPop;
    }
    ierr = MatMult(ts->A,x,y);CHKERRQ(ierr);
  }

  /* apply user-provided boundary conditions (only needed if these are time dependent) */
  ierr = TSComputeRHSBoundaryConditions(ts,t,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetRHSFunction"
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
$     func (TS ts,PetscReal t,Vec u,Vec F,void *ctx);

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
PetscErrorCode PETSCTS_DLLEXPORT TSSetRHSFunction(TS ts,PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,void*),void *ctx)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot set function for linear problem");
  }
  ts->ops->rhsfunction = f;
  ts->funP             = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetRHSMatrix"
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
$     func (TS ts,PetscReal t,Mat *A,Mat *B,PetscInt *flag,void *ctx);

+  t - current timestep
.  A - matrix A, where U_t = A(t) U
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx - [optional] user-defined context for matrix evaluation routine

   Notes: 
   See KSPSetOperators() for important information about setting the flag
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetRHSMatrix(TS ts,Mat A,Mat B,PetscErrorCode (*f)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(ts,1,A,2);
  PetscCheckSameComm(ts,1,B,3);
  if (ts->problem_type == TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Not for nonlinear problems; use TSSetRHSJacobian()");
  }

  ts->ops->rhsmatrix = f;
  ts->jacP           = ctx;
  ts->A              = A;
  ts->B              = B;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetRHSJacobian"
/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of F,
   where U_t = F(U,t), as well as the location to store the matrix.
   Use TSSetRHSMatrix() for linear problems.

   Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  A   - Jacobian matrix
.  B   - preconditioner matrix (usually same as A)
.  f   - the Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$     func (TS ts,PetscReal t,Vec u,Mat *A,Mat *B,MatStructure *flag,void *ctx);

+  t - current timestep
.  u - input vector
.  A - matrix A, where U_t = A(t)u
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx - [optional] user-defined context for matrix evaluation routine

   Notes: 
   See KSPSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the matrix evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner
   
.keywords: TS, timestep, set, right-hand-side, Jacobian

.seealso: TSDefaultComputeJacobianColor(),
          SNESDefaultComputeJacobianColor(), TSSetRHSFunction(), TSSetRHSMatrix()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetRHSJacobian(TS ts,Mat A,Mat B,PetscErrorCode (*f)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(ts,1,A,2);
  PetscCheckSameComm(ts,1,B,3);
  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Not for linear problems; use TSSetRHSMatrix()");
  }

  ts->ops->rhsjacobian = f;
  ts->jacP             = ctx;
  ts->A                = A;
  ts->B                = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSComputeRHSBoundaryConditions"
/*
   TSComputeRHSBoundaryConditions - Evaluates the boundary condition function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
PetscErrorCode TSComputeRHSBoundaryConditions(TS ts,PetscReal t,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(ts,1,x,3);

  if (ts->ops->rhsbc) {
    PetscStackPush("TS user boundary condition function");
    ierr = (*ts->ops->rhsbc)(ts,t,x,ts->bcP);CHKERRQ(ierr);
    PetscStackPop;
    PetscFunctionReturn(0);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetRHSBoundaryConditions"
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
$     func (TS ts,PetscReal t,Vec F,void *ctx);

+   t - current timestep
.   F - function vector
-   ctx - [optional] user-defined function context 

    Level: intermediate

.keywords: TS, timestep, set, boundary conditions, function
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetRHSBoundaryConditions(TS ts,PetscErrorCode (*f)(TS,PetscReal,Vec,void*),void *ctx)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (ts->problem_type != TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"For linear problems only");
  }
  ts->ops->rhsbc = f;
  ts->bcP        = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView"
/*@C
    TSView - Prints the TS data structure.

    Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_view - calls TSView() at end of TSStep()

    Notes:
    The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

    The user can open an alternative visualization context with
    PetscViewerASCIIOpen() - output to a specified file.

    Level: beginner

.keywords: TS, timestep, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSView(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  char           *type;
  PetscTruth     iascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(ts->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"TS Object:\n");CHKERRQ(ierr);
    ierr = TSGetType(ts,(TSType *)&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (ts->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum steps=%D\n",ts->max_steps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum time=%g\n",ts->max_time);CHKERRQ(ierr);
    if (ts->problem_type == TS_NONLINEAR) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%D\n",ts->nonlinear_its);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",ts->linear_its);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = TSGetType(ts,(TSType *)&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",type);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (ts->ksp) {ierr = KSPView(ts->ksp,viewer);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetApplicationContext"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->user = usrP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetApplicationContext"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetApplicationContext(TS ts,void **usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  *usrP = ts->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetTimeStepNumber"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetTimeStepNumber(TS ts,PetscInt* iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidIntPointer(iter,2);
  *iter = ts->steps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetInitialTimeStep"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetInitialTimeStep(TS ts,PetscReal initial_time,PetscReal time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->time_step         = time_step;
  ts->initial_time_step = time_step;
  ts->ptime             = initial_time;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetTimeStep"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetTimeStep(TS ts,PetscReal time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->time_step = time_step;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetTimeStep"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetTimeStep(TS ts,PetscReal* dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidDoublePointer(dt,2);
  *dt = ts->time_step;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetSolution"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetSolution(TS ts,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(v,2);
  *v = ts->vec_sol_always;
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a timestepper ---- */
#undef __FUNCT__  
#define __FUNCT__ "TSSetProblemType"
/*@C
  TSSetProblemType - Sets the type of problem to be solved.

  Not collective

  Input Parameters:
+ ts   - The TS
- type - One of TS_LINEAR, TS_NONLINEAR where these types refer to problems of the forms
.vb
         U_t = A U    
         U_t = A(t) U 
         U_t = F(t,U) 
.ve

   Level: beginner

.keywords: TS, problem type
.seealso: TSSetUp(), TSProblemType, TS
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetProblemType(TS ts, TSProblemType type) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->problem_type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetProblemType"
/*@C
  TSGetProblemType - Gets the type of problem to be solved.

  Not collective

  Input Parameter:
. ts   - The TS

  Output Parameter:
. type - One of TS_LINEAR, TS_NONLINEAR where these types refer to problems of the forms
.vb
         U_t = A U    
         U_t = A(t) U 
         U_t = F(t,U) 
.ve

   Level: beginner

.keywords: TS, problem type
.seealso: TSSetUp(), TSProblemType, TS
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetProblemType(TS ts, TSProblemType *type) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  PetscValidIntPointer(type,2);
  *type = ts->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetUp(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (!ts->vec_sol) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetSolution() first");
  if (!ts->type_name) {
    ierr = TSSetType(ts,TS_EULER);CHKERRQ(ierr);
  }
  ierr = (*ts->ops->setup)(ts);CHKERRQ(ierr);
  ts->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDestroy"
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
PetscErrorCode PETSCTS_DLLEXPORT TSDestroy(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (--ts->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(ts);CHKERRQ(ierr);

  if (ts->ksp) {ierr = KSPDestroy(ts->ksp);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes);CHKERRQ(ierr);}
  ierr = (*(ts)->ops->destroy)(ts);CHKERRQ(ierr);
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->mdestroy[i]) {
      ierr = (*ts->mdestroy[i])(ts->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscHeaderDestroy(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetSNES"
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
   KSP, KSP, and PC contexts as well.

   TSGetSNES() does not work for integrators that do not use SNES; in
   this case TSGetSNES() returns PETSC_NULL in snes.

   Level: beginner

.keywords: timestep, get, SNES
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetSNES(TS ts,SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(snes,2);
  if (ts->problem_type == TS_LINEAR) SETERRQ(PETSC_ERR_ARG_WRONG,"Nonlinear only; use TSGetKSP()");
  *snes = ts->snes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetKSP"
/*@C
   TSGetKSP - Returns the KSP (linear solver) associated with 
   a TS (timestepper) context.

   Not Collective, but KSP is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  ksp - the nonlinear solver context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   KSP and PC contexts as well.

   TSGetKSP() does not work for integrators that do not use KSP;
   in this case TSGetKSP() returns PETSC_NULL in ksp.

   Level: beginner

.keywords: timestep, get, KSP
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetKSP(TS ts,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(ksp,2);
  if (ts->problem_type != TS_LINEAR) SETERRQ(PETSC_ERR_ARG_WRONG,"Linear only; use TSGetSNES()");
  *ksp = ts->ksp;
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNCT__  
#define __FUNCT__ "TSGetDuration"
/*@
   TSGetDuration - Gets the maximum number of timesteps to use and 
   maximum time for iteration.

   Collective on TS

   Input Parameters:
+  ts       - the TS context obtained from TSCreate()
.  maxsteps - maximum number of iterations to use, or PETSC_NULL
-  maxtime  - final time to iterate to, or PETSC_NULL

   Level: intermediate

.keywords: TS, timestep, get, maximum, iterations, time
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetDuration(TS ts, PetscInt *maxsteps, PetscReal *maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  if (maxsteps) {
    PetscValidIntPointer(maxsteps,2);
    *maxsteps = ts->max_steps;
  }
  if (maxtime ) {
    PetscValidScalarPointer(maxtime,3);
    *maxtime  = ts->max_time;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetDuration"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetDuration(TS ts,PetscInt maxsteps,PetscReal maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->max_steps = maxsteps;
  ts->max_time  = maxtime;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetSolution"
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetSolution(TS ts,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  ts->vec_sol        = ts->vec_sol_always = x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultRhsBC"
/*@
  TSDefaultRhsBC - The default boundary condition function which does nothing.

  Collective on TS

  Input Parameters:
+ ts  - The TS context obtained from TSCreate()
. rhs - The Rhs
- ctx - The user-context

  Level: developer

.keywords: TS, Rhs, boundary conditions
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultRhsBC(TS ts,  Vec rhs, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetSystemMatrixBC"
/*@C
  TSSetSystemMatrixBC - Sets the function which applies boundary conditions
  to the system matrix and preconditioner of each system.

  Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts, Mat A, Mat B, void *ctx);

+ A   - The current system matrix
. B   - The current preconditioner
- ctx - The user-context

  Level: intermediate

.keywords: TS, System matrix, boundary conditions
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetSystemMatrixBC(TS ts, PetscErrorCode (*func)(TS, Mat, Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->ops->applymatrixbc = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultSystemMatrixBC"
/*@
  TSDefaultSystemMatrixBC - The default boundary condition function which
  does nothing.

  Collective on TS

  Input Parameters:
+ ts  - The TS context obtained from TSCreate()
. A   - The system matrix
. B   - The preconditioner
- ctx - The user-context

  Level: developer

.keywords: TS, System matrix, boundary conditions
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultSystemMatrixBC(TS ts, Mat A, Mat B, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetSolutionBC"
/*@C
  TSSetSolutionBC - Sets the function which applies boundary conditions
  to the solution of each system. This is necessary in nonlinear systems
  which time dependent boundary conditions.

  Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts, Vec rsol, void *ctx);

+ sol - The current solution vector
- ctx - The user-context

  Level: intermediate

.keywords: TS, solution, boundary conditions
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetSolutionBC(TS ts, PetscErrorCode (*func)(TS, Vec, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->ops->applysolbc = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultSolutionBC"
/*@
  TSDefaultSolutionBC - The default boundary condition function which
  does nothing.

  Collective on TS

  Input Parameters:
+ ts  - The TS context obtained from TSCreate()
. sol - The solution
- ctx - The user-context

  Level: developer

.keywords: TS, solution, boundary conditions
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultSolutionBC(TS ts, Vec sol, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetPreStep"
/*@C
  TSSetPreStep - Sets the general-purpose function
  called once at the beginning of time stepping.

  Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts);

  Level: intermediate

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetPreStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->ops->prestep = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultPreStep"
/*@
  TSDefaultPreStep - The default pre-stepping function which does nothing.

  Collective on TS

  Input Parameters:
. ts  - The TS context obtained from TSCreate()

  Level: developer

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultPreStep(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUpdate"
/*@C
  TSSetUpdate - Sets the general-purpose update function called
  at the beginning of every time step. This function can change
  the time step.

  Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts, double t, double *dt);

+ t   - The current time
- dt  - The current time step

  Level: intermediate

.keywords: TS, update, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetUpdate(TS ts, PetscErrorCode (*func)(TS, PetscReal, PetscReal *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->ops->update = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultUpdate"
/*@
  TSDefaultUpdate - The default update function which does nothing.

  Collective on TS

  Input Parameters:
+ ts  - The TS context obtained from TSCreate()
- t   - The current time

  Output Parameters:
. dt  - The current time step

  Level: developer

.keywords: TS, update, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultUpdate(TS ts, PetscReal t, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetPostStep"
/*@C
  TSSetPostStep - Sets the general-purpose function
  called once at the end of time stepping.

  Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts);

  Level: intermediate

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetPostStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ts->ops->poststep = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultPostStep"
/*@
  TSDefaultPostStep - The default post-stepping function which does nothing.

  Collective on TS

  Input Parameters:
. ts  - The TS context obtained from TSCreate()

  Level: developer

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSDefaultPostStep(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNCT__  
#define __FUNCT__ "TSSetMonitor"
/*@C
   TSSetMonitor - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.   

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  func - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
             monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling sequence of func:
$    int func(TS ts,PetscInt steps,PetscReal time,Vec x,void *mctx)

+    ts - the TS context
.    steps - iteration number
.    time - current time
.    x - current iterate
-    mctx - [optional] monitoring context

   Notes:
   This routine adds an additional monitor to the list of monitors that 
   already has been loaded.

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSDefaultMonitor(), TSClearMonitor()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetMonitor(TS ts,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *mctx,PetscErrorCode (*mdestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (ts->numbermonitors >= MAXTSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  }
  ts->monitor[ts->numbermonitors]           = monitor;
  ts->mdestroy[ts->numbermonitors]          = mdestroy;
  ts->monitorcontext[ts->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSClearMonitor"
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
PetscErrorCode PETSCTS_DLLEXPORT TSClearMonitor(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDefaultMonitor"
PetscErrorCode TSDefaultMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(ts->comm,"timestep %D dt %g time %g\n",step,ts->time_step,ptime);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSStep"
/*@
   TSStep - Steps the requested number of timesteps.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameters:
+  steps - number of iterations until termination
-  ptime - time until termination

   Level: beginner

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetUp(), TSDestroy()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSStep(TS ts,PetscInt *steps,PetscReal *ptime)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  if (!ts->setupcalled) {
    ierr = TSSetUp(ts);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(TS_Step, ts, 0, 0, 0);CHKERRQ(ierr);
  ierr = (*ts->ops->prestep)(ts);CHKERRQ(ierr);
  ierr = (*ts->ops->step)(ts, steps, ptime);CHKERRQ(ierr);
  ierr = (*ts->ops->poststep)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_Step, ts, 0, 0, 0);CHKERRQ(ierr);

  if (!PetscPreLoadingOn) {
    ierr = TSViewFromOptions(ts,ts->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSMonitor"
/*
     Runs the user provided monitor routines, if they exists.
*/
PetscErrorCode TSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec x)
{
  PetscErrorCode ierr;
  PetscInt i,n = ts->numbermonitors;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = (*ts->monitor[i])(ts,step,ptime,x,ts->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSLGMonitorCreate"
/*@C
   TSLGMonitorCreate - Creates a line graph context for use with 
   TS to monitor convergence of preconditioned residual norms.

   Collective on TS

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
.  -ts_xmonitor - automatically sets line graph monitor

   Notes: 
   Use TSLGMonitorDestroy() to destroy this line graph, not PetscDrawLGDestroy().

   Level: intermediate

.keywords: TS, monitor, line graph, residual, seealso

.seealso: TSLGMonitorDestroy(), TSSetMonitor()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSLGMonitorCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscDraw      win;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetType(win,PETSC_DRAW_X);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(win,1,draw);CHKERRQ(ierr);
  ierr = PetscDrawLGIndicateDataPoints(*draw);CHKERRQ(ierr);

  ierr = PetscLogObjectParent(*draw,win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSLGMonitor"
PetscErrorCode TSLGMonitor(TS ts,PetscInt n,PetscReal ptime,Vec v,void *monctx)
{
  PetscDrawLG    lg = (PetscDrawLG) monctx;
  PetscReal      x,y = ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!monctx) {
    MPI_Comm    comm;
    PetscViewer viewer;

    ierr   = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
    ierr   = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  }

  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x = (PetscReal)n;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || (n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "TSLGMonitorDestroy" 
/*@C
   TSLGMonitorDestroy - Destroys a line graph context that was created 
   with TSLGMonitorCreate().

   Collective on PetscDrawLG

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: TS, monitor, line graph, destroy

.seealso: TSLGMonitorCreate(),  TSSetMonitor(), TSLGMonitor();
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSLGMonitorDestroy(PetscDrawLG drawlg)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetTime"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetTime(TS ts,PetscReal* t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidDoublePointer(t,2);
  *t = ts->ptime;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetOptionsPrefix" 
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
PetscErrorCode PETSCTS_DLLEXPORT TSSetOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESSetOptionsPrefix(ts->snes,prefix);CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = KSPSetOptionsPrefix(ts->ksp,prefix);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSAppendOptionsPrefix" 
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
PetscErrorCode PETSCTS_DLLEXPORT TSAppendOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  switch(ts->problem_type) {
    case TS_NONLINEAR:
      ierr = SNESAppendOptionsPrefix(ts->snes,prefix);CHKERRQ(ierr);
      break;
    case TS_LINEAR:
      ierr = KSPAppendOptionsPrefix(ts->ksp,prefix);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetOptionsPrefix"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetOptionsPrefix(TS ts,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSMatrix"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetRHSMatrix(TS ts,Mat *A,Mat *M,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (A)   *A = ts->A;
  if (M)   *M = ts->B;
  if (ctx) *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSJacobian"
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
PetscErrorCode PETSCTS_DLLEXPORT TSGetRHSJacobian(TS ts,Mat *J,Mat *M,void **ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetRHSMatrix(ts,J,M,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSVecViewMonitor"
/*@C
   TSVecViewMonitor - Monitors progress of the TS solvers by calling 
   VecView() for the solution at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: TS,  vector, monitor, view

.seealso: TSSetMonitor(), TSDefaultMonitor(), VecView()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSVecViewMonitor(TS ts,PetscInt step,PetscReal ptime,Vec x,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



