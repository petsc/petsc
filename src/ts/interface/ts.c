#define PETSCTS_DLL

#include "private/tsimpl.h"        /*I "petscts.h"  I*/

/* Logging support */
PetscCookie PETSCTS_DLLEXPORT TS_COOKIE;
PetscLogEvent  TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

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
  if (((PetscObject)ts)->type_name) {
    defaultType = ((PetscObject)ts)->type_name;
  } else {
    defaultType = TSEULER;
  }

  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
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
+  -ts_type <type> - TSEULER, TSBEULER, TSSUNDIALS, TSPSEUDO, TSCRANK_NICHOLSON
.  -ts_max_steps maxsteps - maximum number of time-steps to take
.  -ts_max_time time - maximum time to compute to
.  -ts_dt dt - initial time step
.  -ts_monitor - print information at each timestep
-  -ts_monitor_draw - plot information at each timestep

   Level: beginner

.keywords: TS, timestep, set, options, database

.seealso: TSGetType()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetFromOptions(TS ts)
{
  PetscReal               dt;
  PetscTruth              opt,flg;
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor monviewer;
  char                    monfilename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)ts)->comm, ((PetscObject)ts)->prefix, "Time step options", "TS");CHKERRQ(ierr);

    /* Handle generic TS options */
    ierr = PetscOptionsInt("-ts_max_steps","Maximum number of time steps","TSSetDuration",ts->max_steps,&ts->max_steps,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_max_time","Time to run to","TSSetDuration",ts->max_time,&ts->max_time,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_init_time","Initial time","TSSetInitialTime", ts->ptime, &ts->ptime, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_dt","Initial time step","TSSetInitialTimeStep",ts->initial_time_step,&dt,&opt);CHKERRQ(ierr);
    if (opt) {
      ts->initial_time_step = ts->time_step = dt;
    }

    /* Monitor options */
    ierr = PetscOptionsString("-ts_monitor","Monitor timestep size","TSMonitorDefault","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ts)->comm,monfilename,((PetscObject)ts)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    opt  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ts_monitor_draw","Monitor timestep size graphically","TSMonitorLG",opt,&opt,PETSC_NULL);CHKERRQ(ierr);
    if (opt) {
      ierr = TSMonitorSet(ts,TSMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    opt  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ts_monitor_solution","Monitor solution graphically","TSMonitorSolution",opt,&opt,PETSC_NULL);CHKERRQ(ierr);
    if (opt) {
      ierr = TSMonitorSet(ts,TSMonitorSolution,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
      ierr = KSPSetOperators(ts->ksp,ts->Arhs,ts->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ts->ksp);CHKERRQ(ierr);
    }
    break;
  case TS_NONLINEAR:
    if (ts->snes) {
      /* this is a bit of a hack, but it gets the matrix information into SNES earlier
         so that SNES and KSP have more information to pick reasonable defaults
         before they allow users to set options
       * If ts->A has been set at this point, we are probably using the implicit form
         and Arhs will never be used. */
      ierr = SNESSetJacobian(ts->snes,ts->A?ts->A:ts->Arhs,ts->B,0,ts);CHKERRQ(ierr);
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
  PetscTruth     opt = PETSC_FALSE;
  char           fileName[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(((PetscObject)ts)->prefix, "-ts_view", fileName, PETSC_MAX_PATH_LEN, &opt);CHKERRQ(ierr);
  if (opt && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)ts)->comm,fileName,&viewer);CHKERRQ(ierr);
    ierr = TSView(ts, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }
  opt = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)ts)->prefix, "-ts_view_draw", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscViewerDrawOpen(((PetscObject)ts)->comm, 0, 0, 0, 0, 300, 300, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
    if (title) {
      ierr = PetscDrawSetTitle(draw, (char *)title);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectName((PetscObject)ts);CHKERRQ(ierr);
      ierr = PetscDrawSetTitle(draw, ((PetscObject)ts)->name);CHKERRQ(ierr);
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
+  ts - the TS context
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
/*@
   TSComputeRHSFunction - Evaluates the right-hand-side function. 

   Collective on TS and Vec

   Input Parameters:
+  ts - the TS context
.  t - current time
-  x - state vector

   Output Parameter:
.  y - right hand side

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   If the user did not provide a function but merely a matrix,
   this routine applies the matrix.

   Level: developer

.keywords: TS, compute

.seealso: TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeRHSFunction(TS ts,PetscReal t,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);
  if (ts->ops->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->ops->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    if (ts->ops->rhsmatrix) { /* assemble matrix for this timestep */
      MatStructure flg;
      PetscStackPush("TS user right-hand-side matrix function");
      ierr = (*ts->ops->rhsmatrix)(ts,t,&ts->Arhs,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
      PetscStackPop;
    }
    ierr = MatMult(ts->Arhs,x,y);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSComputeIFunction"
/*@
   TSComputeIFunction - Evaluates the DAE residual written in implicit form F(t,X,Xdot)=0

   Collective on TS and Vec

   Input Parameters:
+  ts - the TS context
.  t - current time
.  X - state vector
-  Xdot - time derivative of state vector

   Output Parameter:
.  Y - right hand side

   Note:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   If the user did did not write their equations in implicit form, this
   function recasts them in implicit form.

   Level: developer

.keywords: TS, compute

.seealso: TSSetIFunction(), TSComputeRHSFunction()
@*/
PetscErrorCode TSComputeIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(X,VEC_COOKIE,3);
  PetscValidHeaderSpecific(Xdot,VEC_COOKIE,4);
  PetscValidHeaderSpecific(Y,VEC_COOKIE,5);

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,X,Xdot,Y);CHKERRQ(ierr);
  if (ts->ops->ifunction) {
    PetscStackPush("TS user implicit function");
    ierr = (*ts->ops->ifunction)(ts,t,X,Xdot,Y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    if (ts->ops->rhsfunction) {
      PetscStackPush("TS user right-hand-side function");
      ierr = (*ts->ops->rhsfunction)(ts,t,X,Y,ts->funP);CHKERRQ(ierr);
      PetscStackPop;
    } else {
      if (ts->ops->rhsmatrix) { /* assemble matrix for this timestep */
        MatStructure flg;
        /* Note: flg is not being used.
           For it to be useful, we'd have to cache it and then apply it in TSComputeIJacobian.
        */
        PetscStackPush("TS user right-hand-side matrix function");
        ierr = (*ts->ops->rhsmatrix)(ts,t,&ts->Arhs,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
        PetscStackPop;
      }
      ierr = MatMult(ts->Arhs,X,Y);CHKERRQ(ierr);
    }

    /* Convert to implicit form: F(X,Xdot) = Alhs * Xdot - Frhs(X) */
    if (ts->Alhs) {
      if (ts->ops->lhsmatrix) {
        MatStructure flg;
        PetscStackPush("TS user left-hand-side matrix function");
        ierr = (*ts->ops->lhsmatrix)(ts,t,&ts->Alhs,PETSC_NULL,&flg,ts->jacP);CHKERRQ(ierr);
        PetscStackPop;
      }
      ierr = VecScale(Y,-1.);CHKERRQ(ierr);
      ierr = MatMultAdd(ts->Alhs,Xdot,Y,Y);CHKERRQ(ierr);
    } else {
      ierr = VecAYPX(Y,-1,Xdot);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(TS_FunctionEval,ts,X,Xdot,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSComputeIJacobian"
/*@
   TSComputeIJacobian - Evaluates the Jacobian of the DAE

   Collective on TS and Vec

   Input
      Input Parameters:
+  ts - the TS context
.  t - current timestep
.  X - state vector
.  Xdot - time derivative of state vector
-  shift - shift to apply, see note below

   Output Parameters:
+  A - Jacobian matrix
.  B - optional preconditioning matrix
-  flag - flag indicating matrix structure

   Notes:
   If F(t,X,Xdot)=0 is the DAE, the required Jacobian is

   dF/dX + shift*dF/dXdot

   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   TSComputeIJacobian() is valid only for TS_NONLINEAR

   Level: developer

.keywords: TS, compute, Jacobian, matrix

.seealso:  TSSetIJacobian()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSComputeIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidHeaderSpecific(X,VEC_COOKIE,3);
  PetscValidHeaderSpecific(Xdot,VEC_COOKIE,4);
  PetscValidPointer(A,6);
  PetscValidHeaderSpecific(*A,MAT_COOKIE,6);
  PetscValidPointer(B,7);
  PetscValidHeaderSpecific(*B,MAT_COOKIE,7);
  PetscValidPointer(flg,8);

  *flg = SAME_NONZERO_PATTERN;  /* In case it we're solving a linear problem in which case it wouldn't get initialized below. */
  ierr = PetscLogEventBegin(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
  if (ts->ops->ijacobian) {
    PetscStackPush("TS user implicit Jacobian");
    ierr = (*ts->ops->ijacobian)(ts,t,X,Xdot,shift,A,B,flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    if (ts->ops->rhsjacobian) {
      PetscStackPush("TS user right-hand-side Jacobian");
      ierr = (*ts->ops->rhsjacobian)(ts,t,X,A,B,flg,ts->jacP);CHKERRQ(ierr);
      PetscStackPop;
    } else {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      if (*A != *B) {
        ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    }

    /* Convert to implicit form */
    /* inefficient because these operations will normally traverse all matrix elements twice */
    ierr = MatScale(*A,-1);CHKERRQ(ierr);
    if (ts->Alhs) {
      ierr = MatAXPY(*A,shift,ts->Alhs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = MatShift(*A,shift);CHKERRQ(ierr);
    }
    if (*A != *B) {
      ierr = MatScale(*B,-1);CHKERRQ(ierr);
      ierr = MatShift(*B,shift);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
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
    The user MUST call either this routine or TSSetMatrices().

    Level: beginner

.keywords: TS, timestep, set, right-hand-side, function

.seealso: TSSetMatrices()
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
#define __FUNCT__ "TSSetMatrices"
/*@C
   TSSetMatrices - Sets the functions to compute the matrices Alhs and Arhs, 
   where Alhs(t) U_t = Arhs(t) U.

   Collective on TS

   Input Parameters:
+  ts   - the TS context obtained from TSCreate()
.  Arhs - matrix
.  frhs - the matrix evaluation routine for Arhs; use PETSC_NULL (PETSC_NULL_FUNCTION in fortran)
          if Arhs is not a function of t.
.  Alhs - matrix or PETSC_NULL if Alhs is an indentity matrix.
.  flhs - the matrix evaluation routine for Alhs; use PETSC_NULL (PETSC_NULL_FUNCTION in fortran)
          if Alhs is not a function of t.
.  flag - flag indicating information about the matrix structure of Arhs and Alhs. 
          The available options are
            SAME_NONZERO_PATTERN - Alhs has the same nonzero structure as Arhs
            DIFFERENT_NONZERO_PATTERN - Alhs has different nonzero structure as Arhs
-  ctx  - [optional] user-defined context for private data for the 
          matrix evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$     func(TS ts,PetscReal t,Mat *A,Mat *B,PetscInt *flag,void *ctx);

+  t - current timestep
.  A - matrix A, where U_t = A(t) U
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx - [optional] user-defined context for matrix evaluation routine

   Notes:  
   The routine func() takes Mat* as the matrix arguments rather than Mat.  
   This allows the matrix evaluation routine to replace Arhs or Alhs with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Important: 
   The user MUST call either this routine or TSSetRHSFunction().

   Level: beginner

.keywords: TS, timestep, set, matrix

.seealso: TSSetRHSFunction()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetMatrices(TS ts,Mat Arhs,PetscErrorCode (*frhs)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),Mat Alhs,PetscErrorCode (*flhs)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),MatStructure flag,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (Arhs){
    PetscValidHeaderSpecific(Arhs,MAT_COOKIE,2);
    PetscCheckSameComm(ts,1,Arhs,2);
    ts->Arhs           = Arhs;
    ts->ops->rhsmatrix = frhs;
  }
  if (Alhs){
    PetscValidHeaderSpecific(Alhs,MAT_COOKIE,4);
    PetscCheckSameComm(ts,1,Alhs,4);
    ts->Alhs           = Alhs;
    ts->ops->lhsmatrix = flhs;
  }
  
  ts->jacP           = ctx;
  ts->matflg         = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetMatrices"
/*@C
   TSGetMatrices - Returns the matrices Arhs and Alhs at the present timestep,
   where Alhs(t) U_t = Arhs(t) U.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  Arhs - The right-hand side matrix
.  Alhs - The left-hand side matrix
-  ctx - User-defined context for matrix evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Level: intermediate

.seealso: TSSetMatrices(), TSGetTimeStep(), TSGetTime(), TSGetTimeStepNumber(), TSGetRHSJacobian()

.keywords: TS, timestep, get, matrix

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetMatrices(TS ts,Mat *Arhs,Mat *Alhs,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (Arhs) *Arhs = ts->Arhs;
  if (Alhs) *Alhs = ts->Alhs;
  if (ctx)  *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetRHSJacobian"
/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of F,
   where U_t = F(U,t), as well as the location to store the matrix.
   Use TSSetMatrices() for linear problems.

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
   completely new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner
   
.keywords: TS, timestep, set, right-hand-side, Jacobian

.seealso: TSDefaultComputeJacobianColor(),
          SNESDefaultComputeJacobianColor(), TSSetRHSFunction(), TSSetMatrices()

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
    SETERRQ(PETSC_ERR_ARG_WRONG,"Not for linear problems; use TSSetMatrices()");
  }

  ts->ops->rhsjacobian = f;
  ts->jacP             = ctx;
  ts->Arhs             = A;
  ts->B                = B;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction"
/*@C
   TSSetIFunction - Set the function to compute F(t,U,U_t) where F = 0 is the DAE to be solved.

   Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  f   - the function evaluation routine
-  ctx - user-defined context for private data for the function evaluation routine (may be PETSC_NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  F   - function vector
-  ctx - [optional] user-defined context for matrix evaluation routine

   Important:
   The user MUST call either this routine, TSSetRHSFunction(), or TSSetMatrices().  This routine must be used when not solving an ODE.

   Level: beginner

.keywords: TS, timestep, set, DAE, Jacobian

.seealso: TSSetMatrices(), TSSetRHSFunction(), TSSetIJacobian()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetIFunction(TS ts,TSIFunction f,void *ctx)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->ops->ifunction = f;
  ts->funP           = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian"
/*@C
   TSSetIJacobian - Set the function to compute the Jacobian of
   G(U) = F(t,U,U0+a*U) where F(t,U,U_t) = 0 is the DAE to be solved.

   Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  A   - Jacobian matrix
.  B   - preconditioning matrix for A (may be same as A)
.  f   - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx);

+  t    - time at step/stage being solved
.  u    - state vector
.  u_t  - time derivative of state vector
.  a    - shift
.  A    - Jacobian of G(U) = F(t,U,U0+a*U), equivalent to dF/dU + a*dF/dU_t
.  B    - preconditioning matrix for A, may be same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Notes:
   The matrices A and B are exactly the matrices that are used by SNES for the nonlinear solve.

   Level: beginner

.keywords: TS, timestep, DAE, Jacobian

.seealso: TSSetIFunction(), TSSetRHSJacobian()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetIJacobian(TS ts,Mat A,Mat B,TSIJacobian f,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (A) PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  if (A) PetscCheckSameComm(ts,1,A,2);
  if (B) PetscCheckSameComm(ts,1,B,3);
  if (f)   ts->ops->ijacobian = f;
  if (ctx) ts->jacP             = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    if (ts->A) {ierr = MatDestroy(ts->A);CHKERRQ(ierr);}
    ts->A = A;
  }
#if 0
  /* The sane and consistent alternative */
  if (B) {
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    if (ts->B) {ierr = MatDestroy(ts->B);CHKERRQ(ierr);}
    ts->B = B;
  }
#else
  /* Don't reference B because TSDestroy() doesn't destroy it.  These ownership semantics are awkward and inconsistent. */
  if (B) ts->B = B;
#endif
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
  const TSType   type;
  PetscTruth     iascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)ts)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"TS Object:\n");CHKERRQ(ierr);
    ierr = TSGetType(ts,&type);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum time=%G\n",ts->max_time);CHKERRQ(ierr);
    if (ts->problem_type == TS_NONLINEAR) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%D\n",ts->nonlinear_its);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",ts->linear_its);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = TSGetType(ts,&type);CHKERRQ(ierr);
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
/*@
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
/*@
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
  if (!((PetscObject)ts)->type_name) {
    ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  }
  ierr = (*ts->ops->setup)(ts);CHKERRQ(ierr);
  ts->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSDestroy"
/*@
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (--((PetscObject)ts)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(ts);CHKERRQ(ierr);
  if (ts->A) {ierr = MatDestroy(ts->A);CHKERRQ(ierr)}
  if (ts->ksp) {ierr = KSPDestroy(ts->ksp);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes);CHKERRQ(ierr);}
  if (ts->ops->destroy) {ierr = (*(ts)->ops->destroy)(ts);CHKERRQ(ierr);}
  ierr = TSMonitorCancel(ts);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetSNES"
/*@
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
  if (((PetscObject)ts)->type_name == PETSC_NULL) 
    SETERRQ(PETSC_ERR_ARG_NULL,"SNES is not created yet. Call TSSetType() first");
  if (ts->problem_type == TS_LINEAR) SETERRQ(PETSC_ERR_ARG_WRONG,"Nonlinear only; use TSGetKSP()");
  *snes = ts->snes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetKSP"
/*@
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
  if (((PetscObject)ts)->type_name == PETSC_NULL) 
    SETERRQ(PETSC_ERR_ARG_NULL,"KSP is not created yet. Call TSSetType() first");
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
#define __FUNCT__ "TSSetPreStep"
/*@C
  TSSetPreStep - Sets the general-purpose function
  called once at the beginning of each time step.

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
#define __FUNCT__ "TSPreStep"
/*@C
  TSPreStep - Runs the user-defined pre-step function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPreStep() is typically used within time stepping implementations,
  so most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPreStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (ts->ops->prestep) {
    PetscStackPush("TS PreStep function");
    CHKMEMQ;
    ierr = (*ts->ops->prestep)(ts);CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
  }
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
#define __FUNCT__ "TSSetPostStep"
/*@C
  TSSetPostStep - Sets the general-purpose function
  called once at the end of each time step.

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
#define __FUNCT__ "TSPostStep"
/*@C
  TSPostStep - Runs the user-defined post-step function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPostStep() is typically used within time stepping implementations,
  so most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, timestep
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPostStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  if (ts->ops->poststep) {
    PetscStackPush("TS PostStep function");
    CHKMEMQ;
    ierr = (*ts->ops->poststep)(ts);CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackPop;
  }
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
#define __FUNCT__ "TSMonitorSet"
/*@C
   TSMonitorSet - Sets an ADDITIONAL function that is to be used at every
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

   Fortran notes: Only a single monitor function can be set for each TS object

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSMonitorDefault(), TSMonitorCancel()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSMonitorSet(TS ts,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *mctx,PetscErrorCode (*mdestroy)(void*))
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
#define __FUNCT__ "TSMonitorCancel"
/*@C
   TSMonitorCancel - Clears all the monitors that have been set on a time-step object.   

   Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSMonitorDefault(), TSMonitorSet()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSMonitorCancel(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->mdestroy[i]) {
      ierr = (*ts->mdestroy[i])(ts->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ts->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSMonitorDefault"
/*@
   TSMonitorDefault - Sets the Default monitor

   Level: intermediate

.keywords: TS, set, monitor

.seealso: TSMonitorDefault(), TSMonitorSet()
@*/
PetscErrorCode TSMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor)ctx;

  PetscFunctionBegin;
  if (!ctx) {
    ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ts)->comm,"stdout",0,&viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIMonitorPrintf(viewer,"timestep %D dt %G time %G\n",step,ts->time_step,ptime);CHKERRQ(ierr);
  if (!ctx) {
    ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);
  }
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
  ierr = (*ts->ops->step)(ts, steps, ptime);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_Step, ts, 0, 0, 0);CHKERRQ(ierr);

  if (!PetscPreLoadingOn) {
    ierr = TSViewFromOptions(ts,((PetscObject)ts)->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSolve"
/*@
   TSSolve - Steps the requested number of timesteps.

   Collective on TS

   Input Parameter:
+  ts - the TS context obtained from TSCreate()
-  x - the solution vector, or PETSC_NULL if it was set with TSSetSolution()

   Level: beginner

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetSolution(), TSStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSolve(TS ts, Vec x)
{
  PetscInt       steps;
  PetscReal      ptime;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  /* set solution vector if provided */
  if (x) { ierr = TSSetSolution(ts, x); CHKERRQ(ierr); }
  /* reset time step and iteration counters */
  ts->steps = 0; ts->linear_its = 0; ts->nonlinear_its = 0;
  /* steps the requested number of timesteps. */
  ierr = TSStep(ts, &steps, &ptime);CHKERRQ(ierr);
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
#define __FUNCT__ "TSMonitorLGCreate"
/*@C
   TSMonitorLGCreate - Creates a line graph context for use with 
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
.  -ts_monitor_draw - automatically sets line graph monitor

   Notes: 
   Use TSMonitorLGDestroy() to destroy this line graph, not PetscDrawLGDestroy().

   Level: intermediate

.keywords: TS, monitor, line graph, residual, seealso

.seealso: TSMonitorLGDestroy(), TSMonitorSet()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSMonitorLGCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
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
#define __FUNCT__ "TSMonitorLG"
PetscErrorCode TSMonitorLG(TS ts,PetscInt n,PetscReal ptime,Vec v,void *monctx)
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
#define __FUNCT__ "TSMonitorLGDestroy" 
/*@C
   TSMonitorLGDestroy - Destroys a line graph context that was created 
   with TSMonitorLGCreate().

   Collective on PetscDrawLG

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: TS, monitor, line graph, destroy

.seealso: TSMonitorLGCreate(),  TSMonitorSet(), TSMonitorLG();
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSMonitorLGDestroy(PetscDrawLG drawlg)
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
#define __FUNCT__ "TSSetTime"
/*@
   TSSetTime - Allows one to reset the time.

   Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time - the time

   Level: intermediate

.seealso: TSGetTime(), TSSetDuration()

.keywords: TS, set, time
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSetTime(TS ts, PetscReal t) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  ts->ptime = t;
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
      if (ts->snes) {
        ierr = SNESSetOptionsPrefix(ts->snes,prefix);CHKERRQ(ierr);
      }
      break;
    case TS_LINEAR:
      if (ts->ksp) {
        ierr = KSPSetOptionsPrefix(ts->ksp,prefix);CHKERRQ(ierr);
      }
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
      if (ts->snes) {
        ierr = SNESAppendOptionsPrefix(ts->snes,prefix);CHKERRQ(ierr);
      }
      break;
    case TS_LINEAR:
      if (ts->ksp) {
        ierr = KSPAppendOptionsPrefix(ts->ksp,prefix);CHKERRQ(ierr);
      }
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

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetMatrices(), TSGetTime(), TSGetTimeStepNumber()

.keywords: TS, timestep, get, matrix, Jacobian
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetRHSJacobian(TS ts,Mat *J,Mat *M,void **ctx)
{
  PetscFunctionBegin;
  if (J) *J = ts->Arhs;
  if (M) *M = ts->B;
  if (ctx) *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetIJacobian"
/*@C
   TSGetIJacobian - Returns the implicit Jacobian at the present timestep.

   Not Collective, but parallel objects are returned if TS is parallel

   Input Parameter:
.  ts  - The TS context obtained from TSCreate()

   Output Parameters:
+  A   - The Jacobian of F(t,U,U_t)
.  B   - The preconditioner matrix, often the same as A
.  f   - The function to compute the matrices
- ctx - User-defined context for Jacobian evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Level: advanced

.seealso: TSGetTimeStep(), TSGetRHSJacobian(), TSGetMatrices(), TSGetTime(), TSGetTimeStepNumber()

.keywords: TS, timestep, get, matrix, Jacobian
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGetIJacobian(TS ts,Mat *A,Mat *B,TSIJacobian *f,void **ctx)
{
  PetscFunctionBegin;
  if (A) *A = ts->A;
  if (B) *B = ts->B;
  if (f) *f = ts->ops->ijacobian;
  if (ctx) *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSMonitorSolution"
/*@C
   TSMonitorSolution - Monitors progress of the TS solvers by calling 
   VecView() for the solution at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSMonitorSolution(TS ts,PetscInt step,PetscReal ptime,Vec x,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!dummy) {
    viewer = PETSC_VIEWER_DRAW_(((PetscObject)ts)->comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



