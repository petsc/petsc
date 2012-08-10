
#include <petsc-private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdmshell.h>

/* Logging support */
PetscClassId  TS_CLASSID;
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
  PetscBool      opt;
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
+  -ts_type <type> - TSEULER, TSBEULER, TSSUNDIALS, TSPSEUDO, TSCN, TSRK, TSTHETA, TSGL, TSSSP
.  -ts_max_steps maxsteps - maximum number of time-steps to take
.  -ts_final_time time - maximum time to compute to
.  -ts_dt dt - initial time step
.  -ts_monitor - print information at each timestep
-  -ts_monitor_draw - plot information at each timestep

   Level: beginner

.keywords: TS, timestep, set, options, database

.seealso: TSGetType()
@*/
PetscErrorCode  TSSetFromOptions(TS ts)
{
  PetscBool      opt,flg;
  PetscErrorCode ierr;
  PetscViewer    monviewer;
  char           monfilename[PETSC_MAX_PATH_LEN];
  SNES           snes;
  TSAdapt        adapt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)ts);CHKERRQ(ierr);
    /* Handle TS type options */
    ierr = TSSetTypeFromOptions(ts);CHKERRQ(ierr);

    /* Handle generic TS options */
    ierr = PetscOptionsInt("-ts_max_steps","Maximum number of time steps","TSSetDuration",ts->max_steps,&ts->max_steps,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_final_time","Time to run to","TSSetDuration",ts->max_time,&ts->max_time,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_init_time","Initial time","TSSetTime",ts->ptime,&ts->ptime,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_dt","Initial time step","TSSetTimeStep",ts->time_step,&ts->time_step,PETSC_NULL);CHKERRQ(ierr);
    opt = ts->exact_final_time == PETSC_DECIDE ? PETSC_FALSE : (PetscBool)ts->exact_final_time;
    ierr = PetscOptionsBool("-ts_exact_final_time","Interpolate output to stop exactly at the final time","TSSetExactFinalTime",opt,&opt,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSSetExactFinalTime(ts,opt);CHKERRQ(ierr);}
    ierr = PetscOptionsInt("-ts_max_snes_failures","Maximum number of nonlinear solve failures","TSSetMaxSNESFailures",ts->max_snes_failures,&ts->max_snes_failures,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_max_reject","Maximum number of step rejections before step fails","TSSetMaxStepRejections",ts->max_reject,&ts->max_reject,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_error_if_step_fails","Error if no step succeeds","TSSetErrorIfStepFails",ts->errorifstepfailed,&ts->errorifstepfailed,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_rtol","Relative tolerance for local truncation error","TSSetTolerances",ts->rtol,&ts->rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_atol","Absolute tolerance for local truncation error","TSSetTolerances",ts->atol,&ts->atol,PETSC_NULL);CHKERRQ(ierr);

    /* Monitor options */
    ierr = PetscOptionsString("-ts_monitor","Monitor timestep size","TSMonitorDefault","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)ts)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-ts_monitor_python","Use Python function","TSMonitorSet",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscPythonMonitorSet((PetscObject)ts,monfilename);CHKERRQ(ierr);}

    opt  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ts_monitor_draw","Monitor timestep size graphically","TSMonitorLG",opt,&opt,PETSC_NULL);CHKERRQ(ierr);
    if (opt) {
      ierr = TSMonitorSet(ts,TSMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    opt  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ts_monitor_solution","Monitor solution graphically","TSMonitorSolution",opt,&opt,PETSC_NULL);CHKERRQ(ierr);
    if (opt) {
      void *ctx;
      ierr = TSMonitorSolutionCreate(ts,PETSC_NULL,&ctx);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorSolution,ctx,TSMonitorSolutionDestroy);CHKERRQ(ierr);
    }
    opt  = PETSC_FALSE;
    ierr = PetscOptionsString("-ts_monitor_solution_binary","Save each solution to a binary file","TSMonitorSolutionBinary",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewer ctx;
      if (monfilename[0]) {
        ierr = PetscViewerBinaryOpen(((PetscObject)ts)->comm,monfilename,FILE_MODE_WRITE,&ctx);CHKERRQ(ierr);
      } else {
        ctx = PETSC_VIEWER_BINARY_(((PetscObject)ts)->comm);
      }
      ierr = TSMonitorSet(ts,TSMonitorSolutionBinary,ctx,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    opt  = PETSC_FALSE;
    ierr = PetscOptionsString("-ts_monitor_solution_vtk","Save each time step to a binary file, use filename-%%03D.vts","TSMonitorSolutionVTK",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      const char *ptr,*ptr2;
      char *filetemplate;
      if (!monfilename[0]) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"-ts_monitor_solution_vtk requires a file template, e.g. filename-%%03D.vts");
      /* Do some cursory validation of the input. */
      ierr = PetscStrstr(monfilename,"%",(char**)&ptr);CHKERRQ(ierr);
      if (!ptr) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"-ts_monitor_solution_vtk requires a file template, e.g. filename-%%03D.vts");
      for (ptr++ ; ptr && *ptr; ptr++) {
        ierr = PetscStrchr("DdiouxX",*ptr,(char**)&ptr2);CHKERRQ(ierr);
        if (!ptr2 && (*ptr < '0' || '9' < *ptr)) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Invalid file template argument to -ts_monitor_solution_vtk, should look like filename-%%03D.vts");
        if (ptr2) break;
      }
      ierr = PetscStrallocpy(monfilename,&filetemplate);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSMonitorSolutionVTK,filetemplate,(PetscErrorCode (*)(void**))TSMonitorSolutionVTKDestroy);CHKERRQ(ierr);
    }

    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetFromOptions(adapt);CHKERRQ(ierr);

    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    if (ts->problem_type == TS_LINEAR) {ierr = SNESSetType(snes,SNESKSPONLY);CHKERRQ(ierr);}

    /* Handle specific TS options */
    if (ts->ops->setfromoptions) {
      ierr = (*ts->ops->setfromoptions)(ts);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)ts);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
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

   Level: developer

.keywords: SNES, compute, Jacobian, matrix

.seealso:  TSSetRHSJacobian(), KSPSetOperators()
@*/
PetscErrorCode  TSComputeRHSJacobian(TS ts,PetscReal t,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;
  PetscInt Xstate;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscCheckSameComm(ts,1,X,3);
  ierr = PetscObjectStateQuery((PetscObject)X,&Xstate);CHKERRQ(ierr);
  if (ts->rhsjacobian.time == t && (ts->problem_type == TS_LINEAR || (ts->rhsjacobian.X == X && ts->rhsjacobian.Xstate == Xstate))) {
    *flg = ts->rhsjacobian.mstructure;
    PetscFunctionReturn(0);
  }

  if (!ts->userops->rhsjacobian && !ts->userops->ijacobian) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetRHSJacobian() and / or TSSetIJacobian()");

  if (ts->userops->rhsjacobian) {
    ierr = PetscLogEventBegin(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
    *flg = DIFFERENT_NONZERO_PATTERN;
    PetscStackPush("TS user Jacobian function");
    ierr = (*ts->userops->rhsjacobian)(ts,t,X,A,B,flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
    ierr = PetscLogEventEnd(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
    /* make sure user returned a correct Jacobian and preconditioner */
    PetscValidHeaderSpecific(*A,MAT_CLASSID,4);
    PetscValidHeaderSpecific(*B,MAT_CLASSID,5);
  } else {
    ierr = MatZeroEntries(*A);CHKERRQ(ierr);
    if (*A != *B) {ierr = MatZeroEntries(*B);CHKERRQ(ierr);}
    *flg = SAME_NONZERO_PATTERN;
  }
  ts->rhsjacobian.time = t;
  ts->rhsjacobian.X = X;
  ierr = PetscObjectStateQuery((PetscObject)X,&ts->rhsjacobian.Xstate);CHKERRQ(ierr);
  ts->rhsjacobian.mstructure = *flg;
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

   Level: developer

.keywords: TS, compute

.seealso: TSSetRHSFunction(), TSComputeIFunction()
@*/
PetscErrorCode TSComputeRHSFunction(TS ts,PetscReal t,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);

  if (!ts->userops->rhsfunction && !ts->userops->ifunction) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetRHSFunction() and / or TSSetIFunction()");

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);
  if (ts->userops->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->userops->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    ierr = VecZeroEntries(y);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(TS_FunctionEval,ts,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSVec_Private"
static PetscErrorCode TSGetRHSVec_Private(TS ts,Vec *Frhs)
{
  Vec            F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *Frhs = PETSC_NULL;
  ierr = TSGetIFunction(ts,&F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (!ts->Frhs) {
    ierr = VecDuplicate(F,&ts->Frhs);CHKERRQ(ierr);
  }
  *Frhs = ts->Frhs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSMats_Private"
static PetscErrorCode TSGetRHSMats_Private(TS ts,Mat *Arhs,Mat *Brhs)
{
  Mat            A,B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetIJacobian(ts,&A,&B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (Arhs) {
    if (!ts->Arhs) {
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&ts->Arhs);CHKERRQ(ierr);
    }
    *Arhs = ts->Arhs;
  }
  if (Brhs) {
    if (!ts->Brhs) {
      ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&ts->Brhs);CHKERRQ(ierr);
    }
    *Brhs = ts->Brhs;
  }
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
.  Xdot - time derivative of state vector
-  imex - flag indicates if the method is IMEX so that the RHSFunction should be kept separate

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
PetscErrorCode TSComputeIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec Y,PetscBool imex)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Xdot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,5);

  if (!ts->userops->rhsfunction && !ts->userops->ifunction) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetRHSFunction() and / or TSSetIFunction()");

  ierr = PetscLogEventBegin(TS_FunctionEval,ts,X,Xdot,Y);CHKERRQ(ierr);
  if (ts->userops->ifunction) {
    PetscStackPush("TS user implicit function");
    ierr = (*ts->userops->ifunction)(ts,t,X,Xdot,Y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
  }
  if (imex) {
    if (!ts->userops->ifunction) {
      ierr = VecCopy(Xdot,Y);CHKERRQ(ierr);
    }
  } else if (ts->userops->rhsfunction) {
    if (ts->userops->ifunction) {
      Vec Frhs;
      ierr = TSGetRHSVec_Private(ts,&Frhs);CHKERRQ(ierr);
      ierr = TSComputeRHSFunction(ts,t,X,Frhs);CHKERRQ(ierr);
      ierr = VecAXPY(Y,-1,Frhs);CHKERRQ(ierr);
    } else {
      ierr = TSComputeRHSFunction(ts,t,X,Y);CHKERRQ(ierr);
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
.  shift - shift to apply, see note below
-  imex - flag indicates if the method is IMEX so that the RHSJacobian should be kept separate

   Output Parameters:
+  A - Jacobian matrix
.  B - optional preconditioning matrix
-  flag - flag indicating matrix structure

   Notes:
   If F(t,X,Xdot)=0 is the DAE, the required Jacobian is

   dF/dX + shift*dF/dXdot

   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Level: developer

.keywords: TS, compute, Jacobian, matrix

.seealso:  TSSetIJacobian()
@*/
PetscErrorCode TSComputeIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *flg,PetscBool imex)
{
  PetscInt Xstate, Xdotstate;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Xdot,VEC_CLASSID,4);
  PetscValidPointer(A,6);
  PetscValidHeaderSpecific(*A,MAT_CLASSID,6);
  PetscValidPointer(B,7);
  PetscValidHeaderSpecific(*B,MAT_CLASSID,7);
  PetscValidPointer(flg,8);
  ierr = PetscObjectStateQuery((PetscObject)X,&Xstate);CHKERRQ(ierr);
  ierr = PetscObjectStateQuery((PetscObject)Xdot,&Xdotstate);CHKERRQ(ierr);
  if (ts->ijacobian.time == t && (ts->problem_type == TS_LINEAR || (ts->ijacobian.X == X && ts->ijacobian.Xstate == Xstate && ts->ijacobian.Xdot == Xdot && ts->ijacobian.Xdotstate == Xdotstate && ts->ijacobian.imex == imex))) {
    *flg = ts->ijacobian.mstructure;
    ierr = MatScale(*A, shift / ts->ijacobian.shift);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!ts->userops->rhsjacobian && !ts->userops->ijacobian) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetRHSJacobian() and / or TSSetIJacobian()");

  *flg = SAME_NONZERO_PATTERN;  /* In case we're solving a linear problem in which case it wouldn't get initialized below. */
  ierr = PetscLogEventBegin(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
  if (ts->userops->ijacobian) {
    *flg = DIFFERENT_NONZERO_PATTERN;
    PetscStackPush("TS user implicit Jacobian");
    ierr = (*ts->userops->ijacobian)(ts,t,X,Xdot,shift,A,B,flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
    /* make sure user returned a correct Jacobian and preconditioner */
    PetscValidHeaderSpecific(*A,MAT_CLASSID,4);
    PetscValidHeaderSpecific(*B,MAT_CLASSID,5);
  }
  if (imex) {
    if (!ts->userops->ijacobian) {  /* system was written as Xdot = F(t,X) */
      ierr = MatZeroEntries(*A);CHKERRQ(ierr);
      ierr = MatShift(*A,shift);CHKERRQ(ierr);
      if (*A != *B) {
        ierr = MatZeroEntries(*B);CHKERRQ(ierr);
        ierr = MatShift(*B,shift);CHKERRQ(ierr);
      }
      *flg = SAME_PRECONDITIONER;
    }
  } else {
    if (!ts->userops->ijacobian) {
      ierr = TSComputeRHSJacobian(ts,t,X,A,B,flg);CHKERRQ(ierr);
      ierr = MatScale(*A,-1);CHKERRQ(ierr);
      ierr = MatShift(*A,shift);CHKERRQ(ierr);
      if (*A != *B) {
        ierr = MatScale(*B,-1);CHKERRQ(ierr);
        ierr = MatShift(*B,shift);CHKERRQ(ierr);
      }
    } else if (ts->userops->rhsjacobian) {
      Mat Arhs,Brhs;
      MatStructure axpy,flg2 = DIFFERENT_NONZERO_PATTERN;
      ierr = TSGetRHSMats_Private(ts,&Arhs,&Brhs);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(ts,t,X,&Arhs,&Brhs,&flg2);CHKERRQ(ierr);
      axpy = (*flg == flg2) ? SAME_NONZERO_PATTERN : DIFFERENT_NONZERO_PATTERN;
      ierr = MatAXPY(*A,-1,Arhs,axpy);CHKERRQ(ierr);
      if (*A != *B) {
        ierr = MatAXPY(*B,-1,Brhs,axpy);CHKERRQ(ierr);
      }
      *flg = PetscMin(*flg,flg2);
    }
  }

  ts->ijacobian.time = t;
  ts->ijacobian.X = X;
  ts->ijacobian.Xdot = Xdot;
  ierr = PetscObjectStateQuery((PetscObject)X,&ts->ijacobian.Xstate);CHKERRQ(ierr);
  ierr = PetscObjectStateQuery((PetscObject)Xdot,&ts->ijacobian.Xdotstate);CHKERRQ(ierr);
  ts->ijacobian.shift = shift;
  ts->ijacobian.imex = imex;
  ts->ijacobian.mstructure = *flg;
  ierr = PetscLogEventEnd(TS_JacobianEval,ts,X,*A,*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetRHSFunction"
/*@C
    TSSetRHSFunction - Sets the routine for evaluating the function,
    F(t,u), where U_t = F(t,u).

    Logically Collective on TS

    Input Parameters:
+   ts - the TS context obtained from TSCreate()
.   r - vector to put the computed right hand side (or PETSC_NULL to have it created)
.   f - routine for evaluating the right-hand-side function
-   ctx - [optional] user-defined context for private data for the 
          function evaluation routine (may be PETSC_NULL)

    Calling sequence of func:
$     func (TS ts,PetscReal t,Vec u,Vec F,void *ctx);

+   t - current timestep
.   u - input vector
.   F - function vector
-   ctx - [optional] user-defined function context 

    Level: beginner

.keywords: TS, timestep, set, right-hand-side, function

.seealso: TSSetRHSJacobian(), TSSetIJacobian()
@*/
PetscErrorCode  TSSetRHSFunction(TS ts,Vec r,PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            ralloc = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,2);
  if (f)   ts->userops->rhsfunction = f;
  if (ctx) ts->funP                 = ctx;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!r && !ts->dm && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&ralloc);CHKERRQ(ierr);
    r = ralloc;
  }
  ierr = SNESSetFunction(snes,r,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = VecDestroy(&ralloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetRHSJacobian"
/*@C
   TSSetRHSJacobian - Sets the function to compute the Jacobian of F,
   where U_t = F(U,t), as well as the location to store the matrix.

   Logically Collective on TS

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

.seealso: SNESDefaultComputeJacobianColor(), TSSetRHSFunction()

@*/
PetscErrorCode  TSSetRHSJacobian(TS ts,Mat A,Mat B,TSRHSJacobian f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(ts,1,A,2);
  if (B) PetscCheckSameComm(ts,1,B,3);

  if (f)   ts->userops->rhsjacobian = f;
  if (ctx) ts->jacP                 = ctx;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!ts->userops->ijacobian) {
    ierr = SNESSetJacobian(snes,A,B,SNESTSFormJacobian,ts);CHKERRQ(ierr);
  }
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Arhs);CHKERRQ(ierr);
    ts->Arhs = A;
  }
  if (B) {
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->Brhs);CHKERRQ(ierr);
    ts->Brhs = B;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction"
/*@C
   TSSetIFunction - Set the function to compute F(t,U,U_t) where F = 0 is the DAE to be solved.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  r   - vector to hold the residual (or PETSC_NULL to have it created internally)
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
   The user MUST call either this routine, TSSetRHSFunction().  This routine must be used when not solving an ODE, for example a DAE.

   Level: beginner

.keywords: TS, timestep, set, DAE, Jacobian

.seealso: TSSetRHSJacobian(), TSSetRHSFunction(), TSSetIJacobian()
@*/
PetscErrorCode  TSSetIFunction(TS ts,Vec res,TSIFunction f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            resalloc = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (res) PetscValidHeaderSpecific(res,VEC_CLASSID,2);
  if (f)   ts->userops->ifunction = f;
  if (ctx) ts->funP           = ctx;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  if (!res && !ts->dm && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&resalloc);CHKERRQ(ierr);
    res = resalloc;
  }
  ierr = SNESSetFunction(snes,res,SNESTSFormFunction,ts);CHKERRQ(ierr);
  ierr = VecDestroy(&resalloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetIFunction"
/*@C
   TSGetIFunction - Returns the vector where the implicit residual is stored and the function/contex to compute it.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
+  r - vector to hold residual (or PETSC_NULL)
.  func - the function to compute residual (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: TS, nonlinear, get, function

.seealso: TSSetIFunction(), SNESGetFunction()
@*/
PetscErrorCode TSGetIFunction(TS ts,Vec *r,TSIFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  SNES snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,r,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (func) *func = ts->userops->ifunction;
  if (ctx)  *ctx  = ts->funP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetRHSFunction"
/*@C
   TSGetRHSFunction - Returns the vector where the right hand side is stored and the function/context to compute it.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
+  r - vector to hold computed right hand side (or PETSC_NULL)
.  func - the function to compute right hand side (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: TS, nonlinear, get, function

.seealso: TSSetRhsfunction(), SNESGetFunction()
@*/
PetscErrorCode TSGetRHSFunction(TS ts,Vec *r,TSRHSFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  SNES snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,r,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (func) *func = ts->userops->rhsfunction;
  if (ctx)  *ctx  = ts->funP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian"
/*@C
   TSSetIJacobian - Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function
        you provided with TSSetIFunction().  

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  A   - Jacobian matrix
.  B   - preconditioning matrix for A (may be same as A)
.  f   - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  a    - shift
.  A    - Jacobian of G(U) = F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
.  B    - preconditioning matrix for A, may be same as A
.  flag - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Notes:
   The matrices A and B are exactly the matrices that are used by SNES for the nonlinear solve.

   The matrix dF/dU + a*dF/dU_t you provide turns out to be 
   the Jacobian of G(U) = F(t,U,W+a*U) where F(t,U,U_t) = 0 is the DAE to be solved.
   The time integrator internally approximates U_t by W+a*U where the positive "shift"
   a and vector W depend on the integration method, step size, and past states. For example with 
   the backward Euler method a = 1/dt and W = -a*U(previous timestep) so
   W + a*U = a*(U - U(previous timestep)) = (U - U(previous timestep))/dt

   Level: beginner

.keywords: TS, timestep, DAE, Jacobian

.seealso: TSSetIFunction(), TSSetRHSJacobian(), SNESDefaultComputeJacobianColor(), SNESDefaultComputeJacobian()

@*/
PetscErrorCode  TSSetIJacobian(TS ts,Mat A,Mat B,TSIJacobian f,void *ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(ts,1,A,2);
  if (B) PetscCheckSameComm(ts,1,B,3);
  if (f)   ts->userops->ijacobian = f;
  if (ctx) ts->jacP           = ctx;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,B,SNESTSFormJacobian,ts);CHKERRQ(ierr);
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
PetscErrorCode  TSView(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  const TSType   type;
  PetscBool      iascii,isstring,isundials;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)ts)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ts,viewer,"TS Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum steps=%D\n",ts->max_steps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum time=%G\n",ts->max_time);CHKERRQ(ierr);
    if (ts->problem_type == TS_NONLINEAR) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%D\n",ts->snes_its);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solve failures=%D\n",ts->num_snes_failures);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",ts->ksp_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of rejected steps=%D\n",ts->reject);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = TSGetType(ts,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",type);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSSUNDIALS,&isundials);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetApplicationContext"
/*@
   TSSetApplicationContext - Sets an optional user-defined context for
   the timesteppers.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  usrP - optional user context

   Level: intermediate

.keywords: TS, timestep, set, application, context

.seealso: TSGetApplicationContext()
@*/
PetscErrorCode  TSSetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->user = usrP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetApplicationContext"
/*@
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
PetscErrorCode  TSGetApplicationContext(TS ts,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *(void**)usrP = ts->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetTimeStepNumber"
/*@
   TSGetTimeStepNumber - Gets the number of time steps completed.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  iter - number of steps completed so far

   Level: intermediate

.keywords: TS, timestep, get, iteration, number
.seealso: TSGetTime(), TSGetTimeStep(), TSSetPreStep(), TSSetPreStage(), TSSetPostStep()
@*/
PetscErrorCode  TSGetTimeStepNumber(TS ts,PetscInt* iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(iter,2);
  *iter = ts->steps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetInitialTimeStep"
/*@
   TSSetInitialTimeStep - Sets the initial timestep to be used,
   as well as the initial time.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  initial_time - the initial time
-  time_step - the size of the timestep

   Level: intermediate

.seealso: TSSetTimeStep(), TSGetTimeStep()

.keywords: TS, set, initial, timestep
@*/
PetscErrorCode  TSSetInitialTimeStep(TS ts,PetscReal initial_time,PetscReal time_step)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = TSSetTimeStep(ts,time_step);CHKERRQ(ierr);
  ierr = TSSetTime(ts,initial_time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetTimeStep"
/*@
   TSSetTimeStep - Allows one to reset the timestep at any time,
   useful for simple pseudo-timestepping codes.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time_step - the size of the timestep

   Level: intermediate

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, set, timestep
@*/
PetscErrorCode  TSSetTimeStep(TS ts,PetscReal time_step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time_step,2);
  ts->time_step = time_step;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetExactFinalTime"
/*@
   TSSetExactFinalTime - Determines whether to interpolate solution to the
      exact final time requested by the user or just returns it at the final time
      it computed.

  Logically Collective on TS

   Input Parameter:
+   ts - the time-step context
-   ft - PETSC_TRUE if interpolates, else PETSC_FALSE

   Level: beginner

.seealso: TSSetDuration()
@*/
PetscErrorCode  TSSetExactFinalTime(TS ts,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ts,flg,2);
  ts->exact_final_time = flg;
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
PetscErrorCode  TSGetTimeStep(TS ts,PetscReal* dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
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
PetscErrorCode  TSGetSolution(TS ts,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(v,2);
  *v = ts->vec_sol;
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
PetscErrorCode  TSSetProblemType(TS ts, TSProblemType type) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->problem_type = type;
  if (type == TS_LINEAR) {
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetType(snes,SNESKSPONLY);CHKERRQ(ierr);
  }
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
         M U_t = A U
         M(t) U_t = A(t) U
         U_t = F(t,U)
.ve

   Level: beginner

.keywords: TS, problem type
.seealso: TSSetUp(), TSProblemType, TS
@*/
PetscErrorCode  TSGetProblemType(TS ts, TSProblemType *type) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
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
PetscErrorCode  TSSetUp(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->setupcalled) PetscFunctionReturn(0);

  if (!((PetscObject)ts)->type_name) {
    ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  }
  if (ts->exact_final_time == PETSC_DECIDE) ts->exact_final_time = PETSC_FALSE;

  if (!ts->vec_sol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TSSetSolution() first");

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);

  if (ts->ops->setup) {
    ierr = (*ts->ops->setup)(ts);CHKERRQ(ierr);
  }

  ts->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset"
/*@
   TSReset - Resets a TS context and removes any allocated Vecs and Mats.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: beginner

.keywords: TS, timestep, reset

.seealso: TSCreate(), TSSetup(), TSDestroy()
@*/
PetscErrorCode  TSReset(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->reset) {
    ierr = (*ts->ops->reset)(ts);CHKERRQ(ierr);
  }
  if (ts->snes) {ierr = SNESReset(ts->snes);CHKERRQ(ierr);}
  ierr = MatDestroy(&ts->Arhs);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->Brhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->Frhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vatol);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vrtol);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->nwork,&ts->work);CHKERRQ(ierr);
  ts->setupcalled = PETSC_FALSE;
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
PetscErrorCode  TSDestroy(TS *ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ts) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ts),TS_CLASSID,1);
  if (--((PetscObject)(*ts))->refct > 0) {*ts = 0; PetscFunctionReturn(0);}

  ierr = TSReset((*ts));CHKERRQ(ierr);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish((*ts));CHKERRQ(ierr);
  if ((*ts)->ops->destroy) {ierr = (*(*ts)->ops->destroy)((*ts));CHKERRQ(ierr);}

  ierr = TSAdaptDestroy(&(*ts)->adapt);CHKERRQ(ierr);
  ierr = SNESDestroy(&(*ts)->snes);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ts)->dm);CHKERRQ(ierr);
  ierr = TSMonitorCancel((*ts));CHKERRQ(ierr);

  ierr = PetscFree((*ts)->userops);

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
PetscErrorCode  TSGetSNES(TS ts,SNES *snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(snes,2);
  if (!ts->snes) {
    ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,ts->snes);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);
    if (ts->dm) {ierr = SNESSetDM(ts->snes,ts->dm);CHKERRQ(ierr);}
    if (ts->problem_type == TS_LINEAR) {
      ierr = SNESSetType(ts->snes,SNESKSPONLY);CHKERRQ(ierr);
    }
  }
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
PetscErrorCode  TSGetKSP(TS ts,KSP *ksp)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!((PetscObject)ts)->type_name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"KSP is not created yet. Call TSSetType() first");
  if (ts->problem_type != TS_LINEAR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Linear only; use TSGetSNES()");
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNCT__
#define __FUNCT__ "TSGetDuration"
/*@
   TSGetDuration - Gets the maximum number of timesteps to use and
   maximum time for iteration.

   Not Collective

   Input Parameters:
+  ts       - the TS context obtained from TSCreate()
.  maxsteps - maximum number of iterations to use, or PETSC_NULL
-  maxtime  - final time to iterate to, or PETSC_NULL

   Level: intermediate

.keywords: TS, timestep, get, maximum, iterations, time
@*/
PetscErrorCode  TSGetDuration(TS ts, PetscInt *maxsteps, PetscReal *maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  if (maxsteps) {
    PetscValidIntPointer(maxsteps,2);
    *maxsteps = ts->max_steps;
  }
  if (maxtime) {
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

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  maxsteps - maximum number of iterations to use
-  maxtime - final time to iterate to

   Options Database Keys:
.  -ts_max_steps <maxsteps> - Sets maxsteps
.  -ts_final_time <maxtime> - Sets maxtime

   Notes:
   The default maximum number of iterations is 5000. Default time is 5.0

   Level: intermediate

.keywords: TS, timestep, set, maximum, iterations

.seealso: TSSetExactFinalTime()
@*/
PetscErrorCode  TSSetDuration(TS ts,PetscInt maxsteps,PetscReal maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,maxsteps,2);
  PetscValidLogicalCollectiveReal(ts,maxtime,2);
  if (maxsteps >= 0) ts->max_steps = maxsteps;
  if (maxtime != PETSC_DEFAULT) ts->max_time  = maxtime;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution"
/*@
   TSSetSolution - Sets the initial solution vector
   for use by the TS routines.

   Logically Collective on TS and Vec

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  x - the solution vector

   Level: beginner

.keywords: TS, timestep, set, solution, initial conditions
@*/
PetscErrorCode  TSSetSolution(TS ts,Vec x)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_sol);CHKERRQ(ierr);
  ts->vec_sol = x;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMShellSetGlobalVector(dm,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetPreStep"
/*@C
  TSSetPreStep - Sets the general-purpose function
  called once at the beginning of each time step.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. func (TS ts);

  Level: intermediate

  Note:
  If a step is rejected, TSStep() will call this routine again before each attempt.
  The last completed time step number can be queried using TSGetTimeStepNumber(), the
  size of the step being attempted can be obtained using TSGetTimeStep().

.keywords: TS, timestep
.seealso: TSSetPreStage(), TSSetPostStep(), TSStep()
@*/
PetscErrorCode  TSSetPreStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->ops->prestep = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPreStep"
/*@
  TSPreStep - Runs the user-defined pre-step function.

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPreStep() is typically used within time stepping implementations,
  so most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, timestep
.seealso: TSSetPreStep(), TSPreStage(), TSPostStep()
@*/
PetscErrorCode  TSPreStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->prestep) {
    PetscStackPush("TS PreStep function");
    ierr = (*ts->ops->prestep)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetPreStage"
/*@C
  TSSetPreStage - Sets the general-purpose function
  called once at the beginning of each stage.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
. PetscErrorCode func(TS ts, PetscReal stagetime);

  Level: intermediate

  Note:
  There may be several stages per time step. If the solve for a given stage fails, the step may be rejected and retried.
  The time step number being computed can be queried using TSGetTimeStepNumber() and the total size of the step being
  attempted can be obtained using TSGetTimeStep(). The time at the start of the step is available via TSGetTime().

.keywords: TS, timestep
.seealso: TSSetPreStep(), TSSetPostStep(), TSGetApplicationContext()
@*/
PetscErrorCode  TSSetPreStage(TS ts, PetscErrorCode (*func)(TS,PetscReal))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->ops->prestage = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPreStage"
/*@
  TSPreStage - Runs the user-defined pre-stage function set using TSSetPreStage()

  Collective on TS

  Input Parameters:
. ts   - The TS context obtained from TSCreate()

  Notes:
  TSPreStage() is typically used within time stepping implementations,
  most users would not generally call this routine themselves.

  Level: developer

.keywords: TS, timestep
.seealso: TSSetPreStep(), TSPreStep(), TSPostStep()
@*/
PetscErrorCode  TSPreStage(TS ts, PetscReal stagetime)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->prestage) {
    PetscStackPush("TS PreStage function");
    ierr = (*ts->ops->prestage)(ts,stagetime);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetPostStep"
/*@C
  TSSetPostStep - Sets the general-purpose function
  called once at the end of each time step.

  Logically Collective on TS

  Input Parameters:
+ ts   - The TS context obtained from TSCreate()
- func - The function

  Calling sequence of func:
$ func (TS ts);

  Level: intermediate

.keywords: TS, timestep
.seealso: TSSetPreStep(), TSSetPreStage(), TSGetTimeStep(), TSGetTimeStepNumber(), TSGetTime()
@*/
PetscErrorCode  TSSetPostStep(TS ts, PetscErrorCode (*func)(TS))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ts->ops->poststep = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPostStep"
/*@
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
PetscErrorCode  TSPostStep(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->ops->poststep) {
    PetscStackPush("TS PostStep function");
    ierr = (*ts->ops->poststep)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSet"
/*@C
   TSMonitorSet - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  monitor - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
             monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling sequence of monitor:
$    int monitor(TS ts,PetscInt steps,PetscReal time,Vec x,void *mctx)

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
PetscErrorCode  TSMonitorSet(TS ts,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *mctx,PetscErrorCode (*mdestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->numbermonitors >= MAXTSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  ts->monitor[ts->numbermonitors]           = monitor;
  ts->mdestroy[ts->numbermonitors]          = mdestroy;
  ts->monitorcontext[ts->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorCancel"
/*@C
   TSMonitorCancel - Clears all the monitors that have been set on a time-step object.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.keywords: TS, timestep, set, monitor

.seealso: TSMonitorDefault(), TSMonitorSet()
@*/
PetscErrorCode  TSMonitorCancel(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->mdestroy[i]) {
      ierr = (*ts->mdestroy[i])(&ts->monitorcontext[i]);CHKERRQ(ierr);
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
PetscErrorCode TSMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ts)->comm);

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D TS dt %g time %g\n",step,(double)ts->time_step,(double)ptime);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetRetainStages"
/*@
   TSSetRetainStages - Request that all stages in the upcoming step be stored so that interpolation will be available.

   Logically Collective on TS

   Input Argument:
.  ts - time stepping context

   Output Argument:
.  flg - PETSC_TRUE or PETSC_FALSE

   Level: intermediate

.keywords: TS, set

.seealso: TSInterpolate(), TSSetPostStep()
@*/
PetscErrorCode TSSetRetainStages(TS ts,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->retain_stages = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSInterpolate"
/*@
   TSInterpolate - Interpolate the solution computed during the previous step to an arbitrary location in the interval

   Collective on TS

   Input Argument:
+  ts - time stepping context
-  t - time to interpolate to

   Output Argument:
.  X - state at given time

   Notes:
   The user should call TSSetRetainStages() before taking a step in which interpolation will be requested.

   Level: intermediate

   Developer Notes:
   TSInterpolate() and the storing of previous steps/stages should be generalized to support delay differential equations and continuous adjoints.

.keywords: TS, set

.seealso: TSSetRetainStages(), TSSetPostStep()
@*/
PetscErrorCode TSInterpolate(TS ts,PetscReal t,Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (t < ts->ptime - ts->time_step_prev || t > ts->ptime) SETERRQ3(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Requested time %G not in last time steps [%G,%G]",t,ts->ptime-ts->time_step_prev,ts->ptime);
  if (!ts->ops->interpolate) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_SUP,"%s does not provide interpolation",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->interpolate)(ts,t,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSStep"
/*@
   TSStep - Steps one time step

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Level: intermediate

   Notes:
   The hook set using TSSetPreStep() is called before each attempt to take the step. In general, the time step size may
   be changed due to adaptive error controller or solve failures. Note that steps may contain multiple stages.

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetUp(), TSDestroy(), TSSolve(), TSSetPreStep(), TSSetPreStage()
@*/
PetscErrorCode  TSStep(TS ts)
{
  PetscReal      ptime_prev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID,1);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  ts->reason = TS_CONVERGED_ITERATING;

  ptime_prev = ts->ptime;
  ierr = PetscLogEventBegin(TS_Step,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*ts->ops->step)(ts);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_Step,ts,0,0,0);CHKERRQ(ierr);
  ts->time_step_prev = ts->ptime - ptime_prev;

  if (ts->reason < 0) {
    if (ts->errorifstepfailed) {
      if (ts->reason == TS_DIVERGED_NONLINEAR_SOLVE) {
        SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s, increase -ts_max_snes_failures or make negative to attempt recovery",TSConvergedReasons[ts->reason]);
      } else SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_NOT_CONVERGED,"TSStep has failed due to %s",TSConvergedReasons[ts->reason]);
    }
  } else if (!ts->reason) {
    if (ts->steps >= ts->max_steps)
      ts->reason = TS_CONVERGED_ITS;
    else if (ts->ptime >= ts->max_time)
      ts->reason = TS_CONVERGED_TIME;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSEvaluateStep"
/*@
   TSEvaluateStep - Evaluate the solution at the end of a time step with a given order of accuracy.

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  order - desired order of accuracy
-  done - whether the step was evaluated at this order (pass PETSC_NULL to generate an error if not available)

   Output Arguments:
.  X - state at the end of the current step

   Level: advanced

   Notes:
   This function cannot be called until all stages have been evaluated.
   It is normally called by adaptive controllers before a step has been accepted and may also be called by the user after TSStep() has returned.

.seealso: TSStep(), TSAdapt
@*/
PetscErrorCode TSEvaluateStep(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidType(ts,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  if (!ts->ops->evaluatestep) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_SUP,"TSEvaluateStep not implemented for type '%s'",((PetscObject)ts)->type_name);
  ierr = (*ts->ops->evaluatestep)(ts,order,X,done);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve"
/*@
   TSSolve - Steps the requested number of timesteps.

   Collective on TS

   Input Parameter:
+  ts - the TS context obtained from TSCreate()
-  x - the solution vector

   Output Parameter:
.  ftime - time of the state vector x upon completion

   Level: beginner

   Notes:
   The final time returned by this function may be different from the time of the internally
   held state accessible by TSGetSolution() and TSGetTime() because the method may have
   stepped over the final time.

.keywords: TS, timestep, solve

.seealso: TSCreate(), TSSetSolution(), TSStep()
@*/
PetscErrorCode TSSolve(TS ts,Vec x,PetscReal *ftime)
{
  PetscBool      flg;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  if (ts->exact_final_time) {   /* Need ts->vec_sol to be distinct so it is not overwritten when we interpolate at the end */
    if (!ts->vec_sol || x == ts->vec_sol) {
      Vec y;
      ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
      ierr = TSSetSolution(ts,y);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr); /* grant ownership */
    }
    ierr = VecCopy(x,ts->vec_sol);CHKERRQ(ierr);
  } else {
    ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  }
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  /* reset time step and iteration counters */
  ts->steps = 0;
  ts->ksp_its = 0;
  ts->snes_its = 0;
  ts->num_snes_failures = 0;
  ts->reject = 0;
  ts->reason = TS_CONVERGED_ITERATING;

  if (ts->ops->solve) {         /* This private interface is transitional and should be removed when all implementations are updated. */
    ierr = (*ts->ops->solve)(ts);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,x);CHKERRQ(ierr);
    if (ftime) *ftime = ts->ptime;
  } else {
    /* steps the requested number of timesteps. */
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    if (ts->steps >= ts->max_steps)
      ts->reason = TS_CONVERGED_ITS;
    else if (ts->ptime >= ts->max_time)
      ts->reason = TS_CONVERGED_TIME;
    while (!ts->reason) {
      ierr = TSStep(ts);CHKERRQ(ierr);
      ierr = TSPostStep(ts);CHKERRQ(ierr);
      ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
    }
    if (ts->exact_final_time && ts->ptime > ts->max_time) {
      ierr = TSInterpolate(ts,ts->max_time,x);CHKERRQ(ierr);
      if (ftime) *ftime = ts->max_time;
    } else {
      ierr = VecCopy(ts->vec_sol,x);CHKERRQ(ierr);
      if (ftime) *ftime = ts->ptime;
    }
  }
  ierr = PetscOptionsGetString(((PetscObject)ts)->prefix,"-ts_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)ts)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = TSView(ts,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitor"
/*@
   TSMonitor - Runs all user-provided monitor routines set using TSMonitorSet()

   Collective on TS

   Input Parameters:
+  ts - time stepping context obtained from TSCreate()
.  step - step number that has just completed
.  ptime - model time of the state
-  x - state at the current model time

   Notes:
   TSMonitor() is typically used within the time stepping implementations.
   Users might call this function when using the TSStep() interface instead of TSSolve().

   Level: advanced

.keywords: TS, timestep
@*/
PetscErrorCode TSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i,n = ts->numbermonitors;

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
PetscErrorCode  TSMonitorLGCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
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
PetscErrorCode  TSMonitorLGDestroy(PetscDrawLG *drawlg)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(*drawlg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetTime"
/*@
   TSGetTime - Gets the time of the most recently completed step.

   Not Collective

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Output Parameter:
.  t  - the current time

   Level: beginner

   Note:
   When called during time step evaluation (e.g. during residual evaluation or via hooks set using TSSetPreStep(),
   TSSetPreStage(), or TSSetPostStep()), the time is the time at the start of the step being evaluated.

.seealso: TSSetInitialTimeStep(), TSGetTimeStep()

.keywords: TS, get, time
@*/
PetscErrorCode  TSGetTime(TS ts,PetscReal* t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidDoublePointer(t,2);
  *t = ts->ptime;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetTime"
/*@
   TSSetTime - Allows one to reset the time.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
-  time - the time

   Level: intermediate

.seealso: TSGetTime(), TSSetDuration()

.keywords: TS, set, time
@*/
PetscErrorCode  TSSetTime(TS ts, PetscReal t) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t,2);
  ts->ptime = t;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetOptionsPrefix" 
/*@C
   TSSetOptionsPrefix - Sets the prefix used for searching for all
   TS options in the database.

   Logically Collective on TS

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
PetscErrorCode  TSSetOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSAppendOptionsPrefix" 
/*@C
   TSAppendOptionsPrefix - Appends to the prefix used for searching for all
   TS options in the database.

   Logically Collective on TS

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
PetscErrorCode  TSAppendOptionsPrefix(TS ts,const char prefix[])
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ts,prefix);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESAppendOptionsPrefix(snes,prefix);CHKERRQ(ierr);
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
PetscErrorCode  TSGetOptionsPrefix(TS ts,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
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
.  func - Function to compute the Jacobian of the RHS
-  ctx - User-defined context for Jacobian evaluation routine

   Notes: You can pass in PETSC_NULL for any return argument you do not need.

   Level: intermediate

.seealso: TSGetTimeStep(), TSGetMatrices(), TSGetTime(), TSGetTimeStepNumber()

.keywords: TS, timestep, get, matrix, Jacobian
@*/
PetscErrorCode  TSGetRHSJacobian(TS ts,Mat *J,Mat *M,TSRHSJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,J,M,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (func) *func = ts->userops->rhsjacobian;
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
PetscErrorCode  TSGetIJacobian(TS ts,Mat *A,Mat *B,TSIJacobian *f,void **ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,A,B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (f) *f = ts->userops->ijacobian;
  if (ctx) *ctx = ts->jacP;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscViewer viewer;
  Vec         initialsolution;
  PetscBool   showinitial;
} TSMonitorSolutionCtx;

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
PetscErrorCode  TSMonitorSolution(TS ts,PetscInt step,PetscReal ptime,Vec x,void *dummy)
{
  PetscErrorCode       ierr;
  TSMonitorSolutionCtx *ictx = (TSMonitorSolutionCtx*)dummy;

  PetscFunctionBegin;
  if (!step && ictx->showinitial) {
    if (!ictx->initialsolution) {
      ierr = VecDuplicate(x,&ictx->initialsolution);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,ictx->initialsolution);CHKERRQ(ierr);
  }
  if (ictx->showinitial) {
    PetscReal pause;
    ierr = PetscViewerDrawGetPause(ictx->viewer,&pause);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetPause(ictx->viewer,0.0);CHKERRQ(ierr);
    ierr = VecView(ictx->initialsolution,ictx->viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetPause(ictx->viewer,pause);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetHold(ictx->viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = VecView(x,ictx->viewer);CHKERRQ(ierr);
  if (ictx->showinitial) {
    ierr = PetscViewerDrawSetHold(ictx->viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSMonitorSolutionDestroy"
/*@C
   TSMonitorSolutionDestroy - Destroys the monitor context for TSMonitorSolution

   Collective on TS

   Input Parameters:
.    ctx - the monitor context

   Level: intermediate

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorSolution()
@*/
PetscErrorCode  TSMonitorSolutionDestroy(void **ctx)
{
  PetscErrorCode       ierr;
  TSMonitorSolutionCtx *ictx = *(TSMonitorSolutionCtx**)ctx;
 
  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&ictx->viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&ictx->initialsolution);CHKERRQ(ierr);
  ierr = PetscFree(ictx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSolutionCreate"
/*@C
   TSMonitorSolutionCreate - Creates the monitor context for TSMonitorSolution

   Collective on TS

   Input Parameter:
.    ts - time-step context

   Output Patameter:
.    ctx - the monitor context

   Level: intermediate

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorSolution()
@*/
PetscErrorCode  TSMonitorSolutionCreate(TS ts,PetscViewer viewer,void **ctx)
{
  PetscErrorCode       ierr;
  TSMonitorSolutionCtx *ictx;
 
  PetscFunctionBegin;
  ierr = PetscNew(TSMonitorSolutionCtx,&ictx);CHKERRQ(ierr);
  *ctx = (void*)ictx;
  if (!viewer) {
    viewer = PETSC_VIEWER_DRAW_(((PetscObject)ts)->comm);
  }
  ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  ictx->viewer      = viewer;
  ictx->showinitial = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)ts)->prefix,"-ts_monitor_solution_initial",&ictx->showinitial,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetDM"
/*@
   TSSetDM - Sets the DM that may be used by some preconditioners

   Logically Collective on TS and DM

   Input Parameters:
+  ts - the preconditioner context
-  dm - the dm

   Level: intermediate


.seealso: TSGetDM(), SNESSetDM(), SNESGetDM()
@*/
PetscErrorCode  TSSetDM(TS ts,DM dm)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  ierr = DMDestroy(&ts->dm);CHKERRQ(ierr);
  ts->dm = dm;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetDM"
/*@
   TSGetDM - Gets the DM that may be used by some preconditioners

   Not Collective

   Input Parameter:
. ts - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate


.seealso: TSSetDM(), SNESSetDM(), SNESGetDM()
@*/
PetscErrorCode  TSGetDM(TS ts,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->dm) {
    ierr = DMShellCreate(((PetscObject)ts)->comm,&ts->dm);CHKERRQ(ierr);
    if (ts->snes) {ierr = SNESSetDM(ts->snes,ts->dm);CHKERRQ(ierr);}
  }
  *dm = ts->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction"
/*@
   SNESTSFormFunction - Function to evaluate nonlinear residual

   Logically Collective on SNES

   Input Parameter:
+ snes - nonlinear solver
. X - the current state at which to evaluate the residual
- ctx - user context, must be a TS

   Output Parameter:
. F - the nonlinear residual

   Notes:
   This function is not normally called by users and is automatically registered with the SNES used by TS.
   It is most frequently passed to MatFDColoringSetFunction().

   Level: advanced

.seealso: SNESSetFunction(), MatFDColoringSetFunction()
@*/
PetscErrorCode  SNESTSFormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  TS ts = (TS)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscValidHeaderSpecific(ts,TS_CLASSID,4);
  ierr = (ts->ops->snesfunction)(snes,X,F,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian"
/*@
   SNESTSFormJacobian - Function to evaluate the Jacobian

   Collective on SNES

   Input Parameter:
+ snes - nonlinear solver
. X - the current state at which to evaluate the residual
- ctx - user context, must be a TS

   Output Parameter:
+ A - the Jacobian
. B - the preconditioning matrix (may be the same as A)
- flag - indicates any structure change in the matrix

   Notes:
   This function is not normally called by users and is automatically registered with the SNES used by TS.

   Level: developer

.seealso: SNESSetJacobian()
@*/
PetscErrorCode  SNESTSFormJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  TS ts = (TS)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidPointer(A,3);
  PetscValidHeaderSpecific(*A,MAT_CLASSID,3);
  PetscValidPointer(B,4);
  PetscValidHeaderSpecific(*B,MAT_CLASSID,4);
  PetscValidPointer(flag,5);
  PetscValidHeaderSpecific(ts,TS_CLASSID,6);
  ierr = (ts->ops->snesjacobian)(snes,X,A,B,flag,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeRHSFunctionLinear"
/*@C
   TSComputeRHSFunctionLinear - Evaluate the right hand side via the user-provided Jacobian, for linear problems only

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  t - time at which to evaluate
.  X - state at which to evaluate
-  ctx - context

   Output Arguments:
.  F - right hand side

   Level: intermediate

   Notes:
   This function is intended to be passed to TSSetRHSFunction() to evaluate the right hand side for linear problems.
   The matrix (and optionally the evaluation context) should be passed to TSSetRHSJacobian().

.seealso: TSSetRHSFunction(), TSSetRHSJacobian(), TSComputeRHSJacobianConstant()
@*/
PetscErrorCode TSComputeRHSFunctionLinear(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Mat Arhs,Brhs;
  MatStructure flg2;

  PetscFunctionBegin;
  ierr = TSGetRHSMats_Private(ts,&Arhs,&Brhs);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(ts,t,X,&Arhs,&Brhs,&flg2);CHKERRQ(ierr);
  ierr = MatMult(Arhs,X,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeRHSJacobianConstant"
/*@C
   TSComputeRHSJacobianConstant - Reuses a Jacobian that is time-independent.

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  t - time at which to evaluate
.  X - state at which to evaluate
-  ctx - context

   Output Arguments:
+  A - pointer to operator
.  B - pointer to preconditioning matrix
-  flg - matrix structure flag

   Level: intermediate

   Notes:
   This function is intended to be passed to TSSetRHSJacobian() to evaluate the Jacobian for linear time-independent problems.

.seealso: TSSetRHSFunction(), TSSetRHSJacobian(), TSComputeRHSFunctionLinear()
@*/
PetscErrorCode TSComputeRHSJacobianConstant(TS ts,PetscReal t,Vec X,Mat *A,Mat *B,MatStructure *flg,void *ctx)
{

  PetscFunctionBegin;
  *flg = SAME_PRECONDITIONER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunctionLinear"
/*@C
   TSComputeIFunctionLinear - Evaluate the left hand side via the user-provided Jacobian, for linear problems only

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  t - time at which to evaluate
.  X - state at which to evaluate
.  Xdot - time derivative of state vector
-  ctx - context

   Output Arguments:
.  F - left hand side

   Level: intermediate

   Notes:
   The assumption here is that the left hand side is of the form A*Xdot (and not A*Xdot + B*X). For other cases, the
   user is required to write their own TSComputeIFunction.
   This function is intended to be passed to TSSetIFunction() to evaluate the left hand side for linear problems.
   The matrix (and optionally the evaluation context) should be passed to TSSetIJacobian().

.seealso: TSSetIFunction(), TSSetIJacobian(), TSComputeIJacobianConstant()
@*/
PetscErrorCode TSComputeIFunctionLinear(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Mat A,B;
  MatStructure flg2;

  PetscFunctionBegin;
  ierr = TSGetIJacobian(ts,&A,&B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(ts,t,X,Xdot,1.0,&A,&B,&flg2,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMult(A,Xdot,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobianConstant"
/*@C
   TSComputeIJacobianConstant - Reuses a Jacobian that is time-independent.

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  t - time at which to evaluate
.  X - state at which to evaluate
.  Xdot - time derivative of state vector
.  shift - shift to apply
-  ctx - context

   Output Arguments:
+  A - pointer to operator
.  B - pointer to preconditioning matrix
-  flg - matrix structure flag

   Level: intermediate

   Notes:
   This function is intended to be passed to TSSetIJacobian() to evaluate the Jacobian for linear time-independent problems.

.seealso: TSSetIFunction(), TSSetIJacobian(), TSComputeIFunctionLinear()
@*/
PetscErrorCode TSComputeIJacobianConstant(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *flg,void *ctx)
{

  PetscFunctionBegin;
  *flg = SAME_PRECONDITIONER;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSGetConvergedReason"
/*@
   TSGetConvergedReason - Gets the reason the TS iteration was stopped.

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see TSConvergedReason or the 
            manual pages for the individual convergence tests for complete lists

   Level: intermediate

   Notes:
   Can only be called after the call to TSSolve() is complete.

.keywords: TS, nonlinear, set, convergence, test

.seealso: TSSetConvergenceTest(), TSConvergedReason
@*/
PetscErrorCode  TSGetConvergedReason(TS ts,TSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = ts->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetSNESIterations"
/*@
   TSGetSNESIterations - Gets the total number of nonlinear iterations
   used by the time integrator.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  nits - number of nonlinear iterations

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.keywords: TS, get, number, nonlinear, iterations

.seealso:  TSGetKSPIterations()
@*/
PetscErrorCode TSGetSNESIterations(TS ts,PetscInt *nits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(nits,2);
  *nits = ts->snes_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetKSPIterations"
/*@
   TSGetKSPIterations - Gets the total number of linear iterations
   used by the time integrator.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.keywords: TS, get, number, linear, iterations

.seealso:  TSGetSNESIterations(), SNESGetKSPIterations()
@*/
PetscErrorCode TSGetKSPIterations(TS ts,PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = ts->ksp_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetStepRejections"
/*@
   TSGetStepRejections - Gets the total number of rejected steps.

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  rejects - number of steps rejected

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.keywords: TS, get, number

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetSNESFailures(), TSSetMaxSNESFailures(), TSSetErrorIfStepFails()
@*/
PetscErrorCode TSGetStepRejections(TS ts,PetscInt *rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(rejects,2);
  *rejects = ts->reject;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSNESFailures"
/*@
   TSGetSNESFailures - Gets the total number of failed SNES solves

   Not Collective

   Input Parameter:
.  ts - TS context

   Output Parameter:
.  fails - number of failed nonlinear solves

   Notes:
   This counter is reset to zero for each successive call to TSSolve().

   Level: intermediate

.keywords: TS, get, number

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSSetMaxSNESFailures()
@*/
PetscErrorCode TSGetSNESFailures(TS ts,PetscInt *fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(fails,2);
  *fails = ts->num_snes_failures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetMaxStepRejections"
/*@
   TSSetMaxStepRejections - Sets the maximum number of step rejections before a step fails

   Not Collective

   Input Parameter:
+  ts - TS context
-  rejects - maximum number of rejected steps, pass -1 for unlimited

   Notes:
   The counter is reset to zero for each step

   Options Database Key:
 .  -ts_max_reject - Maximum number of step rejections before a step fails

   Level: intermediate

.keywords: TS, set, maximum, number

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxSNESFailures(), TSGetStepRejections(), TSGetSNESFailures(), TSSetErrorIfStepFails(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetMaxStepRejections(TS ts,PetscInt rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_reject = rejects;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetMaxSNESFailures"
/*@
   TSSetMaxSNESFailures - Sets the maximum number of failed SNES solves

   Not Collective

   Input Parameter:
+  ts - TS context
-  fails - maximum number of failed nonlinear solves, pass -1 for unlimited

   Notes:
   The counter is reset to zero for each successive call to TSSolve().

   Options Database Key:
 .  -ts_max_snes_failures - Maximum number of nonlinear solve failures

   Level: intermediate

.keywords: TS, set, maximum, number

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSGetSNESFailures(), SNESGetConvergedReason(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetMaxSNESFailures(TS ts,PetscInt fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_snes_failures = fails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetErrorIfStepFails()"
/*@
   TSSetErrorIfStepFails - Error if no step succeeds

   Not Collective

   Input Parameter:
+  ts - TS context
-  err - PETSC_TRUE to error if no step succeeds, PETSC_FALSE to return without failure

   Options Database Key:
 .  -ts_error_if_step_fails - Error if no step succeeds

   Level: intermediate

.keywords: TS, set, error

.seealso:  TSGetSNESIterations(), TSGetKSPIterations(), TSSetMaxStepRejections(), TSGetStepRejections(), TSGetSNESFailures(), TSSetErrorIfStepFails(), TSGetConvergedReason()
@*/
PetscErrorCode TSSetErrorIfStepFails(TS ts,PetscBool err)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->errorifstepfailed = err;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSolutionBinary"
/*@C
   TSMonitorSolutionBinary - Monitors progress of the TS solvers by VecView() for the solution at each timestep. Normally the viewer is a binary file

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  x - current state
-  viewer - binary viewer

   Level: intermediate

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode  TSMonitorSolutionBinary(TS ts,PetscInt step,PetscReal ptime,Vec x,void *viewer)
{
  PetscErrorCode       ierr;
  PetscViewer          v = (PetscViewer)viewer;

  PetscFunctionBegin;
  ierr = VecView(x,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSolutionVTK"
/*@C
   TSMonitorSolutionVTK - Monitors progress of the TS solvers by VecView() for the solution at each timestep.

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  x - current state
-  filenametemplate - string containing a format specifier for the integer time step (e.g. %03D)

   Level: intermediate

   Notes:
   The VTK format does not allow writing multiple time steps in the same file, therefore a different file will be written for each time step.
   These are named according to the file name template.

   This function is normally passed as an argument to TSMonitorSet() along with TSMonitorSolutionVTKDestroy().

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode TSMonitorSolutionVTK(TS ts,PetscInt step,PetscReal ptime,Vec x,void *filenametemplate)
{
  PetscErrorCode ierr;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof filename,(const char*)filenametemplate,step);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(((PetscObject)ts)->comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorSolutionVTKDestroy"
/*@C
   TSMonitorSolutionVTKDestroy - Destroy context for monitoring

   Collective on TS

   Input Parameters:
.  filenametemplate - string containing a format specifier for the integer time step (e.g. %03D)

   Level: intermediate

   Note:
   This function is normally passed to TSMonitorSet() along with TSMonitorSolutionVTK().

.keywords: TS,  vector, monitor, view

.seealso: TSMonitorSet(), TSMonitorSolutionVTK()
@*/
PetscErrorCode TSMonitorSolutionVTKDestroy(void *filenametemplate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*(char**)filenametemplate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGetAdapt"
/*@
   TSGetAdapt - Get the adaptive controller context for the current method

   Collective on TS if controller has not been created yet

   Input Arguments:
.  ts - time stepping context

   Output Arguments:
.  adapt - adaptive controller

   Level: intermediate

.seealso: TSAdapt, TSAdaptSetType(), TSAdaptChoose()
@*/
PetscErrorCode TSGetAdapt(TS ts,TSAdapt *adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(adapt,2);
  if (!ts->adapt) {
    ierr = TSAdaptCreate(((PetscObject)ts)->comm,&ts->adapt);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,ts->adapt);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ts->adapt,(PetscObject)ts,1);CHKERRQ(ierr);
  }
  *adapt = ts->adapt;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetTolerances"
/*@
   TSSetTolerances - Set tolerances for local truncation error when using adaptive controller

   Logically Collective

   Input Arguments:
+  ts - time integration context
.  atol - scalar absolute tolerances, PETSC_DECIDE to leave current value
.  vatol - vector of absolute tolerances or PETSC_NULL, used in preference to atol if present
.  rtol - scalar relative tolerances, PETSC_DECIDE to leave current value
-  vrtol - vector of relative tolerances or PETSC_NULL, used in preference to atol if present

   Level: beginner

.seealso: TS, TSAdapt, TSVecNormWRMS(), TSGetTolerances()
@*/
PetscErrorCode TSSetTolerances(TS ts,PetscReal atol,Vec vatol,PetscReal rtol,Vec vrtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (atol != PETSC_DECIDE && atol != PETSC_DEFAULT) ts->atol = atol;
  if (vatol) {
    ierr = PetscObjectReference((PetscObject)vatol);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vatol);CHKERRQ(ierr);
    ts->vatol = vatol;
  }
  if (rtol != PETSC_DECIDE && rtol != PETSC_DEFAULT) ts->rtol = rtol;
  if (vrtol) {
    ierr = PetscObjectReference((PetscObject)vrtol);CHKERRQ(ierr);
    ierr = VecDestroy(&ts->vrtol);CHKERRQ(ierr);
    ts->vrtol = vrtol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetTolerances"
/*@
   TSGetTolerances - Get tolerances for local truncation error when using adaptive controller

   Logically Collective

   Input Arguments:
.  ts - time integration context

   Output Arguments:
+  atol - scalar absolute tolerances, PETSC_NULL to ignore
.  vatol - vector of absolute tolerances, PETSC_NULL to ignore
.  rtol - scalar relative tolerances, PETSC_NULL to ignore
-  vrtol - vector of relative tolerances, PETSC_NULL to ignore

   Level: beginner

.seealso: TS, TSAdapt, TSVecNormWRMS(), TSSetTolerances()
@*/
PetscErrorCode TSGetTolerances(TS ts,PetscReal *atol,Vec *vatol,PetscReal *rtol,Vec *vrtol)
{

  PetscFunctionBegin;
  if (atol)  *atol  = ts->atol;
  if (vatol) *vatol = ts->vatol;
  if (rtol)  *rtol  = ts->rtol;
  if (vrtol) *vrtol = ts->vrtol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSErrorNormWRMS"
/*@
   TSErrorNormWRMS - compute a weighted norm of the difference between a vector and the current state

   Collective on TS

   Input Arguments:
+  ts - time stepping context
-  Y - state vector to be compared to ts->vec_sol

   Output Arguments:
.  norm - weighted norm, a value of 1.0 is considered small

   Level: developer

.seealso: TSSetTolerances()
@*/
PetscErrorCode TSErrorNormWRMS(TS ts,Vec Y,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt i,n,N;
  const PetscScalar *x,*y;
  Vec X;
  PetscReal sum,gsum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(Y,VEC_CLASSID,2);
  PetscValidPointer(norm,3);
  X = ts->vec_sol;
  PetscCheckSameTypeAndComm(X,1,Y,2);
  if (X == Y) SETERRQ(((PetscObject)X)->comm,PETSC_ERR_ARG_IDN,"Y cannot be the TS solution vector");

  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  sum = 0.;
  if (ts->vatol && ts->vrtol) {
    const PetscScalar *atol,*rtol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscReal tol = PetscRealPart(atol[i]) + PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(x[i]),PetscAbsScalar(y[i]));
      sum += PetscSqr(PetscAbsScalar(y[i] - x[i]) / tol);
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else if (ts->vatol) {       /* vector atol, scalar rtol */
    const PetscScalar *atol;
    ierr = VecGetArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscReal tol = PetscRealPart(atol[i]) + ts->rtol * PetscMax(PetscAbsScalar(x[i]),PetscAbsScalar(y[i]));
      sum += PetscSqr(PetscAbsScalar(y[i] - x[i]) / tol);
    }
    ierr = VecRestoreArrayRead(ts->vatol,&atol);CHKERRQ(ierr);
  } else if (ts->vrtol) {       /* scalar atol, vector rtol */
    const PetscScalar *rtol;
    ierr = VecGetArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscReal tol = ts->atol + PetscRealPart(rtol[i]) * PetscMax(PetscAbsScalar(x[i]),PetscAbsScalar(y[i]));
      sum += PetscSqr(PetscAbsScalar(y[i] - x[i]) / tol);
    }
    ierr = VecRestoreArrayRead(ts->vrtol,&rtol);CHKERRQ(ierr);
  } else {                      /* scalar atol, scalar rtol */
    for (i=0; i<n; i++) {
      PetscReal tol = ts->atol + ts->rtol * PetscMax(PetscAbsScalar(x[i]),PetscAbsScalar(y[i]));
      sum += PetscSqr(PetscAbsScalar(y[i] - x[i]) / tol);
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&sum,&gsum,1,MPIU_REAL,MPIU_SUM,((PetscObject)ts)->comm);CHKERRQ(ierr);
  *norm = PetscSqrtReal(gsum / N);
  if (PetscIsInfOrNanScalar(*norm)) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetCFLTimeLocal"
/*@
   TSSetCFLTimeLocal - Set the local CFL constraint relative to forward Euler

   Logically Collective on TS

   Input Arguments:
+  ts - time stepping context
-  cfltime - maximum stable time step if using forward Euler (value can be different on each process)

   Note:
   After calling this function, the global CFL time can be obtained by calling TSGetCFLTime()

   Level: intermediate

.seealso: TSGetCFLTime(), TSADAPTCFL
@*/
PetscErrorCode TSSetCFLTimeLocal(TS ts,PetscReal cfltime)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->cfltime_local = cfltime;
  ts->cfltime = -1.;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetCFLTime"
/*@
   TSGetCFLTime - Get the maximum stable time step according to CFL criteria applied to forward Euler

   Collective on TS

   Input Arguments:
.  ts - time stepping context

   Output Arguments:
.  cfltime - maximum stable time step for forward Euler

   Level: advanced

.seealso: TSSetCFLTimeLocal()
@*/
PetscErrorCode TSGetCFLTime(TS ts,PetscReal *cfltime)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->cfltime < 0) {
    ierr = MPI_Allreduce(&ts->cfltime_local,&ts->cfltime,1,MPIU_REAL,MPIU_MIN,((PetscObject)ts)->comm);CHKERRQ(ierr);
  }
  *cfltime = ts->cfltime;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSVISetVariableBounds"
/*@
   TSVISetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu

   Input Parameters:
.  ts   - the TS context.
.  xl   - lower bound.
.  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to 
   SNES_VI_NINF and SNES_VI_INF respectively during SNESSetUp().

   Level: advanced

@*/
PetscErrorCode TSVISetVariableBounds(TS ts, Vec xl, Vec xu)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <mex.h>

typedef struct {char *funcname; mxArray *ctx;} TSMatlabContext;

#undef __FUNCT__
#define __FUNCT__ "TSComputeFunction_Matlab"
/*
   TSComputeFunction_Matlab - Calls the function that has been set with
                         TSSetFunctionMatlab().

   Collective on TS

   Input Parameters:
+  snes - the TS context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by TSSetFunction()

   Notes:
   TSComputeFunction() is typically used within nonlinear solvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: TS, nonlinear, compute, function

.seealso: TSSetFunction(), TSGetFunction()
*/
PetscErrorCode  TSComputeFunction_Matlab(TS snes,PetscReal time,Vec x,Vec xdot,Vec y, void *ctx)
{
  PetscErrorCode   ierr;
  TSMatlabContext *sctx = (TSMatlabContext *)ctx;
  int              nlhs = 1,nrhs = 7;
  mxArray          *plhs[1],*prhs[7];
  long long int    lx = 0,lxdot = 0,ly = 0,ls = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(xdot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(y,VEC_CLASSID,5);
  PetscCheckSameComm(snes,1,x,3);
  PetscCheckSameComm(snes,1,y,5);

  ierr = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lxdot,&xdot,sizeof(xdot));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&ly,&y,sizeof(x));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar(time);
  prhs[2] =  mxCreateDoubleScalar((double)lx);
  prhs[3] =  mxCreateDoubleScalar((double)lxdot);
  prhs[4] =  mxCreateDoubleScalar((double)ly);
  prhs[5] =  mxCreateString(sctx->funcname);
  prhs[6] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscTSComputeFunctionInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(prhs[5]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetFunctionMatlab"
/*
   TSSetFunctionMatlab - Sets the function evaluation routine and function
   vector for use by the TS routines in solving ODEs
   equations from MATLAB. Here the function is a string containing the name of a MATLAB function

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context
-  func - function evaluation routine

   Calling sequence of func:
$    func (TS ts,PetscReal time,Vec x,Vec xdot,Vec f,void *ctx);

   Level: beginner

.keywords: TS, nonlinear, set, function

.seealso: TSGetFunction(), TSComputeFunction(), TSSetJacobian(), TSSetFunction()
*/
PetscErrorCode  TSSetFunctionMatlab(TS ts,const char *func,mxArray *ctx)
{
  PetscErrorCode  ierr;
  TSMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(TSMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /*
     This should work, but it doesn't
  sctx->ctx = ctx;
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = TSSetIFunction(ts,PETSC_NULL,TSComputeFunction_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeJacobian_Matlab"
/*
   TSComputeJacobian_Matlab - Calls the function that has been set with
                         TSSetJacobianMatlab().

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  x - input vector
.  A, B - the matrices
-  ctx - user context

   Output Parameter:
.  flag - structure of the matrix

   Level: developer

.keywords: TS, nonlinear, compute, function

.seealso: TSSetFunction(), TSGetFunction()
@*/
PetscErrorCode  TSComputeJacobian_Matlab(TS ts,PetscReal time,Vec x,Vec xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *flag, void *ctx)
{
  PetscErrorCode  ierr;
  TSMatlabContext *sctx = (TSMatlabContext *)ctx;
  int             nlhs = 2,nrhs = 9;
  mxArray         *plhs[2],*prhs[9];
  long long int   lx = 0,lxdot = 0,lA = 0,ls = 0, lB = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);

  /* call Matlab function in ctx with arguments x and y */

  ierr = PetscMemcpy(&ls,&ts,sizeof(ts));CHKERRQ(ierr);
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr);
  ierr = PetscMemcpy(&lxdot,&xdot,sizeof(x));CHKERRQ(ierr);
  ierr = PetscMemcpy(&lA,A,sizeof(x));CHKERRQ(ierr);
  ierr = PetscMemcpy(&lB,B,sizeof(x));CHKERRQ(ierr);
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)time);
  prhs[2] =  mxCreateDoubleScalar((double)lx);
  prhs[3] =  mxCreateDoubleScalar((double)lxdot);
  prhs[4] =  mxCreateDoubleScalar((double)shift);
  prhs[5] =  mxCreateDoubleScalar((double)lA);
  prhs[6] =  mxCreateDoubleScalar((double)lB);
  prhs[7] =  mxCreateString(sctx->funcname);
  prhs[8] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscTSComputeJacobianInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  *flag   =  (MatStructure) mxGetScalar(plhs[1]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(prhs[5]);
  mxDestroyArray(prhs[6]);
  mxDestroyArray(prhs[7]);
  mxDestroyArray(plhs[0]);
  mxDestroyArray(plhs[1]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetJacobianMatlab"
/*
   TSSetJacobianMatlab - Sets the Jacobian function evaluation routine and two empty Jacobian matrices
   vector for use by the TS routines in solving ODEs from MATLAB. Here the function is a string containing the name of a MATLAB function

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context
.  A,B - Jacobian matrices
.  func - function evaluation routine
-  ctx - user context

   Calling sequence of func:
$    flag = func (TS ts,PetscReal time,Vec x,Vec xdot,Mat A,Mat B,void *ctx);


   Level: developer

.keywords: TS, nonlinear, set, function

.seealso: TSGetFunction(), TSComputeFunction(), TSSetJacobian(), TSSetFunction()
*/
PetscErrorCode  TSSetJacobianMatlab(TS ts,Mat A,Mat B,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  TSMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(TSMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /*
     This should work, but it doesn't
  sctx->ctx = ctx;
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = TSSetIJacobian(ts,A,B,TSComputeJacobian_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitor_Matlab"
/*
   TSMonitor_Matlab - Calls the function that has been set with TSMonitorSetMatlab().

   Collective on TS

.seealso: TSSetFunction(), TSGetFunction()
@*/
PetscErrorCode  TSMonitor_Matlab(TS ts,PetscInt it, PetscReal time,Vec x, void *ctx)
{
  PetscErrorCode  ierr;
  TSMatlabContext *sctx = (TSMatlabContext *)ctx;
  int             nlhs = 1,nrhs = 6;
  mxArray         *plhs[1],*prhs[6];
  long long int   lx = 0,ls = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,4);

  ierr = PetscMemcpy(&ls,&ts,sizeof(ts));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)it);
  prhs[2] =  mxCreateDoubleScalar((double)time);
  prhs[3] =  mxCreateDoubleScalar((double)lx);
  prhs[4] =  mxCreateString(sctx->funcname);
  prhs[5] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscTSMonitorInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSMonitorSetMatlab"
/*
   TSMonitorSetMatlab - Sets the monitor function from Matlab

   Level: developer

.keywords: TS, nonlinear, set, function

.seealso: TSGetFunction(), TSComputeFunction(), TSSetJacobian(), TSSetFunction()
*/
PetscErrorCode  TSMonitorSetMatlab(TS ts,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  TSMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(TSMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /*
     This should work, but it doesn't
  sctx->ctx = ctx;
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = TSMonitorSet(ts,TSMonitor_Matlab,sctx,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
