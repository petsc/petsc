#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscdm.h>
#include <petscds.h>
#include <petscdmswarm.h>
#include <petscdraw.h>

/*@C
   TSMonitor - Runs all user-provided monitor routines set using TSMonitorSet()

   Collective on TS

   Input Parameters:
+  ts - time stepping context obtained from TSCreate()
.  step - step number that has just completed
.  ptime - model time of the state
-  u - state at the current model time

   Notes:
   TSMonitor() is typically used automatically within the time stepping implementations.
   Users would almost never call this routine directly.

   A step of -1 indicates that the monitor is being called on a solution obtained by interpolating from computed solutions

   Level: developer

@*/
PetscErrorCode TSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec u)
{
  DM             dm;
  PetscInt       i,n = ts->numbermonitors;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,4);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm,step,ptime);CHKERRQ(ierr);

  ierr = VecLockReadPush(u);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = (*ts->monitor[i])(ts,step,ptime,u,ts->monitorcontext[i]);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on TS

   Input Parameters:
+  ts - TS object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the TS or PetscViewer objects

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  TSMonitorSetFromOptions(TS ts,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,PetscViewerAndFormat*),PetscErrorCode (*monitorsetup)(TS,PetscViewerAndFormat*))
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ts),((PetscObject) ts)->options,((PetscObject)ts)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewerAndFormat *vf;
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    if (monitorsetup) {
      ierr = (*monitorsetup)(ts,vf);CHKERRQ(ierr);
    }
    ierr = TSMonitorSet(ts,(PetscErrorCode (*)(TS,PetscInt,PetscReal,Vec,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSet - Sets an ADDITIONAL function that is to be used at every
   timestep to display the iteration's  progress.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  monitor - monitoring routine
.  mctx - [optional] user-defined context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling sequence of monitor:
$    PetscErrorCode monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)

+    ts - the TS context
.    steps - iteration number (after the final time step the monitor routine may be called with a step of -1, this indicates the solution has been interpolated to this time)
.    time - current time
.    u - current iterate
-    mctx - [optional] monitoring context

   Notes:
   This routine adds an additional monitor to the list of monitors that
   already has been loaded.

   Fortran Notes:
    Only a single monitor function can be set for each TS object

   Level: intermediate

.seealso: TSMonitorDefault(), TSMonitorCancel()
@*/
PetscErrorCode  TSMonitorSet(TS ts,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *mctx,PetscErrorCode (*mdestroy)(void**))
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numbermonitors;i++) {
    ierr = PetscMonitorCompare((PetscErrorCode (*)(void))monitor,mctx,mdestroy,(PetscErrorCode (*)(void))ts->monitor[i],ts->monitorcontext[i],ts->monitordestroy[i],&identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  if (ts->numbermonitors >= MAXTSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  ts->monitor[ts->numbermonitors]          = monitor;
  ts->monitordestroy[ts->numbermonitors]   = mdestroy;
  ts->monitorcontext[ts->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorCancel - Clears all the monitors that have been set on a time-step object.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context obtained from TSCreate()

   Notes:
   There is no way to remove a single, specific monitor.

   Level: intermediate

.seealso: TSMonitorDefault(), TSMonitorSet()
@*/
PetscErrorCode  TSMonitorCancel(TS ts)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitordestroy[i]) {
      ierr = (*ts->monitordestroy[i])(&ts->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ts->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDefault - The Default monitor, prints the timestep and time for each step

   Level: intermediate

.seealso:  TSMonitorSet()
@*/
PetscErrorCode TSMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer =  vf->viewer;
  PetscBool      iascii,ibinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,5);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&ibinary);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
    if (step == -1) { /* this indicates it is an interpolated solution */
      ierr = PetscViewerASCIIPrintf(viewer,"Interpolated solution at time %g between steps %D and %D\n",(double)ptime,ts->steps-1,ts->steps);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"%D TS dt %g time %g%s",step,(double)ts->time_step,(double)ptime,ts->steprollback ? " (r)\n" : "\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  } else if (ibinary) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
    if (!rank) {
      PetscBool skipHeader;
      PetscInt  classid = REAL_FILE_CLASSID;

      ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
      if (!skipHeader) {
         ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
       }
      ierr = PetscRealView(1,&ptime,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscRealView(0,&ptime,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorExtreme - Prints the extreme values of the solution at each timestep

   Level: intermediate

.seealso:  TSMonitorSet()
@*/
PetscErrorCode TSMonitorExtreme(TS ts,PetscInt step,PetscReal ptime,Vec v,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer =  vf->viewer;
  PetscBool      iascii;
  PetscReal      max,min;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,5);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (iascii) {
    ierr = VecMax(v,NULL,&max);CHKERRQ(ierr);
    ierr = VecMin(v,NULL,&min);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D TS dt %g time %g%s max %g min %g\n",step,(double)ts->time_step,(double)ptime,ts->steprollback ? " (r)" : "",(double)max,(double)min);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ts)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGCtxCreate - Creates a TSMonitorLGCtx context for use with
   TS to monitor the solution process graphically in various ways

   Collective on TS

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of the window
.  m, n - the screen width and height in pixels
-  howoften - if positive then determines the frequency of the plotting, if -1 then only at the final time

   Output Parameter:
.  ctx - the context

   Options Database Key:
+  -ts_monitor_lg_timestep - automatically sets line graph monitor
+  -ts_monitor_lg_timestep_log - automatically sets line graph monitor
.  -ts_monitor_lg_solution - monitor the solution (or certain values of the solution by calling TSMonitorLGSetDisplayVariables() or TSMonitorLGCtxSetDisplayVariables())
.  -ts_monitor_lg_error -  monitor the error
.  -ts_monitor_lg_ksp_iterations - monitor the number of KSP iterations needed for each timestep
.  -ts_monitor_lg_snes_iterations - monitor the number of SNES iterations needed for each timestep
-  -lg_use_markers <true,false> - mark the data points (at each time step) on the plot; default is true

   Notes:
   Use TSMonitorLGCtxDestroy() to destroy.

   One can provide a function that transforms the solution before plotting it with TSMonitorLGCtxSetTransform() or TSMonitorLGSetTransform()

   Many of the functions that control the monitoring have two forms: TSMonitorLGSet/GetXXXX() and TSMonitorLGCtxSet/GetXXXX() the first take a TS object as the
   first argument (if that TS object does not have a TSMonitorLGCtx associated with it the function call is ignored) and the second takes a TSMonitorLGCtx object
   as the first argument.

   One can control the names displayed for each solution or error variable with TSMonitorLGCtxSetVariableNames() or TSMonitorLGSetVariableNames()

   Level: intermediate

.seealso: TSMonitorLGTimeStep(), TSMonitorSet(), TSMonitorLGSolution(), TSMonitorLGError(), TSMonitorDefault(), VecView(),
           TSMonitorLGCtxCreate(), TSMonitorLGCtxSetVariableNames(), TSMonitorLGCtxGetVariableNames(),
           TSMonitorLGSetVariableNames(), TSMonitorLGGetVariableNames(), TSMonitorLGSetDisplayVariables(), TSMonitorLGCtxSetDisplayVariables(),
           TSMonitorLGCtxSetTransform(), TSMonitorLGSetTransform(), TSMonitorLGError(), TSMonitorLGSNESIterations(), TSMonitorLGKSPIterations(),
           TSMonitorEnvelopeCtxCreate(), TSMonitorEnvelopeGetBounds(), TSMonitorEnvelopeCtxDestroy(), TSMonitorEnvelop()

@*/
PetscErrorCode  TSMonitorLGCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorLGCtx *ctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,1,&(*ctx)->lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetFromOptions((*ctx)->lg);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorLGTimeStep(TS ts,PetscInt step,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorLGCtx ctx = (TSMonitorLGCtx) monctx;
  PetscReal      x   = ptime,y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates an interpolated solution */
  if (!step) {
    PetscDrawAxis axis;
    const char *ylabel = ctx->semilogy ? "Log Time Step" : "Time Step";
    ierr = PetscDrawLGGetAxis(ctx->lg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Timestep as function of time","Time",ylabel);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(ctx->lg);CHKERRQ(ierr);
  }
  ierr = TSGetTimeStep(ts,&y);CHKERRQ(ierr);
  if (ctx->semilogy) y = PetscLog10Real(y);
  ierr = PetscDrawLGAddPoint(ctx->lg,&x,&y);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    ierr = PetscDrawLGDraw(ctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(ctx->lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGCtxDestroy - Destroys a line graph context that was created
   with TSMonitorLGCtxCreate().

   Collective on TSMonitorLGCtx

   Input Parameter:
.  ctx - the monitor context

   Level: intermediate

.seealso: TSMonitorLGCtxCreate(),  TSMonitorSet(), TSMonitorLGTimeStep();
@*/
PetscErrorCode  TSMonitorLGCtxDestroy(TSMonitorLGCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((*ctx)->transformdestroy) {
    ierr = ((*ctx)->transformdestroy)((*ctx)->transformctx);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGDestroy(&(*ctx)->lg);CHKERRQ(ierr);
  ierr = PetscStrArrayDestroy(&(*ctx)->names);CHKERRQ(ierr);
  ierr = PetscStrArrayDestroy(&(*ctx)->displaynames);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->displayvariables);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->displayvalues);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Creates a TS Monitor SPCtx for use with DMSwarm particle visualizations */
PetscErrorCode TSMonitorSPCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,PetscInt retain,PetscBool phase,TSMonitorSPCtx *ctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(draw,1,&(*ctx)->sp);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  (*ctx)->howoften = howoften;
  (*ctx)->retain   = retain;
  (*ctx)->phase    = phase;
  PetscFunctionReturn(0);
}

/*
  Destroys a TSMonitorSPCtx that was created with TSMonitorSPCtxCreate
*/
PetscErrorCode TSMonitorSPCtxDestroy(TSMonitorSPCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscDrawSPDestroy(&(*ctx)->sp);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/*@C
   TSMonitorDrawSolution - Monitors progress of the TS solvers by calling
   VecView() for the solution at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or NULL

   Options Database:
.   -ts_monitor_draw_solution_initial - show initial solution as well as current solution

   Notes:
    the initial solution and current solution are not display with a common axis scaling so generally the option -ts_monitor_draw_solution_initial
       will look bad

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode  TSMonitorDrawSolution(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dummy)
{
  PetscErrorCode   ierr;
  TSMonitorDrawCtx ictx = (TSMonitorDrawCtx)dummy;
  PetscDraw        draw;

  PetscFunctionBegin;
  if (!step && ictx->showinitial) {
    if (!ictx->initialsolution) {
      ierr = VecDuplicate(u,&ictx->initialsolution);CHKERRQ(ierr);
    }
    ierr = VecCopy(u,ictx->initialsolution);CHKERRQ(ierr);
  }
  if (!(((ictx->howoften > 0) && (!(step % ictx->howoften))) || ((ictx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);

  if (ictx->showinitial) {
    PetscReal pause;
    ierr = PetscViewerDrawGetPause(ictx->viewer,&pause);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetPause(ictx->viewer,0.0);CHKERRQ(ierr);
    ierr = VecView(ictx->initialsolution,ictx->viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetPause(ictx->viewer,pause);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetHold(ictx->viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = VecView(u,ictx->viewer);CHKERRQ(ierr);
  if (ictx->showtimestepandtime) {
    PetscReal xl,yl,xr,yr,h;
    char      time[32];

    ierr = PetscViewerDrawGetDraw(ictx->viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscSNPrintf(time,32,"Timestep %d Time %g",(int)step,(double)ptime);CHKERRQ(ierr);
    ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
    h    = yl + .95*(yr - yl);
    ierr = PetscDrawStringCentered(draw,.5*(xl+xr),h,PETSC_DRAW_BLACK,time);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  }

  if (ictx->showinitial) {
    ierr = PetscViewerDrawSetHold(ictx->viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDrawSolutionPhase - Monitors progress of the TS solvers by plotting the solution as a phase diagram

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or NULL

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode  TSMonitorDrawSolutionPhase(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dummy)
{
  PetscErrorCode    ierr;
  TSMonitorDrawCtx  ictx = (TSMonitorDrawCtx)dummy;
  PetscDraw         draw;
  PetscDrawAxis     axis;
  PetscInt          n;
  PetscMPIInt       size;
  PetscReal         U0,U1,xl,yl,xr,yr,h;
  char              time[32];
  const PetscScalar *U;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)ts),&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Only allowed for sequential runs");
  ierr = VecGetSize(u,&n);CHKERRQ(ierr);
  if (n != 2) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Only for ODEs with two unknowns");

  ierr = PetscViewerDrawGetDraw(ictx->viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawAxis(ictx->viewer,0,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisGetLimits(axis,&xl,&xr,&yl,&yr);CHKERRQ(ierr);
  if (!step) {
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawAxisDraw(axis);CHKERRQ(ierr);
  }

  ierr = VecGetArrayRead(u,&U);CHKERRQ(ierr);
  U0 = PetscRealPart(U[0]);
  U1 = PetscRealPart(U[1]);
  ierr = VecRestoreArrayRead(u,&U);CHKERRQ(ierr);
  if ((U0 < xl) || (U1 < yl) || (U0 > xr) || (U1 > yr)) PetscFunctionReturn(0);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  ierr = PetscDrawPoint(draw,U0,U1,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  if (ictx->showtimestepandtime) {
    ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
    ierr = PetscSNPrintf(time,32,"Timestep %d Time %g",(int)step,(double)ptime);CHKERRQ(ierr);
    h    = yl + .95*(yr - yl);
    ierr = PetscDrawStringCentered(draw,.5*(xl+xr),h,PETSC_DRAW_BLACK,time);CHKERRQ(ierr);
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDrawCtxDestroy - Destroys the monitor context for TSMonitorDrawSolution()

   Collective on TS

   Input Parameters:
.    ctx - the monitor context

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorDrawSolution(), TSMonitorDrawError()
@*/
PetscErrorCode  TSMonitorDrawCtxDestroy(TSMonitorDrawCtx *ictx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&(*ictx)->viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ictx)->initialsolution);CHKERRQ(ierr);
  ierr = PetscFree(*ictx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDrawCtxCreate - Creates the monitor context for TSMonitorDrawCtx

   Collective on TS

   Input Parameter:
.    ts - time-step context

   Output Patameter:
.    ctx - the monitor context

   Options Database:
.   -ts_monitor_draw_solution_initial - show initial solution as well as current solution

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorDrawCtx()
@*/
PetscErrorCode  TSMonitorDrawCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorDrawCtx *ctx)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(comm,host,label,x,y,m,n,&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions((*ctx)->viewer);CHKERRQ(ierr);

  (*ctx)->howoften    = howoften;
  (*ctx)->showinitial = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ts_monitor_draw_solution_initial",&(*ctx)->showinitial,NULL);CHKERRQ(ierr);

  (*ctx)->showtimestepandtime = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ts_monitor_draw_solution_show_time",&(*ctx)->showtimestepandtime,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDrawSolutionFunction - Monitors progress of the TS solvers by calling
   VecView() for the solution provided by TSSetSolutionFunction() at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or NULL

   Options Database:
.  -ts_monitor_draw_solution_function - Monitor error graphically, requires user to have provided TSSetSolutionFunction()

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSSetSolutionFunction()
@*/
PetscErrorCode  TSMonitorDrawSolutionFunction(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dummy)
{
  PetscErrorCode   ierr;
  TSMonitorDrawCtx ctx    = (TSMonitorDrawCtx)dummy;
  PetscViewer      viewer = ctx->viewer;
  Vec              work;

  PetscFunctionBegin;
  if (!(((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);
  ierr = VecDuplicate(u,&work);CHKERRQ(ierr);
  ierr = TSComputeSolutionFunction(ts,ptime,work);CHKERRQ(ierr);
  ierr = VecView(work,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorDrawError - Monitors progress of the TS solvers by calling
   VecView() for the error at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
-  dummy - either a viewer or NULL

   Options Database:
.  -ts_monitor_draw_error - Monitor error graphically, requires user to have provided TSSetSolutionFunction()

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSSetSolutionFunction()
@*/
PetscErrorCode  TSMonitorDrawError(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dummy)
{
  PetscErrorCode   ierr;
  TSMonitorDrawCtx ctx    = (TSMonitorDrawCtx)dummy;
  PetscViewer      viewer = ctx->viewer;
  Vec              work;

  PetscFunctionBegin;
  if (!(((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason))) PetscFunctionReturn(0);
  ierr = VecDuplicate(u,&work);CHKERRQ(ierr);
  ierr = TSComputeSolutionFunction(ts,ptime,work);CHKERRQ(ierr);
  ierr = VecAXPY(work,-1.0,u);CHKERRQ(ierr);
  ierr = VecView(work,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSolution - Monitors progress of the TS solvers by VecView() for the solution at each timestep. Normally the viewer is a binary file or a PetscDraw object

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current state
-  vf - viewer and its format

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode  TSMonitorSolution(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerPushFormat(vf->viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(u,vf->viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(vf->viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSolutionVTK - Monitors progress of the TS solvers by VecView() for the solution at each timestep.

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current state
-  filenametemplate - string containing a format specifier for the integer time step (e.g. %03D)

   Level: intermediate

   Notes:
   The VTK format does not allow writing multiple time steps in the same file, therefore a different file will be written for each time step.
   These are named according to the file name template.

   This function is normally passed as an argument to TSMonitorSet() along with TSMonitorSolutionVTKDestroy().

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView()
@*/
PetscErrorCode TSMonitorSolutionVTK(TS ts,PetscInt step,PetscReal ptime,Vec u,void *filenametemplate)
{
  PetscErrorCode ierr;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  ierr = PetscSNPrintf(filename,sizeof(filename),(const char*)filenametemplate,step);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSolutionVTKDestroy - Destroy context for monitoring

   Collective on TS

   Input Parameters:
.  filenametemplate - string containing a format specifier for the integer time step (e.g. %03D)

   Level: intermediate

   Note:
   This function is normally passed to TSMonitorSet() along with TSMonitorSolutionVTK().

.seealso: TSMonitorSet(), TSMonitorSolutionVTK()
@*/
PetscErrorCode TSMonitorSolutionVTKDestroy(void *filenametemplate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*(char**)filenametemplate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGSolution - Monitors progress of the TS solvers by plotting each component of the solution vector
       in a time based line graph

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - the TSMonitorLGCtx object that contains all the options for the monitoring, this is created with TSMonitorLGCtxCreate()

   Options Database:
.   -ts_monitor_lg_solution_variables

   Level: intermediate

   Notes:
    Each process in a parallel run displays its component solutions in a separate window

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGCtxCreate(), TSMonitorLGCtxSetVariableNames(), TSMonitorLGCtxGetVariableNames(),
           TSMonitorLGSetVariableNames(), TSMonitorLGGetVariableNames(), TSMonitorLGSetDisplayVariables(), TSMonitorLGCtxSetDisplayVariables(),
           TSMonitorLGCtxSetTransform(), TSMonitorLGSetTransform(), TSMonitorLGError(), TSMonitorLGSNESIterations(), TSMonitorLGKSPIterations(),
           TSMonitorEnvelopeCtxCreate(), TSMonitorEnvelopeGetBounds(), TSMonitorEnvelopeCtxDestroy(), TSMonitorEnvelop()
@*/
PetscErrorCode  TSMonitorLGSolution(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dctx)
{
  PetscErrorCode    ierr;
  TSMonitorLGCtx    ctx = (TSMonitorLGCtx)dctx;
  const PetscScalar *yy;
  Vec               v;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!step) {
    PetscDrawAxis axis;
    PetscInt      dim;
    ierr = PetscDrawLGGetAxis(ctx->lg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Solution as function of time","Time","Solution");CHKERRQ(ierr);
    if (!ctx->names) {
      PetscBool flg;
      /* user provides names of variables to plot but no names has been set so assume names are integer values */
      ierr = PetscOptionsHasName(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-ts_monitor_lg_solution_variables",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscInt i,n;
        char     **names;
        ierr = VecGetSize(u,&n);CHKERRQ(ierr);
        ierr = PetscMalloc1(n+1,&names);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscMalloc1(5,&names[i]);CHKERRQ(ierr);
          ierr = PetscSNPrintf(names[i],5,"%D",i);CHKERRQ(ierr);
        }
        names[n] = NULL;
        ctx->names = names;
      }
    }
    if (ctx->names && !ctx->displaynames) {
      char      **displaynames;
      PetscBool flg;
      ierr = VecGetLocalSize(u,&dim);CHKERRQ(ierr);
      ierr = PetscCalloc1(dim+1,&displaynames);CHKERRQ(ierr);
      ierr = PetscOptionsGetStringArray(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-ts_monitor_lg_solution_variables",displaynames,&dim,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = TSMonitorLGCtxSetDisplayVariables(ctx,(const char *const *)displaynames);CHKERRQ(ierr);
      }
      ierr = PetscStrArrayDestroy(&displaynames);CHKERRQ(ierr);
    }
    if (ctx->displaynames) {
      ierr = PetscDrawLGSetDimension(ctx->lg,ctx->ndisplayvariables);CHKERRQ(ierr);
      ierr = PetscDrawLGSetLegend(ctx->lg,(const char *const *)ctx->displaynames);CHKERRQ(ierr);
    } else if (ctx->names) {
      ierr = VecGetLocalSize(u,&dim);CHKERRQ(ierr);
      ierr = PetscDrawLGSetDimension(ctx->lg,dim);CHKERRQ(ierr);
      ierr = PetscDrawLGSetLegend(ctx->lg,(const char *const *)ctx->names);CHKERRQ(ierr);
    } else {
      ierr = VecGetLocalSize(u,&dim);CHKERRQ(ierr);
      ierr = PetscDrawLGSetDimension(ctx->lg,dim);CHKERRQ(ierr);
    }
    ierr = PetscDrawLGReset(ctx->lg);CHKERRQ(ierr);
  }

  if (!ctx->transform) v = u;
  else {ierr = (*ctx->transform)(ctx->transformctx,u,&v);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(v,&yy);CHKERRQ(ierr);
  if (ctx->displaynames) {
    PetscInt i;
    for (i=0; i<ctx->ndisplayvariables; i++)
      ctx->displayvalues[i] = PetscRealPart(yy[ctx->displayvariables[i]]);
    ierr = PetscDrawLGAddCommonPoint(ctx->lg,ptime,ctx->displayvalues);CHKERRQ(ierr);
  } else {
#if defined(PETSC_USE_COMPLEX)
    PetscInt  i,n;
    PetscReal *yreal;
    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&yreal);CHKERRQ(ierr);
    for (i=0; i<n; i++) yreal[i] = PetscRealPart(yy[i]);
    ierr = PetscDrawLGAddCommonPoint(ctx->lg,ptime,yreal);CHKERRQ(ierr);
    ierr = PetscFree(yreal);CHKERRQ(ierr);
#else
    ierr = PetscDrawLGAddCommonPoint(ctx->lg,ptime,yy);CHKERRQ(ierr);
#endif
  }
  ierr = VecRestoreArrayRead(v,&yy);CHKERRQ(ierr);
  if (ctx->transform) {ierr = VecDestroy(&v);CHKERRQ(ierr);}

  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    ierr = PetscDrawLGDraw(ctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(ctx->lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGSetVariableNames - Sets the name of each component in the solution vector so that it may be displayed in the plot

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  names - the names of the components, final string must be NULL

   Level: intermediate

   Notes:
    If the TS object does not have a TSMonitorLGCtx associated with it then this function is ignored

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetDisplayVariables(), TSMonitorLGCtxSetVariableNames()
@*/
PetscErrorCode  TSMonitorLGSetVariableNames(TS ts,const char * const *names)
{
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitor[i] == TSMonitorLGSolution) {
      ierr = TSMonitorLGCtxSetVariableNames((TSMonitorLGCtx)ts->monitorcontext[i],names);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGCtxSetVariableNames - Sets the name of each component in the solution vector so that it may be displayed in the plot

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  names - the names of the components, final string must be NULL

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetDisplayVariables(), TSMonitorLGSetVariableNames()
@*/
PetscErrorCode  TSMonitorLGCtxSetVariableNames(TSMonitorLGCtx ctx,const char * const *names)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscStrArrayDestroy(&ctx->names);CHKERRQ(ierr);
  ierr = PetscStrArrayallocpy(names,&ctx->names);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGGetVariableNames - Gets the name of each component in the solution vector so that it may be displayed in the plot

   Collective on TS

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  names - the names of the components, final string must be NULL

   Level: intermediate

   Notes:
    If the TS object does not have a TSMonitorLGCtx associated with it then this function is ignored

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetDisplayVariables()
@*/
PetscErrorCode  TSMonitorLGGetVariableNames(TS ts,const char *const **names)
{
  PetscInt       i;

  PetscFunctionBegin;
  *names = NULL;
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitor[i] == TSMonitorLGSolution) {
      TSMonitorLGCtx  ctx = (TSMonitorLGCtx) ts->monitorcontext[i];
      *names = (const char *const *)ctx->names;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGCtxSetDisplayVariables - Sets the variables that are to be display in the monitor

   Collective on TS

   Input Parameters:
+  ctx - the TSMonitorLG context
-  displaynames - the names of the components, final string must be NULL

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetVariableNames()
@*/
PetscErrorCode  TSMonitorLGCtxSetDisplayVariables(TSMonitorLGCtx ctx,const char * const *displaynames)
{
  PetscInt          j = 0,k;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!ctx->names) PetscFunctionReturn(0);
  ierr = PetscStrArrayDestroy(&ctx->displaynames);CHKERRQ(ierr);
  ierr = PetscStrArrayallocpy(displaynames,&ctx->displaynames);CHKERRQ(ierr);
  while (displaynames[j]) j++;
  ctx->ndisplayvariables = j;
  ierr = PetscMalloc1(ctx->ndisplayvariables,&ctx->displayvariables);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->ndisplayvariables,&ctx->displayvalues);CHKERRQ(ierr);
  j = 0;
  while (displaynames[j]) {
    k = 0;
    while (ctx->names[k]) {
      PetscBool flg;
      ierr = PetscStrcmp(displaynames[j],ctx->names[k],&flg);CHKERRQ(ierr);
      if (flg) {
        ctx->displayvariables[j] = k;
        break;
      }
      k++;
    }
    j++;
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGSetDisplayVariables - Sets the variables that are to be display in the monitor

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  displaynames - the names of the components, final string must be NULL

   Notes:
    If the TS object does not have a TSMonitorLGCtx associated with it then this function is ignored

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetVariableNames()
@*/
PetscErrorCode  TSMonitorLGSetDisplayVariables(TS ts,const char * const *displaynames)
{
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitor[i] == TSMonitorLGSolution) {
      ierr = TSMonitorLGCtxSetDisplayVariables((TSMonitorLGCtx)ts->monitorcontext[i],displaynames);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGSetTransform - Solution vector will be transformed by provided function before being displayed

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  transform - the transform function
.  destroy - function to destroy the optional context
-  ctx - optional context used by transform function

   Notes:
    If the TS object does not have a TSMonitorLGCtx associated with it then this function is ignored

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetVariableNames(), TSMonitorLGCtxSetTransform()
@*/
PetscErrorCode  TSMonitorLGSetTransform(TS ts,PetscErrorCode (*transform)(void*,Vec,Vec*),PetscErrorCode (*destroy)(void*),void *tctx)
{
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitor[i] == TSMonitorLGSolution) {
      ierr = TSMonitorLGCtxSetTransform((TSMonitorLGCtx)ts->monitorcontext[i],transform,destroy,tctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGCtxSetTransform - Solution vector will be transformed by provided function before being displayed

   Collective on TSLGCtx

   Input Parameters:
+  ts - the TS context
.  transform - the transform function
.  destroy - function to destroy the optional context
-  ctx - optional context used by transform function

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetVariableNames(), TSMonitorLGSetTransform()
@*/
PetscErrorCode  TSMonitorLGCtxSetTransform(TSMonitorLGCtx ctx,PetscErrorCode (*transform)(void*,Vec,Vec*),PetscErrorCode (*destroy)(void*),void *tctx)
{
  PetscFunctionBegin;
  ctx->transform    = transform;
  ctx->transformdestroy = destroy;
  ctx->transformctx = tctx;
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorLGError - Monitors progress of the TS solvers by plotting each component of the error
       in a time based line graph

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - TSMonitorLGCtx object created with TSMonitorLGCtxCreate()

   Level: intermediate

   Notes:
    Each process in a parallel run displays its component errors in a separate window

   The user must provide the solution using TSSetSolutionFunction() to use this monitor.

   Options Database Keys:
.  -ts_monitor_lg_error - create a graphical monitor of error history

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSSetSolutionFunction()
@*/
PetscErrorCode  TSMonitorLGError(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dummy)
{
  PetscErrorCode    ierr;
  TSMonitorLGCtx    ctx = (TSMonitorLGCtx)dummy;
  const PetscScalar *yy;
  Vec               y;

  PetscFunctionBegin;
  if (!step) {
    PetscDrawAxis axis;
    PetscInt      dim;
    ierr = PetscDrawLGGetAxis(ctx->lg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Error in solution as function of time","Time","Error");CHKERRQ(ierr);
    ierr = VecGetLocalSize(u,&dim);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(ctx->lg,dim);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(ctx->lg);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(u,&y);CHKERRQ(ierr);
  ierr = TSComputeSolutionFunction(ts,ptime,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&yy);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *yreal;
    PetscInt  i,n;
    ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&yreal);CHKERRQ(ierr);
    for (i=0; i<n; i++) yreal[i] = PetscRealPart(yy[i]);
    ierr = PetscDrawLGAddCommonPoint(ctx->lg,ptime,yreal);CHKERRQ(ierr);
    ierr = PetscFree(yreal);CHKERRQ(ierr);
  }
#else
  ierr = PetscDrawLGAddCommonPoint(ctx->lg,ptime,yy);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArrayRead(y,&yy);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    ierr = PetscDrawLGDraw(ctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(ctx->lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSPSwarmSolution - Graphically displays phase plots of DMSwarm particles on a scatter plot

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - the TSMonitorSPCtx object that contains all the options for the monitoring, this is created with TSMonitorSPCtxCreate()

   Options Database:
+ -ts_monitor_sp_swarm <n>          - Monitor the solution every n steps, or -1 for plotting only the final solution
. -ts_monitor_sp_swarm_retain <n>   - Retain n old points so we can see the history, or -1 for all points
- -ts_monitor_sp_swarm_phase <bool> - Plot in phase space, as opposed to coordinate space

   Level: intermediate

.seealso: TSMonitoSet()
@*/
PetscErrorCode TSMonitorSPSwarmSolution(TS ts, PetscInt step, PetscReal ptime, Vec u, void *dctx)
{
  TSMonitorSPCtx     ctx = (TSMonitorSPCtx) dctx;
  DM                 dm, cdm;
  const PetscScalar *yy;
  PetscReal         *y, *x;
  PetscInt           Np, p, dim = 2;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!step) {
    PetscDrawAxis axis;
    PetscReal     dmboxlower[2], dmboxupper[2];
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (dim != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Monitor only supports two dimensional fields");
    ierr = DMSwarmGetCellDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetBoundingBox(cdm, dmboxlower, dmboxupper);CHKERRQ(ierr);
    ierr = VecGetLocalSize(u, &Np);CHKERRQ(ierr);
    Np /= dim*2;
    ierr = PetscDrawSPGetAxis(ctx->sp,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Particles","X","V");CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLimits(axis, dmboxlower[0], dmboxupper[0], -5, 5);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetHoldLimits(axis, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDrawSPSetDimension(ctx->sp, Np);CHKERRQ(ierr);
    ierr = PetscDrawSPReset(ctx->sp);CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(u, &Np);CHKERRQ(ierr);
  Np /= dim*2;
  ierr = VecGetArrayRead(u,&yy);CHKERRQ(ierr);
  ierr = PetscMalloc2(Np, &x, Np, &y);CHKERRQ(ierr);
  /* get points from solution vector */
  for (p = 0; p < Np; ++p) {
    if (ctx->phase) {
      x[p] = PetscRealPart(yy[p*dim*2]);
      y[p] = PetscRealPart(yy[p*dim*2 + dim]);
    } else {
      x[p] = PetscRealPart(yy[p*dim*2]);
      y[p] = PetscRealPart(yy[p*dim*2 + 1]);
    }
  }
  ierr = VecRestoreArrayRead(u,&yy);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    PetscDraw draw;
    ierr = PetscDrawSPGetDraw(ctx->sp, &draw);CHKERRQ(ierr);
    if ((ctx->retain == 0) || (ctx->retain > 0 && !(step % ctx->retain))) {
      ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    }
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawSPReset(ctx->sp);CHKERRQ(ierr);
    ierr = PetscDrawSPAddPoint(ctx->sp, x, y);CHKERRQ(ierr);
    ierr = PetscDrawSPDraw(ctx->sp, PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscDrawSPSave(ctx->sp);CHKERRQ(ierr);
  }
  ierr = PetscFree2(x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorError - Monitors progress of the TS solvers by printing the 2 norm of the error at each timestep

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - unused context

   Level: intermediate

   The user must provide the solution using TSSetSolutionFunction() to use this monitor.

   Options Database Keys:
.  -ts_monitor_error - create a graphical monitor of error history

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSSetSolutionFunction()
@*/
PetscErrorCode TSMonitorError(TS ts,PetscInt step,PetscReal ptime,Vec u,PetscViewerAndFormat *vf)
{
  DM             dm;
  PetscDS        ds = NULL;
  PetscInt       Nf = -1, f;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  if (dm) {ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);}
  if (ds) {ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);}
  if (Nf <= 0) {
    Vec       y;
    PetscReal nrm;

    ierr = VecDuplicate(u,&y);CHKERRQ(ierr);
    ierr = TSComputeSolutionFunction(ts,ptime,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,u);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)vf->viewer,PETSCVIEWERASCII,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecNorm(y,NORM_2,&nrm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vf->viewer,"2-norm of error %g\n",(double)nrm);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)vf->viewer,PETSCVIEWERDRAW,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(y,vf->viewer);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  } else {
    PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void            **ctxs;
    Vec               v;
    PetscReal         ferrors[1];

    ierr = PetscMalloc2(Nf, &exactFuncs, Nf, &ctxs);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);CHKERRQ(ierr);}
    ierr = DMComputeL2FieldDiff(dm, ptime, exactFuncs, ctxs, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [", (int) step, (double) ptime);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      if (f > 0) {ierr = PetscPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%2.3g", (double) ferrors[f]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "]\n");CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

    ierr = PetscOptionsHasName(NULL, NULL, "-exact_vec_view", &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
      ierr = DMProjectFunction(dm, ptime, exactFuncs, ctxs, INSERT_ALL_VALUES, v);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) v, "Exact Solution");CHKERRQ(ierr);
      ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);
    }
    ierr = PetscFree2(exactFuncs, ctxs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorLGSNESIterations(TS ts,PetscInt n,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorLGCtx ctx = (TSMonitorLGCtx) monctx;
  PetscReal      x   = ptime,y;
  PetscErrorCode ierr;
  PetscInt       its;

  PetscFunctionBegin;
  if (n < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!n) {
    PetscDrawAxis axis;
    ierr = PetscDrawLGGetAxis(ctx->lg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Nonlinear iterations as function of time","Time","SNES Iterations");CHKERRQ(ierr);
    ierr = PetscDrawLGReset(ctx->lg);CHKERRQ(ierr);
    ctx->snes_its = 0;
  }
  ierr = TSGetSNESIterations(ts,&its);CHKERRQ(ierr);
  y    = its - ctx->snes_its;
  ierr = PetscDrawLGAddPoint(ctx->lg,&x,&y);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(n % ctx->howoften)) && (n > -1)) || ((ctx->howoften == -1) && (n == -1))) {
    ierr = PetscDrawLGDraw(ctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(ctx->lg);CHKERRQ(ierr);
  }
  ctx->snes_its = its;
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorLGKSPIterations(TS ts,PetscInt n,PetscReal ptime,Vec v,void *monctx)
{
  TSMonitorLGCtx ctx = (TSMonitorLGCtx) monctx;
  PetscReal      x   = ptime,y;
  PetscErrorCode ierr;
  PetscInt       its;

  PetscFunctionBegin;
  if (n < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!n) {
    PetscDrawAxis axis;
    ierr = PetscDrawLGGetAxis(ctx->lg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Linear iterations as function of time","Time","KSP Iterations");CHKERRQ(ierr);
    ierr = PetscDrawLGReset(ctx->lg);CHKERRQ(ierr);
    ctx->ksp_its = 0;
  }
  ierr = TSGetKSPIterations(ts,&its);CHKERRQ(ierr);
  y    = its - ctx->ksp_its;
  ierr = PetscDrawLGAddPoint(ctx->lg,&x,&y);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(n % ctx->howoften)) && (n > -1)) || ((ctx->howoften == -1) && (n == -1))) {
    ierr = PetscDrawLGDraw(ctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(ctx->lg);CHKERRQ(ierr);
  }
  ctx->ksp_its = its;
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorEnvelopeCtxCreate - Creates a context for use with TSMonitorEnvelope()

   Collective on TS

   Input Parameters:
.  ts  - the ODE solver object

   Output Parameter:
.  ctx - the context

   Level: intermediate

.seealso: TSMonitorLGTimeStep(), TSMonitorSet(), TSMonitorLGSolution(), TSMonitorLGError()

@*/
PetscErrorCode  TSMonitorEnvelopeCtxCreate(TS ts,TSMonitorEnvelopeCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorEnvelope - Monitors the maximum and minimum value of each component of the solution

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u  - current solution
-  dctx - the envelope context

   Options Database:
.  -ts_monitor_envelope

   Level: intermediate

   Notes:
    after a solve you can use TSMonitorEnvelopeGetBounds() to access the envelope

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorEnvelopeGetBounds(), TSMonitorEnvelopeCtxCreate()
@*/
PetscErrorCode  TSMonitorEnvelope(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dctx)
{
  PetscErrorCode       ierr;
  TSMonitorEnvelopeCtx ctx = (TSMonitorEnvelopeCtx)dctx;

  PetscFunctionBegin;
  if (!ctx->max) {
    ierr = VecDuplicate(u,&ctx->max);CHKERRQ(ierr);
    ierr = VecDuplicate(u,&ctx->min);CHKERRQ(ierr);
    ierr = VecCopy(u,ctx->max);CHKERRQ(ierr);
    ierr = VecCopy(u,ctx->min);CHKERRQ(ierr);
  } else {
    ierr = VecPointwiseMax(ctx->max,u,ctx->max);CHKERRQ(ierr);
    ierr = VecPointwiseMin(ctx->min,u,ctx->min);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorEnvelopeGetBounds - Gets the bounds for the components of the solution

   Collective on TS

   Input Parameter:
.  ts - the TS context

   Output Parameters:
+  max - the maximum values
-  min - the minimum values

   Notes:
    If the TS does not have a TSMonitorEnvelopeCtx associated with it then this function is ignored

   Level: intermediate

.seealso: TSMonitorSet(), TSMonitorDefault(), VecView(), TSMonitorLGSetDisplayVariables()
@*/
PetscErrorCode  TSMonitorEnvelopeGetBounds(TS ts,Vec *max,Vec *min)
{
  PetscInt i;

  PetscFunctionBegin;
  if (max) *max = NULL;
  if (min) *min = NULL;
  for (i=0; i<ts->numbermonitors; i++) {
    if (ts->monitor[i] == TSMonitorEnvelope) {
      TSMonitorEnvelopeCtx  ctx = (TSMonitorEnvelopeCtx) ts->monitorcontext[i];
      if (max) *max = ctx->max;
      if (min) *min = ctx->min;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorEnvelopeCtxDestroy - Destroys a context that was created  with TSMonitorEnvelopeCtxCreate().

   Collective on TSMonitorEnvelopeCtx

   Input Parameter:
.  ctx - the monitor context

   Level: intermediate

.seealso: TSMonitorLGCtxCreate(),  TSMonitorSet(), TSMonitorLGTimeStep()
@*/
PetscErrorCode  TSMonitorEnvelopeCtxDestroy(TSMonitorEnvelopeCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&(*ctx)->min);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ctx)->max);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSDMSwarmMonitorMoments - Monitors the first three moments of a DMSarm being evolved by the TS

  Not collective

  Input Parameters:
+ ts   - the TS context
. step - current timestep
. t    - current time
. u    - current solution
- ctx  - not used

  Options Database:
. -ts_dmswarm_monitor_moments

  Level: intermediate

  Notes:
  This requires a DMSwarm be attached to the TS.

.seealso: TSMonitorSet(), TSMonitorDefault(), DMSWARM
@*/
PetscErrorCode TSDMSwarmMonitorMoments(TS ts, PetscInt step, PetscReal t, Vec U, PetscViewerAndFormat *vf)
{
  DM                 sw;
  const PetscScalar *u;
  PetscReal          m = 1.0, totE = 0., totMom[3] = {0., 0., 0.};
  PetscInt           dim, d, Np, p;
  MPI_Comm           comm;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);
  if (!sw || step%ts->monitorFrequency != 0) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  Np  /= dim;
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      totE      += PetscRealPart(u[p*dim+d]*u[p*dim+d]);
      totMom[d] += PetscRealPart(u[p*dim+d]);
    }
  }
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) totMom[d] *= m;
  totE *= 0.5*m;
  ierr = PetscPrintf(comm, "Step %4D Total Energy: %10.8lf", step, (double) totE);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {ierr = PetscPrintf(comm, "    Total Momentum %c: %10.8lf", 'x'+d, (double) totMom[d]);CHKERRQ(ierr);}
  ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
