#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TSEventInitialize"
/*
  TSEventInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventInitialize(TSEvent event,TS ts,PetscReal t,Vec U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(0);
  PetscValidPointer(event,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  event->ptime_prev = t;
  ierr = (*event->eventhandler)(ts,t,U,event->fvalue_prev,event->ctx);CHKERRQ(ierr);
  /* Initialize the event recorder */
  event->recorder.ctr = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventDestroy"
PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  if (!*event) PetscFunctionReturn(0);
  ierr = PetscFree((*event)->fvalue);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscFree((*event)->direction);CHKERRQ(ierr);
  ierr = PetscFree((*event)->terminate);CHKERRQ(ierr);
  ierr = PetscFree((*event)->events_zero);CHKERRQ(ierr);
  ierr = PetscFree((*event)->vtol);CHKERRQ(ierr);
  for (i=0; i < MAXEVENTRECORDERS; i++) {
    ierr = PetscFree((*event)->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&(*event)->monitor);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetEventTolerances"
/*@
   TSSetEventTolerances - Set tolerances for event zero crossings when using event handler

   Logically Collective

   Input Arguments:
+  ts - time integration context
.  tol - scalar tolerance, PETSC_DECIDE to leave current value
-  vtol - array of tolerances or NULL, used in preference to tol if present

   Options Database Keys:
.  -ts_event_tol <tol> tolerance for event zero crossing

   Notes:
   Must call TSSetEventHandler() before setting the tolerances.

   The size of vtol is equal to the number of events.

   Level: beginner

.seealso: TS, TSEvent, TSSetEventHandler()
@*/
PetscErrorCode TSSetEventTolerances(TS ts,PetscReal tol,PetscReal vtol[])
{
  TSEvent        event;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (vtol) PetscValidRealPointer(vtol,3);
  if (!ts->event) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set the events first by calling TSSetEventHandler()");

  event = ts->event;
  if (vtol) {
    for (i=0; i < event->nevents; i++) event->vtol[i] = vtol[i];
  } else {
    if (tol != PETSC_DECIDE || tol != PETSC_DEFAULT) {
      for (i=0; i < event->nevents; i++) event->vtol[i] = tol;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetEventHandler"
/*@C
   TSSetEventHandler - Sets a monitoring function used for detecting events

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  nevents - number of local events
.  direction - direction of zero crossing to be detected. -1 => Zero crossing in negative direction,
               +1 => Zero crossing in positive direction, 0 => both ways (one for each event)
.  terminate - flag to indicate whether time stepping should be terminated after
               event is detected (one for each event)
.  eventhandler - event monitoring routine
.  postevent - [optional] post-event function
-  ctx       - [optional] user-defined context for private data for the
               event monitor and post event routine (use NULL if no
               context is desired)

   Calling sequence of eventhandler:
   PetscErrorCode EventHandler(TS ts,PetscReal t,Vec U,PetscScalar fvalue[],void* ctx)

   Input Parameters:
+  ts  - the TS context
.  t   - current time
.  U   - current iterate
-  ctx - [optional] context passed with eventhandler

   Output parameters:
.  fvalue    - function value of events at time t

   Calling sequence of postevent:
   PetscErrorCode PostEvent(TS ts,PetscInt nevents_zero, PetscInt events_zero[], PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events whose event function is zero
.  events_zero  - indices of local events which have reached zero
.  t            - current time
.  U            - current solution
.  forwardsolve - Flag to indicate whether TS is doing a forward solve (1) or adjoint solve (0)
-  ctx          - the context passed with eventhandler

   Level: intermediate

.keywords: TS, event, set, monitor

.seealso: TSCreate(), TSSetTimeStep(), TSSetConvergedReason()
@*/
PetscErrorCode TSSetEventHandler(TS ts,PetscInt nevents,PetscInt direction[],PetscBool terminate[],PetscErrorCode (*eventhandler)(TS,PetscReal,Vec,PetscScalar[],void*),PetscErrorCode (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*),void *ctx)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscInt       i;
  PetscBool      flg;
  PetscReal      tol=1e-6;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(direction,2);
  PetscValidIntPointer(terminate,3);

  ierr = PetscNewLog(ts,&event);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->direction);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->terminate);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->vtol);CHKERRQ(ierr);
  for (i=0; i < nevents; i++) {
    event->direction[i] = direction[i];
    event->terminate[i] = terminate[i];
  }
  ierr = PetscMalloc1(nevents,&event->events_zero);CHKERRQ(ierr);
  event->nevents = nevents;
  event->eventhandler = eventhandler;
  event->postevent = postevent;
  event->ctx = ctx;

  for (i=0; i < MAXEVENTRECORDERS; i++) {
    ierr = PetscMalloc1(event->nevents,&event->recorder.eventidx[i]);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(((PetscObject)ts)->comm,"","TS Event options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-ts_event_tol","Scalar event tolerance for zero crossing check","",tol,&tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-ts_event_monitor","Print choices made by event handler","",&flg);CHKERRQ(ierr);
  }
  PetscOptionsEnd();
  for (i=0; i < event->nevents; i++) event->vtol[i] = tol;
  if (flg) {ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,"stdout",&event->monitor);CHKERRQ(ierr);}

  ierr = TSEventDestroy(&ts->event);CHKERRQ(ierr);
  ts->event = event;
  PetscFunctionReturn(0);
}

/*
   Helper rutine to handle user postenvents and recording
*/
#undef __FUNCT__
#define __FUNCT__ "TSPostEvent"
static PetscErrorCode TSPostEvent(TS ts,PetscReal t,Vec U)
{
  PetscErrorCode ierr;
  TSEvent        event = ts->event;
  PetscBool      terminate = PETSC_FALSE;
  PetscInt       i,ctr,stepnum;
  PetscBool      ts_terminate;
  PetscBool      forwardsolve = PETSC_TRUE; /* Flag indicating that TS is doing a forward solve */

  PetscFunctionBegin;
  if (event->postevent) {
    ierr = (*event->postevent)(ts,event->nevents_zero,event->events_zero,t,U,forwardsolve,event->ctx);CHKERRQ(ierr);
  }

  /* Handle termination events */
  for (i=0; i < event->nevents_zero; i++) {
    terminate = (PetscBool)(terminate || event->terminate[event->events_zero[i]]);
  }
  ierr = MPIU_Allreduce(&terminate,&ts_terminate,1,MPIU_BOOL,MPI_LOR,((PetscObject)ts)->comm);CHKERRQ(ierr);
  if (ts_terminate) {
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);
    event->status = TSEVENT_NONE;
  } else {
    event->status = TSEVENT_RESET_NEXTSTEP;
  }

  event->ptime_prev = t;
  /* Reset event residual functions as states might get changed by the postevent callback */
  if (event->postevent) {ierr = (*event->eventhandler)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);}
  /* Cache current event residual functions */
  for (i=0; i < event->nevents; i++) event->fvalue_prev[i] = event->fvalue[i];

  /* Record the event in the event recorder */
  ierr = TSGetTimeStepNumber(ts,&stepnum);CHKERRQ(ierr);
  ctr = event->recorder.ctr;
  if (ctr == MAXEVENTRECORDERS) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Exceeded limit (=%d) for number of events recorded",MAXEVENTRECORDERS);
  event->recorder.time[ctr] = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (i=0; i < event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventHandler"
PetscErrorCode TSEventHandler(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscReal      t;
  Vec            U;
  PetscInt       i;
  PetscReal      dt,dt_min;
  PetscInt       rollback=0,in[2],out[2];
  PetscInt       fvalue_sign,fvalueprev_sign;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) PetscFunctionReturn(0);
  event = ts->event;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  if (event->status == TSEVENT_NONE) {
    event->timestep_prev = dt;
  }

  if (event->status == TSEVENT_RESET_NEXTSTEP) {
    /* Restore previous time step */
    dt = event->timestep_prev;
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    event->status = TSEVENT_NONE;
  }

  if (event->status == TSEVENT_NONE) {
    event->ptime_end = t;
  }

  ierr = (*event->eventhandler)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);

  for (i=0; i < event->nevents; i++) {
    if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i]) {
      event->status = TSEVENT_ZERO;
      continue;
    }
    fvalue_sign = PetscSign(PetscRealPart(event->fvalue[i]));
    fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
    if (fvalueprev_sign != 0 && (fvalue_sign != fvalueprev_sign) && (PetscAbsScalar(event->fvalue_prev[i]) > event->vtol[i])) {
      switch (event->direction[i]) {
      case -1:
        if (fvalue_sign < 0) {
          rollback = 1;
          /* Compute linearly interpolated new time step */
          dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
          if (event->monitor) {
            ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: Event %D interval located [%g - %g]\n",i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
          }
        }
        break;
      case 1:
        if (fvalue_sign > 0) {
          rollback = 1;
          /* Compute linearly interpolated new time step */
          dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
          if (event->monitor) {
            ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: Event %D interval located [%g - %g]\n",i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
          }
        }
        break;
      case 0:
        rollback = 1;
        /* Compute linearly interpolated new time step */
        dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
        if (event->monitor) {
          ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: Event %D interval located [%g - %g]\n",i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
        }
        break;
      }
    }
  }

  if (rollback) event->status = TSEVENT_LOCATED_INTERVAL;
  in[0] = event->status; in[1] = rollback;
  ierr = MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  event->status = (TSEventStatus)out[0]; rollback = out[1];
  if (rollback) event->status = TSEVENT_LOCATED_INTERVAL;

  event->nevents_zero = 0;
  if (event->status == TSEVENT_ZERO) {
    for (i=0; i < event->nevents; i++) {
      if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i]) {
        event->events_zero[event->nevents_zero++] = i;
        if (event->monitor) {
          ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: Event %D zero crossing at time %g\n",i,(double)t);CHKERRQ(ierr);
        }
      }
    }
    ierr = TSPostEvent(ts,t,U);CHKERRQ(ierr);

    dt = event->ptime_end - event->ptime_prev;
    if (PetscAbsReal(dt) < PETSC_SMALL) dt += event->timestep_prev; /* XXX Should be done better */
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (event->status == TSEVENT_LOCATED_INTERVAL) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    ts->steps--;
    ts->total_steps--;
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_ITERATING);CHKERRQ(ierr);
    event->status = TSEVENT_PROCESSING;
  } else {
    for (i=0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev = t;
    if (event->status == TSEVENT_PROCESSING) {
      dt = event->ptime_end - event->ptime_prev;
    }
  }

  ierr = MPIU_Allreduce(&dt,&dt_min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt_min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdjointEventHandler"
PetscErrorCode TSAdjointEventHandler(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscReal      t;
  Vec            U;
  PetscInt       ctr;
  PetscBool      forwardsolve=PETSC_FALSE; /* Flag indicating that TS is doing an adjoint solve */

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) PetscFunctionReturn(0);
  event = ts->event;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  ctr = event->recorder.ctr-1;
  if (ctr >= 0 && PetscAbsReal(t - event->recorder.time[ctr]) < PETSC_SMALL) {
    /* Call the user postevent function */
    if (event->postevent) {
      ierr = (*event->postevent)(ts,event->recorder.nevents[ctr],event->recorder.eventidx[ctr],t,U,forwardsolve,event->ctx);CHKERRQ(ierr);
      event->recorder.ctr--;
    }
  }

  PetscBarrier((PetscObject)ts);
  PetscFunctionReturn(0);
}
