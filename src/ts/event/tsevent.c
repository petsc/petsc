#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

/*
  TSEventInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventInitialize(TSEvent event,TS ts,PetscReal t,Vec U)
{
  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(0);
  PetscValidPointer(event,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  event->ptime_prev = t;
  event->iterctr = 0;
  PetscCall((*event->eventhandler)(ts,t,U,event->fvalue_prev,event->ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  if (!*event) PetscFunctionReturn(0);
  if (--(*event)->refct > 0) {*event = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscFree((*event)->fvalue));
  PetscCall(PetscFree((*event)->fvalue_prev));
  PetscCall(PetscFree((*event)->fvalue_right));
  PetscCall(PetscFree((*event)->zerocrossing));
  PetscCall(PetscFree((*event)->side));
  PetscCall(PetscFree((*event)->direction));
  PetscCall(PetscFree((*event)->terminate));
  PetscCall(PetscFree((*event)->events_zero));
  PetscCall(PetscFree((*event)->vtol));

  for (i=0; i < (*event)->recsize; i++) {
    PetscCall(PetscFree((*event)->recorder.eventidx[i]));
  }
  PetscCall(PetscFree((*event)->recorder.eventidx));
  PetscCall(PetscFree((*event)->recorder.nevents));
  PetscCall(PetscFree((*event)->recorder.stepnum));
  PetscCall(PetscFree((*event)->recorder.time));

  PetscCall(PetscViewerDestroy(&(*event)->monitor));
  PetscCall(PetscFree(*event));
  PetscFunctionReturn(0);
}

/*@
  TSSetPostEventIntervalStep - Set the time-step used immediately following the event interval

  Logically Collective

  Input Parameters:
+ ts - time integration context
- dt - post event interval step

  Options Database Keys:
. -ts_event_post_eventinterval_step <dt> time-step after event interval

  Notes:
  TSSetPostEventIntervalStep allows one to set a time-step that is used immediately following an event interval.

  This function should be called from the postevent function set with TSSetEventHandler().

  The post event interval time-step should be selected based on the dynamics following the event.
  If the dynamics are stiff, a conservative (small) step should be used.
  If not, then a larger time-step can be used.

  Level: Advanced
  .seealso: TS, TSEvent, TSSetEventHandler()
@*/
PetscErrorCode TSSetPostEventIntervalStep(TS ts,PetscReal dt)
{
  PetscFunctionBegin;
  ts->event->timestep_posteventinterval = dt;
  PetscFunctionReturn(0);
}

/*@
   TSSetEventTolerances - Set tolerances for event zero crossings when using event handler

   Logically Collective

   Input Parameters:
+  ts - time integration context
.  tol - scalar tolerance, PETSC_DECIDE to leave current value
-  vtol - array of tolerances or NULL, used in preference to tol if present

   Options Database Keys:
.  -ts_event_tol <tol> - tolerance for event zero crossing

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
  PetscCheck(ts->event,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set the events first by calling TSSetEventHandler()");

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

/*@C
   TSSetEventHandler - Sets a function used for detecting events

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
   PetscErrorCode PetscEventHandler(TS ts,PetscReal t,Vec U,PetscScalar fvalue[],void* ctx)

   Input Parameters:
+  ts  - the TS context
.  t   - current time
.  U   - current iterate
-  ctx - [optional] context passed with eventhandler

   Output parameters:
.  fvalue    - function value of events at time t

   Calling sequence of postevent:
   PetscErrorCode PostEvent(TS ts,PetscInt nevents_zero,PetscInt events_zero[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events whose event function is zero
.  events_zero  - indices of local events which have reached zero
.  t            - current time
.  U            - current solution
.  forwardsolve - Flag to indicate whether TS is doing a forward solve (1) or adjoint solve (0)
-  ctx          - the context passed with eventhandler

   Level: intermediate

.seealso: TSCreate(), TSSetTimeStep(), TSSetConvergedReason()
@*/
PetscErrorCode TSSetEventHandler(TS ts,PetscInt nevents,PetscInt direction[],PetscBool terminate[],PetscErrorCode (*eventhandler)(TS,PetscReal,Vec,PetscScalar[],void*),PetscErrorCode (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*),void *ctx)
{
  TSAdapt        adapt;
  PetscReal      hmin;
  TSEvent        event;
  PetscInt       i;
  PetscBool      flg;
#if defined PETSC_USE_REAL_SINGLE
  PetscReal      tol=1e-4;
#else
  PetscReal      tol=1e-6;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (nevents) {
    PetscValidIntPointer(direction,3);
    PetscValidBoolPointer(terminate,4);
  }

  PetscCall(PetscNewLog(ts,&event));
  PetscCall(PetscMalloc1(nevents,&event->fvalue));
  PetscCall(PetscMalloc1(nevents,&event->fvalue_prev));
  PetscCall(PetscMalloc1(nevents,&event->fvalue_right));
  PetscCall(PetscMalloc1(nevents,&event->zerocrossing));
  PetscCall(PetscMalloc1(nevents,&event->side));
  PetscCall(PetscMalloc1(nevents,&event->direction));
  PetscCall(PetscMalloc1(nevents,&event->terminate));
  PetscCall(PetscMalloc1(nevents,&event->vtol));
  for (i=0; i < nevents; i++) {
    event->direction[i] = direction[i];
    event->terminate[i] = terminate[i];
    event->zerocrossing[i] = PETSC_FALSE;
    event->side[i] = 0;
  }
  PetscCall(PetscMalloc1(nevents,&event->events_zero));
  event->nevents = nevents;
  event->eventhandler = eventhandler;
  event->postevent = postevent;
  event->ctx = ctx;
  event->timestep_posteventinterval = ts->time_step;
  PetscCall(TSGetAdapt(ts,&adapt));
  PetscCall(TSAdaptGetStepLimits(adapt,&hmin,NULL));
  event->timestep_min = hmin;

  event->recsize = 8;  /* Initial size of the recorder */
  PetscOptionsBegin(((PetscObject)ts)->comm,((PetscObject)ts)->prefix,"TS Event options","TS");
  {
    PetscCall(PetscOptionsReal("-ts_event_tol","Scalar event tolerance for zero crossing check","TSSetEventTolerances",tol,&tol,NULL));
    PetscCall(PetscOptionsName("-ts_event_monitor","Print choices made by event handler","",&flg));
    PetscCall(PetscOptionsInt("-ts_event_recorder_initial_size","Initial size of event recorder","",event->recsize,&event->recsize,NULL));
    PetscCall(PetscOptionsReal("-ts_event_post_eventinterval_step","Time step after event interval","",event->timestep_posteventinterval,&event->timestep_posteventinterval,NULL));
    PetscCall(PetscOptionsReal("-ts_event_post_event_step","Time step after event","",event->timestep_postevent,&event->timestep_postevent,NULL));
    PetscCall(PetscOptionsReal("-ts_event_dt_min","Minimum time step considered for TSEvent","",event->timestep_min,&event->timestep_min,NULL));
  }
  PetscOptionsEnd();

  PetscCall(PetscMalloc1(event->recsize,&event->recorder.time));
  PetscCall(PetscMalloc1(event->recsize,&event->recorder.stepnum));
  PetscCall(PetscMalloc1(event->recsize,&event->recorder.nevents));
  PetscCall(PetscMalloc1(event->recsize,&event->recorder.eventidx));
  for (i=0; i < event->recsize; i++) {
    PetscCall(PetscMalloc1(event->nevents,&event->recorder.eventidx[i]));
  }
  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  for (i=0; i < event->nevents; i++) event->vtol[i] = tol;
  if (flg) PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,"stdout",&event->monitor));

  PetscCall(TSEventDestroy(&ts->event));
  ts->event = event;
  ts->event->refct = 1;
  PetscFunctionReturn(0);
}

/*
  TSEventRecorderResize - Resizes (2X) the event recorder arrays whenever the recording limit (event->recsize)
                          is reached.
*/
static PetscErrorCode TSEventRecorderResize(TSEvent event)
{
  PetscReal      *time;
  PetscInt       *stepnum;
  PetscInt       *nevents;
  PetscInt       **eventidx;
  PetscInt       i,fact=2;

  PetscFunctionBegin;

  /* Create large arrays */
  PetscCall(PetscMalloc1(fact*event->recsize,&time));
  PetscCall(PetscMalloc1(fact*event->recsize,&stepnum));
  PetscCall(PetscMalloc1(fact*event->recsize,&nevents));
  PetscCall(PetscMalloc1(fact*event->recsize,&eventidx));
  for (i=0; i < fact*event->recsize; i++) {
    PetscCall(PetscMalloc1(event->nevents,&eventidx[i]));
  }

  /* Copy over data */
  PetscCall(PetscArraycpy(time,event->recorder.time,event->recsize));
  PetscCall(PetscArraycpy(stepnum,event->recorder.stepnum,event->recsize));
  PetscCall(PetscArraycpy(nevents,event->recorder.nevents,event->recsize));
  for (i=0; i < event->recsize; i++) {
    PetscCall(PetscArraycpy(eventidx[i],event->recorder.eventidx[i],event->recorder.nevents[i]));
  }

  /* Destroy old arrays */
  for (i=0; i < event->recsize; i++) {
    PetscCall(PetscFree(event->recorder.eventidx[i]));
  }
  PetscCall(PetscFree(event->recorder.eventidx));
  PetscCall(PetscFree(event->recorder.nevents));
  PetscCall(PetscFree(event->recorder.stepnum));
  PetscCall(PetscFree(event->recorder.time));

  /* Set pointers */
  event->recorder.time = time;
  event->recorder.stepnum = stepnum;
  event->recorder.nevents = nevents;
  event->recorder.eventidx = eventidx;

  /* Double size */
  event->recsize *= fact;

  PetscFunctionReturn(0);
}

/*
   Helper routine to handle user postevents and recording
*/
static PetscErrorCode TSPostEvent(TS ts,PetscReal t,Vec U)
{
  TSEvent        event = ts->event;
  PetscBool      terminate = PETSC_FALSE;
  PetscBool      restart = PETSC_FALSE;
  PetscInt       i,ctr,stepnum;
  PetscBool      inflag[2],outflag[2];
  PetscBool      forwardsolve = PETSC_TRUE; /* Flag indicating that TS is doing a forward solve */

  PetscFunctionBegin;
  if (event->postevent) {
    PetscObjectState state_prev,state_post;
    PetscCall(PetscObjectStateGet((PetscObject)U,&state_prev));
    PetscCall((*event->postevent)(ts,event->nevents_zero,event->events_zero,t,U,forwardsolve,event->ctx));
    PetscCall(PetscObjectStateGet((PetscObject)U,&state_post));
    if (state_prev != state_post) restart = PETSC_TRUE;
  }

  /* Handle termination events and step restart */
  for (i=0; i<event->nevents_zero; i++) if (event->terminate[event->events_zero[i]]) terminate = PETSC_TRUE;
  inflag[0] = restart; inflag[1] = terminate;
  PetscCall(MPIU_Allreduce(inflag,outflag,2,MPIU_BOOL,MPI_LOR,((PetscObject)ts)->comm));
  restart = outflag[0]; terminate = outflag[1];
  if (restart) PetscCall(TSRestartStep(ts));
  if (terminate) PetscCall(TSSetConvergedReason(ts,TS_CONVERGED_EVENT));
  event->status = terminate ? TSEVENT_NONE : TSEVENT_RESET_NEXTSTEP;

  /* Reset event residual functions as states might get changed by the postevent callback */
  if (event->postevent) {
    PetscCall(VecLockReadPush(U));
    PetscCall((*event->eventhandler)(ts,t,U,event->fvalue,event->ctx));
    PetscCall(VecLockReadPop(U));
  }

  /* Cache current time and event residual functions */
  event->ptime_prev = t;
  for (i=0; i<event->nevents; i++)
    event->fvalue_prev[i] = event->fvalue[i];

  /* Record the event in the event recorder */
  PetscCall(TSGetStepNumber(ts,&stepnum));
  ctr = event->recorder.ctr;
  if (ctr == event->recsize) {
    PetscCall(TSEventRecorderResize(event));
  }
  event->recorder.time[ctr] = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (i=0; i<event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(0);
}

/* Uses Anderson-Bjorck variant of regula falsi method */
static inline PetscReal TSEventComputeStepSize(PetscReal tleft,PetscReal t,PetscReal tright,PetscScalar fleft,PetscScalar f,PetscScalar fright,PetscInt side,PetscReal dt)
{
  PetscReal new_dt, scal = 1.0;
  if (PetscRealPart(fleft)*PetscRealPart(f) < 0) {
    if (side == 1) {
      scal = (PetscRealPart(fright) - PetscRealPart(f))/PetscRealPart(fright);
      if (scal < PETSC_SMALL) scal = 0.5;
    }
    new_dt = (scal*PetscRealPart(fleft)*t - PetscRealPart(f)*tleft)/(scal*PetscRealPart(fleft) - PetscRealPart(f)) - tleft;
  } else {
    if (side == -1) {
      scal = (PetscRealPart(fleft) - PetscRealPart(f))/PetscRealPart(fleft);
      if (scal < PETSC_SMALL) scal = 0.5;
    }
    new_dt = (PetscRealPart(f)*tright - scal*PetscRealPart(fright)*t)/(PetscRealPart(f) - scal*PetscRealPart(fright)) - t;
  }
  return PetscMin(dt,new_dt);
}

static PetscErrorCode TSEventDetection(TS ts)
{
  TSEvent        event = ts->event;
  PetscReal      t;
  PetscInt       i;
  PetscInt       fvalue_sign,fvalueprev_sign;
  PetscInt       in,out;

  PetscFunctionBegin;
  PetscCall(TSGetTime(ts,&t));
  for (i=0; i < event->nevents; i++) {
    if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i]) {
      if (!event->iterctr) event->zerocrossing[i] = PETSC_TRUE;
      event->status = TSEVENT_LOCATED_INTERVAL;
      if (event->monitor) {
        PetscCall(PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D interval detected due to zero value (tol=%g) [%g - %g]\n",event->iterctr,i,(double)event->vtol[i],(double)event->ptime_prev,(double)t));
      }
      continue;
    }
    if (PetscAbsScalar(event->fvalue_prev[i]) < event->vtol[i]) continue; /* avoid duplicative detection if the previous endpoint is an event location */
    fvalue_sign = PetscSign(PetscRealPart(event->fvalue[i]));
    fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
    if (fvalueprev_sign != 0 && (fvalue_sign != fvalueprev_sign)) {
      if (!event->iterctr) event->zerocrossing[i] = PETSC_TRUE;
      event->status = TSEVENT_LOCATED_INTERVAL;
      if (event->monitor) {
        PetscCall(PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D interval detected due to sign change [%g - %g]\n",event->iterctr,i,(double)event->ptime_prev,(double)t));
      }
    }
  }
  in = (PetscInt)event->status;
  PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ts)));
  event->status = (TSEventStatus)out;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEventLocation(TS ts,PetscReal *dt)
{
  TSEvent        event = ts->event;
  PetscInt       i;
  PetscReal      t;
  PetscInt       fvalue_sign,fvalueprev_sign;
  PetscInt       rollback=0,in[2],out[2];

  PetscFunctionBegin;
  PetscCall(TSGetTime(ts,&t));
  event->nevents_zero = 0;
  for (i=0; i < event->nevents; i++) {
    if (event->zerocrossing[i]) {
      if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i] || *dt < event->timestep_min || PetscAbsReal((*dt)/((event->ptime_right-event->ptime_prev)/2)) < event->vtol[i]) { /* stopping criteria */
        event->status = TSEVENT_ZERO;
        event->fvalue_right[i] = event->fvalue[i];
        continue;
      }
      /* Compute new time step */
      *dt = TSEventComputeStepSize(event->ptime_prev,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],*dt);
      fvalue_sign = PetscSign(PetscRealPart(event->fvalue[i]));
      fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
      switch (event->direction[i]) {
      case -1:
        if (fvalue_sign < 0) {
          rollback = 1;
          event->fvalue_right[i] = event->fvalue[i];
          event->side[i] = 1;
        }
        break;
      case 1:
        if (fvalue_sign > 0) {
          rollback = 1;
          event->fvalue_right[i] = event->fvalue[i];
          event->side[i] = 1;
        }
        break;
      case 0:
        if (fvalue_sign != fvalueprev_sign) { /* trigger rollback only when there is a sign change */
          rollback = 1;
          event->fvalue_right[i] = event->fvalue[i];
          event->side[i] = 1;
        }
        break;
      }
      if (event->status == TSEVENT_PROCESSING) event->side[i] = -1;
    }
  }
  in[0] = (PetscInt)event->status; in[1] = rollback;
  PetscCall(MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ts)));
  event->status = (TSEventStatus)out[0]; rollback = out[1];
  /* If rollback is true, the status will be overwritten so that an event at the endtime of current time step will be postponed to guarantee corret order */
  if (rollback) event->status = TSEVENT_LOCATED_INTERVAL;
  if (event->status == TSEVENT_ZERO) {
    for (i=0; i < event->nevents; i++) {
      if (event->zerocrossing[i]) {
        if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i] || *dt < event->timestep_min || PetscAbsReal((*dt)/((event->ptime_right-event->ptime_prev)/2)) < event->vtol[i]) { /* stopping criteria */
          event->events_zero[event->nevents_zero++] = i;
          if (event->monitor) {
            PetscCall(PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D zero crossing located at time %g\n",event->iterctr,i,(double)t));
          }
          event->zerocrossing[i] = PETSC_FALSE;
        }
      }
      event->side[i] = 0;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSEventHandler(TS ts)
{
  TSEvent        event;
  PetscReal      t;
  Vec            U;
  PetscInt       i;
  PetscReal      dt,dt_min,dt_reset = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) PetscFunctionReturn(0);
  event = ts->event;

  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetSolution(ts,&U));

  if (event->status == TSEVENT_NONE) {
    event->timestep_prev = dt;
    event->ptime_end = t;
  }
  if (event->status == TSEVENT_RESET_NEXTSTEP) {
    /* user has specified a PostEventInterval dt */
    dt = event->timestep_posteventinterval;
    if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
      PetscReal maxdt = ts->max_time-t;
      dt = dt > maxdt ? maxdt : (PetscIsCloseAtTol(dt,maxdt,10*PETSC_MACHINE_EPSILON,0) ? maxdt : dt);
    }
    PetscCall(TSSetTimeStep(ts,dt));
    event->status = TSEVENT_NONE;
  }

  PetscCall(VecLockReadPush(U));
  PetscCall((*event->eventhandler)(ts,t,U,event->fvalue,event->ctx));
  PetscCall(VecLockReadPop(U));

  /* Detect the events */
  PetscCall(TSEventDetection(ts));

  /* Locate the events */
  if (event->status == TSEVENT_LOCATED_INTERVAL || event->status == TSEVENT_PROCESSING) {
    /* Approach the zero crosing by setting a new step size */
    PetscCall(TSEventLocation(ts,&dt));
    /* Roll back when new events are detected */
    if (event->status == TSEVENT_LOCATED_INTERVAL) {
      PetscCall(TSRollBack(ts));
      PetscCall(TSSetConvergedReason(ts,TS_CONVERGED_ITERATING));
      event->iterctr++;
    }
    PetscCall(MPIU_Allreduce(&dt,&dt_min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts)));
    if (dt_reset > 0.0 && dt_reset < dt_min) dt_min = dt_reset;
    PetscCall(TSSetTimeStep(ts,dt_min));
    /* Found the zero crossing */
    if (event->status == TSEVENT_ZERO) {
      PetscCall(TSPostEvent(ts,t,U));

      dt = event->ptime_end - t;
      if (PetscAbsReal(dt) < PETSC_SMALL) { /* we hit the event, continue with the candidate time step */
        dt = event->timestep_prev;
        event->status = TSEVENT_NONE;
      }
      if (event->timestep_postevent) { /* user has specified a PostEvent dt*/
        dt = event->timestep_postevent;
      }
      if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
        PetscReal maxdt = ts->max_time-t;
        dt = dt > maxdt ? maxdt : (PetscIsCloseAtTol(dt,maxdt,10*PETSC_MACHINE_EPSILON,0) ? maxdt : dt);
      }
      PetscCall(TSSetTimeStep(ts,dt));
      event->iterctr = 0;
    }
    /* Have not found the zero crosing yet */
    if (event->status == TSEVENT_PROCESSING) {
      if (event->monitor) {
        PetscCall(PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Stepping forward as no event detected in interval [%g - %g]\n",event->iterctr,(double)event->ptime_prev,(double)t));
      }
      event->iterctr++;
    }
  }
  if (event->status == TSEVENT_LOCATED_INTERVAL) { /* The step has been rolled back */
    event->status = TSEVENT_PROCESSING;
    event->ptime_right = t;
  } else {
    for (i=0; i < event->nevents; i++) event->fvalue_prev[i] = event->fvalue[i];
    event->ptime_prev = t;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSAdjointEventHandler(TS ts)
{
  TSEvent        event;
  PetscReal      t;
  Vec            U;
  PetscInt       ctr;
  PetscBool      forwardsolve=PETSC_FALSE; /* Flag indicating that TS is doing an adjoint solve */

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) PetscFunctionReturn(0);
  event = ts->event;

  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&U));

  ctr = event->recorder.ctr-1;
  if (ctr >= 0 && PetscAbsReal(t - event->recorder.time[ctr]) < PETSC_SMALL) {
    /* Call the user postevent function */
    if (event->postevent) {
      PetscCall((*event->postevent)(ts,event->recorder.nevents[ctr],event->recorder.eventidx[ctr],t,U,forwardsolve,event->ctx));
      event->recorder.ctr--;
    }
  }

  PetscBarrier((PetscObject)ts);
  PetscFunctionReturn(0);
}

/*@
  TSGetNumEvents - Get the numbers of events set

  Logically Collective

  Input Parameter:
. ts - the TS context

  Output Parameter:
. nevents - number of events

  Level: intermediate

.seealso: TSSetEventHandler()

@*/
PetscErrorCode TSGetNumEvents(TS ts,PetscInt * nevents)
{
  PetscFunctionBegin;
  *nevents = ts->event->nevents;
  PetscFunctionReturn(0);
}
