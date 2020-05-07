#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

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
  event->iterctr = 0;
  ierr = (*event->eventhandler)(ts,t,U,event->fvalue_prev,event->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  if (!*event) PetscFunctionReturn(0);
  if (--(*event)->refct > 0) {*event = NULL; PetscFunctionReturn(0);}

  ierr = PetscFree((*event)->fvalue);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_right);CHKERRQ(ierr);
  ierr = PetscFree((*event)->zerocrossing);CHKERRQ(ierr);
  ierr = PetscFree((*event)->side);CHKERRQ(ierr);
  ierr = PetscFree((*event)->direction);CHKERRQ(ierr);
  ierr = PetscFree((*event)->terminate);CHKERRQ(ierr);
  ierr = PetscFree((*event)->events_zero);CHKERRQ(ierr);
  ierr = PetscFree((*event)->vtol);CHKERRQ(ierr);

  for (i=0; i < (*event)->recsize; i++) {
    ierr = PetscFree((*event)->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*event)->recorder.eventidx);CHKERRQ(ierr);
  ierr = PetscFree((*event)->recorder.nevents);CHKERRQ(ierr);
  ierr = PetscFree((*event)->recorder.stepnum);CHKERRQ(ierr);
  ierr = PetscFree((*event)->recorder.time);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&(*event)->monitor);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSSetPostEventIntervalStep - Set the time-step used immediately following the event interval

  Logically Collective

  Input Arguments:
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
  PetscErrorCode ierr;
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
    PetscValidIntPointer(direction,2);
    PetscValidIntPointer(terminate,3);
  }

  ierr = PetscNewLog(ts,&event);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue_right);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->zerocrossing);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->side);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->direction);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->terminate);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->vtol);CHKERRQ(ierr);
  for (i=0; i < nevents; i++) {
    event->direction[i] = direction[i];
    event->terminate[i] = terminate[i];
    event->zerocrossing[i] = PETSC_FALSE;
    event->side[i] = 0;
  }
  ierr = PetscMalloc1(nevents,&event->events_zero);CHKERRQ(ierr);
  event->nevents = nevents;
  event->eventhandler = eventhandler;
  event->postevent = postevent;
  event->ctx = ctx;
  event->timestep_posteventinterval = ts->time_step;

  event->recsize = 8;  /* Initial size of the recorder */
  ierr = PetscOptionsBegin(((PetscObject)ts)->comm,((PetscObject)ts)->prefix,"TS Event options","TS");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-ts_event_tol","Scalar event tolerance for zero crossing check","TSSetEventTolerances",tol,&tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-ts_event_monitor","Print choices made by event handler","",&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_event_recorder_initial_size","Initial size of event recorder","",event->recsize,&event->recsize,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_event_post_eventinterval_step","Time step after event interval","",event->timestep_posteventinterval,&event->timestep_posteventinterval,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc1(event->recsize,&event->recorder.time);CHKERRQ(ierr);
  ierr = PetscMalloc1(event->recsize,&event->recorder.stepnum);CHKERRQ(ierr);
  ierr = PetscMalloc1(event->recsize,&event->recorder.nevents);CHKERRQ(ierr);
  ierr = PetscMalloc1(event->recsize,&event->recorder.eventidx);CHKERRQ(ierr);
  for (i=0; i < event->recsize; i++) {
    ierr = PetscMalloc1(event->nevents,&event->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  for (i=0; i < event->nevents; i++) event->vtol[i] = tol;
  if (flg) {ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,"stdout",&event->monitor);CHKERRQ(ierr);}

  ierr = TSEventDestroy(&ts->event);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscReal      *time;
  PetscInt       *stepnum;
  PetscInt       *nevents;
  PetscInt       **eventidx;
  PetscInt       i,fact=2;

  PetscFunctionBegin;

  /* Create large arrays */
  ierr = PetscMalloc1(fact*event->recsize,&time);CHKERRQ(ierr);
  ierr = PetscMalloc1(fact*event->recsize,&stepnum);CHKERRQ(ierr);
  ierr = PetscMalloc1(fact*event->recsize,&nevents);CHKERRQ(ierr);
  ierr = PetscMalloc1(fact*event->recsize,&eventidx);CHKERRQ(ierr);
  for (i=0; i < fact*event->recsize; i++) {
    ierr = PetscMalloc1(event->nevents,&eventidx[i]);CHKERRQ(ierr);
  }

  /* Copy over data */
  ierr = PetscArraycpy(time,event->recorder.time,event->recsize);CHKERRQ(ierr);
  ierr = PetscArraycpy(stepnum,event->recorder.stepnum,event->recsize);CHKERRQ(ierr);
  ierr = PetscArraycpy(nevents,event->recorder.nevents,event->recsize);CHKERRQ(ierr);
  for (i=0; i < event->recsize; i++) {
    ierr = PetscArraycpy(eventidx[i],event->recorder.eventidx[i],event->recorder.nevents[i]);CHKERRQ(ierr);
  }

  /* Destroy old arrays */
  for (i=0; i < event->recsize; i++) {
    ierr = PetscFree(event->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(event->recorder.eventidx);CHKERRQ(ierr);
  ierr = PetscFree(event->recorder.nevents);CHKERRQ(ierr);
  ierr = PetscFree(event->recorder.stepnum);CHKERRQ(ierr);
  ierr = PetscFree(event->recorder.time);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  TSEvent        event = ts->event;
  PetscBool      terminate = PETSC_FALSE;
  PetscBool      restart = PETSC_FALSE;
  PetscInt       i,ctr,stepnum;
  PetscBool      inflag[2],outflag[2];
  PetscBool      forwardsolve = PETSC_TRUE; /* Flag indicating that TS is doing a forward solve */

  PetscFunctionBegin;
  if (event->postevent) {
    PetscObjectState state_prev,state_post;
    ierr = PetscObjectStateGet((PetscObject)U,&state_prev);CHKERRQ(ierr);
    ierr = (*event->postevent)(ts,event->nevents_zero,event->events_zero,t,U,forwardsolve,event->ctx);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&state_post);CHKERRQ(ierr);
    if (state_prev != state_post) restart = PETSC_TRUE;
  }

  /* Handle termination events and step restart */
  for (i=0; i<event->nevents_zero; i++) if (event->terminate[event->events_zero[i]]) terminate = PETSC_TRUE;
  inflag[0] = restart; inflag[1] = terminate;
  ierr = MPIU_Allreduce(inflag,outflag,2,MPIU_BOOL,MPI_LOR,((PetscObject)ts)->comm);CHKERRQ(ierr);
  restart = outflag[0]; terminate = outflag[1];
  if (restart) {ierr = TSRestartStep(ts);CHKERRQ(ierr);}
  if (terminate) {ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);}
  event->status = terminate ? TSEVENT_NONE : TSEVENT_RESET_NEXTSTEP;

  /* Reset event residual functions as states might get changed by the postevent callback */
  if (event->postevent) {
    ierr = VecLockReadPush(U);CHKERRQ(ierr);
    ierr = (*event->eventhandler)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);
    ierr = VecLockReadPop(U);CHKERRQ(ierr);
  }

  /* Cache current time and event residual functions */
  event->ptime_prev = t;
  for (i=0; i<event->nevents; i++)
    event->fvalue_prev[i] = event->fvalue[i];

  /* Record the event in the event recorder */
  ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
  ctr = event->recorder.ctr;
  if (ctr == event->recsize) {
    ierr = TSEventRecorderResize(event);CHKERRQ(ierr);
  }
  event->recorder.time[ctr] = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (i=0; i<event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(0);
}

/* Uses Anderson-Bjorck variant of regula falsi method */
PETSC_STATIC_INLINE PetscReal TSEventComputeStepSize(PetscReal tleft,PetscReal t,PetscReal tright,PetscScalar fleft,PetscScalar f,PetscScalar fright,PetscInt side,PetscReal dt)
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
    dt = event->timestep_posteventinterval;
    if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
      PetscReal maxdt = ts->max_time-t;

      dt = dt > maxdt ? maxdt : (PetscIsCloseAtTol(dt,maxdt,10*PETSC_MACHINE_EPSILON,0) ? maxdt : dt);
    }
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    event->status = TSEVENT_NONE;
  }

  if (event->status == TSEVENT_NONE) {
    event->ptime_end = t;
  }

  ierr = VecLockReadPush(U);CHKERRQ(ierr);
  ierr = (*event->eventhandler)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);
  ierr = VecLockReadPop(U);CHKERRQ(ierr);

  for (i=0; i < event->nevents; i++) {
    if (PetscAbsScalar(event->fvalue[i]) < event->vtol[i]) {
      event->status = TSEVENT_ZERO;
      event->fvalue_right[i] = event->fvalue[i];
      continue;
    }
    fvalue_sign = PetscSign(PetscRealPart(event->fvalue[i]));
    fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
    if (fvalueprev_sign != 0 && (fvalue_sign != fvalueprev_sign) && (PetscAbsScalar(event->fvalue_prev[i]) > event->vtol[i])) {
      switch (event->direction[i]) {
      case -1:
        if (fvalue_sign < 0) {
          rollback = 1;

          /* Compute new time step */
          dt = TSEventComputeStepSize(event->ptime_prev,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);

          if (event->monitor) {
            ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D interval detected [%g - %g]\n",event->iterctr,i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
          }
          event->fvalue_right[i] = event->fvalue[i];
          event->side[i] = 1;

          if (!event->iterctr) event->zerocrossing[i] = PETSC_TRUE;
          event->status = TSEVENT_LOCATED_INTERVAL;
        }
        break;
      case 1:
        if (fvalue_sign > 0) {
          rollback = 1;

          /* Compute new time step */
          dt = TSEventComputeStepSize(event->ptime_prev,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);

          if (event->monitor) {
            ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D interval detected [%g - %g]\n",event->iterctr,i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
          }
          event->fvalue_right[i] = event->fvalue[i];
          event->side[i] = 1;

          if (!event->iterctr) event->zerocrossing[i] = PETSC_TRUE;
          event->status = TSEVENT_LOCATED_INTERVAL;
        }
        break;
      case 0:
        rollback = 1;

        /* Compute new time step */
        dt = TSEventComputeStepSize(event->ptime_prev,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);

        if (event->monitor) {
          ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Event %D interval detected [%g - %g]\n",event->iterctr,i,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
        }
        event->fvalue_right[i] = event->fvalue[i];
        event->side[i] = 1;

        if (!event->iterctr) event->zerocrossing[i] = PETSC_TRUE;
        event->status = TSEVENT_LOCATED_INTERVAL;

        break;
      }
    }
  }

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
          ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: Event %D zero crossing at time %g located in %D iterations\n",i,(double)t,event->iterctr);CHKERRQ(ierr);
        }
        event->zerocrossing[i] = PETSC_FALSE;
      }
      event->side[i] = 0;
    }
    ierr = TSPostEvent(ts,t,U);CHKERRQ(ierr);

    dt = event->ptime_end - t;
    if (PetscAbsReal(dt) < PETSC_SMALL) { /* we hit the event, continue with the candidate time step */
      dt = event->timestep_prev;
      event->status = TSEVENT_NONE;
    }
    if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
      PetscReal maxdt = ts->max_time-t;

      dt = dt > maxdt ? maxdt : (PetscIsCloseAtTol(dt,maxdt,10*PETSC_MACHINE_EPSILON,0) ? maxdt : dt);
    }
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    event->iterctr = 0;
    PetscFunctionReturn(0);
  }

  if (event->status == TSEVENT_LOCATED_INTERVAL) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_ITERATING);CHKERRQ(ierr);
    event->status = TSEVENT_PROCESSING;
    event->ptime_right = t;
  } else {
    for (i=0; i < event->nevents; i++) {
      if (event->status == TSEVENT_PROCESSING && event->zerocrossing[i]) {
        /* Compute new time step */
        dt = TSEventComputeStepSize(event->ptime_prev,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);
        event->side[i] = -1;
      }
      event->fvalue_prev[i] = event->fvalue[i];
    }
    if (event->monitor && event->status == TSEVENT_PROCESSING) {
      ierr = PetscViewerASCIIPrintf(event->monitor,"TSEvent: iter %D - Stepping forward as no event detected in interval [%g - %g]\n",event->iterctr,(double)event->ptime_prev,(double)t);CHKERRQ(ierr);
    }
    event->ptime_prev = t;
  }

  if (event->status == TSEVENT_PROCESSING) event->iterctr++;

  ierr = MPIU_Allreduce(&dt,&dt_min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt_min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
