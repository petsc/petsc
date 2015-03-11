
#include <petsc-private/tsimpl.h> /*I  "petscts.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TSEventMonitorInitialize"
/*
  TSEventMonitorInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventMonitorInitialize(TS ts)
{
  PetscErrorCode ierr;
  PetscReal      t;
  Vec            U;
  TSEvent        event=ts->event;

  PetscFunctionBegin;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&event->initial_timestep);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  event->ptime_prev = t;
  ierr = (*event->monitor)(ts,t,U,event->fvalue_prev,event->monitorcontext);CHKERRQ(ierr);

  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetEventMonitor"
/*@C
   TSSetEventMonitor - Sets a monitoring function used for detecting events

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  nevents - number of local events
.  direction - direction of zero crossing to be detected. -1 => Zero crossing in negative direction,
               +1 => Zero crossing in positive direction, 0 => both ways (one for each event)
.  terminate - flag to indicate whether time stepping should be terminated after
               event is detected (one for each event)
.  eventmonitor - event monitoring routine
.  postevent - [optional] post-event function
-  mectx - [optional] user-defined context for private data for the
              event monitor and post event routine (use NULL if no
              context is desired)

   Calling sequence of eventmonitor:
   PetscErrorCode EventMonitor(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void* mectx)

   Input Parameters:
+  ts  - the TS context
.  t   - current time
.  U   - current iterate
-  ctx - [optional] context passed with eventmonitor

   Output parameters:
.  fvalue    - function value of events at time t
               
   Calling sequence of postevent:
   PetscErrorCode PostEvent(TS ts,PetscInt nevents_zero, PetscInt events_zero, PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events whose event function is zero
.  events_zero  - indices of local events which have reached zero
.  t            - current time
.  U            - current solution
.  forwardsolve - Flag to indicate whether TS is doing a forward solve (1) or adjoint solve (0)
-  ctx          - the context passed with eventmonitor

   Level: intermediate

.keywords: TS, event, set, monitor

.seealso: TSCreate(), TSSetTimeStep(), TSSetConvergedReason()
@*/
PetscErrorCode TSSetEventMonitor(TS ts,PetscInt nevents,PetscInt *direction,PetscBool *terminate,PetscErrorCode (*eventmonitor)(TS,PetscReal,Vec,PetscScalar*,void*),PetscErrorCode (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*),void *mectx)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscNew(&event);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->direction);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevents,&event->terminate);CHKERRQ(ierr);
  for (i=0; i < nevents; i++) {
    event->direction[i] = direction[i];
    event->terminate[i] = terminate[i];
  }
  ierr = PetscMalloc1(nevents,&event->events_zero);CHKERRQ(ierr);
  event->monitor = eventmonitor;
  event->postevent = postevent;
  event->monitorcontext = (void*)mectx;
  event->nevents = nevents;

  for(i=0; i < MAXEVENTRECORDERS; i++) {
    ierr = PetscMalloc1(nevents,&event->recorder.eventidx[i]);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TS Event options","");CHKERRQ(ierr);
  {
    event->tol = 1.0e-6;
    ierr = PetscOptionsReal("-ts_event_tol","","",event->tol,&event->tol,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ts->event = event;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPostEvent"
/*
   TSPostEvent - Does post event processing by calling the user-defined postevent function

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events whose event function is zero
.  events_zero  - indices of local events which have reached zero
.  t            - current time
.  U            - current solution
.  forwardsolve - Flag to indicate whether TS is doing a forward solve (1) or adjoint solve (0)
-  ctx          - the context passed with eventmonitor

   Level: intermediate

.keywords: TS, event, set, monitor

.seealso: TSSetEventMonitor(),TSEvent
*/
#undef __FUNCT__
#define __FUNCT__ "TSPostEvent"
PetscErrorCode TSPostEvent(TS ts,PetscInt nevents_zero,PetscInt events_zero[],PetscReal t,Vec U,PetscBool forwardsolve,void *ctx)
{
  PetscErrorCode ierr;
  TSEvent        event=ts->event;
  PetscBool      terminate=PETSC_FALSE;
  PetscInt       i,ctr,stepnum;
  PetscBool      ts_terminate;

  PetscFunctionBegin;
  if (event->postevent) {
    ierr = (*event->postevent)(ts,nevents_zero,events_zero,t,U,forwardsolve,ctx);CHKERRQ(ierr);
  }
  for(i = 0; i < nevents_zero;i++) {
    terminate = (PetscBool)(terminate || event->terminate[events_zero[i]]);
  }
  ierr = MPI_Allreduce(&terminate,&ts_terminate,1,MPIU_BOOL,MPI_LOR,((PetscObject)ts)->comm);CHKERRQ(ierr);
  if (terminate) {
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);
    event->status = TSEVENT_NONE;
  } else {
    event->status = TSEVENT_RESET_NEXTSTEP;
  }

  /* Record the event in the event recorder */
  ierr = TSGetTimeStepNumber(ts,&stepnum);CHKERRQ(ierr);
  ctr = event->recorder.ctr;
  if (ctr == MAXEVENTRECORDERS) {
    SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Exceeded limit (=%d) for number of events recorded",MAXEVENTRECORDERS);
  }
  event->recorder.time[ctr] = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = nevents_zero;
  for(i=0; i < nevents_zero; i++) event->recorder.eventidx[ctr][i] = events_zero[i];
  event->recorder.ctr++;

  /* Reset the event residual functions as states might get changed by the postevent callback */
  ierr = (*event->monitor)(ts,t,U,event->fvalue_prev,event->monitorcontext);CHKERRQ(ierr);
  event->ptime_prev  = t;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventMonitorDestroy"
PetscErrorCode TSEventMonitorDestroy(TSEvent *event)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscFree((*event)->fvalue);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscFree((*event)->direction);CHKERRQ(ierr);
  ierr = PetscFree((*event)->terminate);CHKERRQ(ierr);
  ierr = PetscFree((*event)->events_zero);CHKERRQ(ierr);
  for(i=0; i < MAXEVENTRECORDERS; i++) {
    ierr = PetscFree((*event)->recorder.eventidx[i]);
  }
  ierr = PetscFree(*event);CHKERRQ(ierr);
  *event = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventMonitor"
PetscErrorCode TSEventMonitor(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event=ts->event;
  PetscReal      t;
  Vec            U;
  PetscInt       i;
  PetscReal      dt;
  TSEventStatus  status = event->status;
  PetscInt       rollback=0,in[2],out[2];
  PetscBool      forwardsolve=PETSC_TRUE; /* Flag indicating that TS is doing a forward solve */

  PetscFunctionBegin;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (event->status == TSEVENT_RESET_NEXTSTEP) {
    /* Take initial time step */
    dt = event->initial_timestep;
    ts->time_step = dt;
    event->status = TSEVENT_NONE;
  }

  if (event->status == TSEVENT_NONE) {
    event->tstepend   = t;
  }

  event->nevents_zero = 0;

  ierr = (*event->monitor)(ts,t,U,event->fvalue,event->monitorcontext);CHKERRQ(ierr);
  if (event->status != TSEVENT_NONE) {
    for (i=0; i < event->nevents; i++) {
      if (PetscAbsScalar(event->fvalue[i]) < event->tol) {
	event->status = TSEVENT_ZERO;
	event->events_zero[event->nevents_zero++] = i;
      }
    }
  }

  status = event->status;
  ierr = MPI_Allreduce((PetscEnum*)&status,(PetscEnum*)&event->status,1,MPIU_ENUM,MPI_MAX,((PetscObject)ts)->comm);CHKERRQ(ierr);

  if (event->status == TSEVENT_ZERO) {
    ierr = TSPostEvent(ts,event->nevents_zero,event->events_zero,t,U,forwardsolve,event->monitorcontext);CHKERRQ(ierr);
    dt = event->tstepend-t;
    if(PetscAbsReal(dt) < PETSC_SMALL) dt += event->initial_timestep;
    ts->time_step = dt;
    PetscFunctionReturn(0);
  }

  for (i = 0; i < event->nevents; i++) {
    PetscInt fvalue_sign,fvalueprev_sign;
    fvalue_sign = PetscSign(PetscRealPart(event->fvalue[i]));
    fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
    if (fvalueprev_sign != 0 && (fvalue_sign != fvalueprev_sign)) {
      switch (event->direction[i]) {
      case -1:
	if (fvalue_sign < 0) {
	  rollback = 1;
	  /* Compute linearly interpolated new time step */
	  dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
	}
	break;
      case 1:
	if (fvalue_sign > 0) { 
	  rollback = 1;
	  /* Compute linearly interpolated new time step */
	  dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
	}
	break;
      case 0: 
	rollback = 1; 
	/* Compute linearly interpolated new time step */
	dt = PetscMin(dt,PetscRealPart(-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i])));
	break;
      }
    }
  }
  if (rollback) event->status = TSEVENT_LOCATED_INTERVAL;
  
  in[0] = event->status;
  in[1] = rollback;
  ierr = MPI_Allreduce(in,out,2,MPIU_INT,MPI_MAX,((PetscObject)ts)->comm);CHKERRQ(ierr);
  
  rollback = out[1];
  if (rollback) {
    event->status = TSEVENT_LOCATED_INTERVAL;
  }

  if (event->status == TSEVENT_LOCATED_INTERVAL) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    ts->steps--;
    ts->total_steps--;
    event->status = TSEVENT_PROCESSING;
  } else {
    for (i = 0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev  = t;
    if (event->status == TSEVENT_PROCESSING) {
      dt = event->tstepend - event->ptime_prev;
    }
  }
  ierr = MPI_Allreduce(&dt,&(ts->time_step),1,MPIU_REAL,MPI_MIN,((PetscObject)ts)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdjointEventMonitor"
PetscErrorCode TSAdjointEventMonitor(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event=ts->event;
  PetscReal      t;
  Vec            U;
  PetscInt       ctr;
  PetscBool      forwardsolve=PETSC_FALSE; /* Flag indicating that TS is doing an adjoint solve */

  PetscFunctionBegin;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  ctr = event->recorder.ctr-1;
  if(ctr >= 0 && PetscAbsReal(t - event->recorder.time[ctr]) < PETSC_SMALL) {
    /* Call the user postevent function */
    if(event->postevent) {
      ierr = (*event->postevent)(ts,event->recorder.nevents[ctr],event->recorder.eventidx[ctr],t,U,forwardsolve,event->monitorcontext);CHKERRQ(ierr);
      event->recorder.ctr--;
    }
  }

  PetscBarrier((PetscObject)ts);
  PetscFunctionReturn(0);
}



