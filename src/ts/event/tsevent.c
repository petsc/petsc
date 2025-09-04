#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

/*
  Fills array sign[] with signs of array f[]. If abs(f[i]) < vtol[i], the zero sign is taken.
  All arrays should have length 'nev'
*/
static inline void TSEventCalcSigns(PetscInt nev, const PetscReal *f, const PetscReal *vtol, PetscInt *sign)
{
  for (PetscInt i = 0; i < nev; i++) {
    if (PetscAbsReal(f[i]) < vtol[i]) sign[i] = 0;
    else sign[i] = PetscSign(f[i]);
  }
}

/*
  TSEventInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventInitialize(TSEvent event, TS ts, PetscReal t, Vec U)
{
  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(event, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 4);
  event->ptime_prev    = t;
  event->iterctr       = 0;
  event->processing    = PETSC_FALSE;
  event->revisit_right = PETSC_FALSE;
  PetscCallBack("TSEvent indicator", (*event->indicator)(ts, t, U, event->fvalue_prev, event->ctx));
  TSEventCalcSigns(event->nevents, event->fvalue_prev, event->vtol, event->fsign_prev); // by this time event->vtol should have been defined
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscFunctionBegin;
  PetscAssertPointer(event, 1);
  if (!*event) PetscFunctionReturn(PETSC_SUCCESS);
  if (--(*event)->refct > 0) {
    *event = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscFree((*event)->fvalue_prev));
  PetscCall(PetscFree((*event)->fvalue));
  PetscCall(PetscFree((*event)->fvalue_right));
  PetscCall(PetscFree((*event)->fsign_prev));
  PetscCall(PetscFree((*event)->fsign));
  PetscCall(PetscFree((*event)->fsign_right));
  PetscCall(PetscFree((*event)->side));
  PetscCall(PetscFree((*event)->side_prev));
  PetscCall(PetscFree((*event)->justrefined_AB));
  PetscCall(PetscFree((*event)->gamma_AB));
  PetscCall(PetscFree((*event)->direction));
  PetscCall(PetscFree((*event)->terminate));
  PetscCall(PetscFree((*event)->events_zero));
  PetscCall(PetscFree((*event)->vtol));

  for (PetscInt i = 0; i < (*event)->recsize; i++) PetscCall(PetscFree((*event)->recorder.eventidx[i]));
  PetscCall(PetscFree((*event)->recorder.eventidx));
  PetscCall(PetscFree((*event)->recorder.nevents));
  PetscCall(PetscFree((*event)->recorder.stepnum));
  PetscCall(PetscFree((*event)->recorder.time));

  PetscCall(PetscViewerDestroy(&(*event)->monitor));
  PetscCall(PetscFree(*event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-section-header-unknown
/*@
  TSSetPostEventStep - Set the first time step to use after the event

  Logically Collective

  Input Parameters:
+ ts  - time integration context
- dt1 - first post event step

  Options Database Key:
. -ts_event_post_event_step <dt1> - first time step after the event

  Level: advanced

  Notes:
  `TSSetPostEventStep()` allows one to set a time step to use immediately following an event.
  Note, if `TSAdapt` is allowed to interfere and reject steps, a large 'dt1' set by `TSSetPostEventStep()` may get truncated,
  resulting in a smaller actual post-event step. See also the warning below regarding the `TSAdapt`.

  The post-event time steps should be selected based on the post-event dynamics.
  If the dynamics are stiff, or a significant jump in the equations or the state vector has taken place at the event,
  conservative (small) steps should be employed. If not, then larger time steps may be appropriate.

  This function accepts either a numerical value for `dt1`, or `PETSC_DECIDE`. The special value `PETSC_DECIDE` signals the event handler to follow
  the originally planned trajectory, and is assumed by default.

  To describe the way `PETSC_DECIDE` affects the post-event steps, consider a trajectory of time points t1 -> t2 -> t3 -> t4.
  Suppose the `TS` has reached and calculated the solution at point t3, and has planned the next move: t3 -> t4.
  At this moment, an event between t2 and t3 is detected, and after a few iterations it is resolved at point `te`, t2 < te < t3.
  After event `te`, two post-event steps can be specified: the first one dt1 (`TSSetPostEventStep()`),
  and the second one dt2 (`TSSetPostEventSecondStep()`). Both post-event steps can be either `PETSC_DECIDE`, or a number.
  Four different combinations are possible\:

  1. dt1 = `PETSC_DECIDE`, dt2 = `PETSC_DECIDE`. Then, after `te` `TS` goes to t3, and then to t4. This is the all-default behaviour.

  2. dt1 = `PETSC_DECIDE`, dt2 = x2 (numerical). Then, after `te` `TS` goes to t3, and then to t3+x2.

  3. dt1 = x1 (numerical), dt2 = x2 (numerical). Then, after `te` `TS` goes to te+x1, and then to te+x1+x2.

  4. dt1 = x1 (numerical), dt2 = `PETSC_DECIDE`. Then, after `te` `TS` goes to te+x1, and event handler does not interfere to the subsequent steps.

  In the special case when `te` == t3 with a good precision, the post-event step te -> t3 is not performed, so behaviour of (1) and (2) becomes\:

  1a. After `te` `TS` goes to t4, and event handler does not interfere to the subsequent steps.

  2a. After `te` `TS` goes to t4, and then to t4+x2.

  Warning! When the second post-event step (either `PETSC_DECIDE` or a numerical value) is managed by the event handler, i.e. in cases 1, 2, 3 and 2a,
  `TSAdapt` will never analyse (and never do a reasonable rejection of) the first post-event step. The first post-event step will always be accepted.
  In this situation, it is the user's responsibility to make sure the step size is appropriate!
  In cases 4 and 1a, however, `TSAdapt` will analyse the first post-event step, and is allowed to reject it.

  This function can be called not only in the initial setup, but also inside the `postevent()` callback set with `TSSetEventHandler()`,
  affecting the post-event steps for the current event, and the subsequent ones.
  Thus, the strategy of the post-event time step definition can be adjusted on the fly.
  In case several events are triggered in the given time point, only a single postevent handler is invoked,
  and the user is to determine what post-event time step is more appropriate in this situation.

  The default value is `PETSC_DECIDE`.

  Developer Notes:
  Event processing starts after visiting point t3, which means ts->adapt->dt_span_cached has been set to whatever value is required
  when planning the step t3 -> t4.

.seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`, `TSSetPostEventSecondStep()`
@*/
PetscErrorCode TSSetPostEventStep(TS ts, PetscReal dt1)
{
  PetscFunctionBegin;
  ts->event->timestep_postevent = dt1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-section-header-unknown
/*@
  TSSetPostEventSecondStep - Set the second time step to use after the event

  Logically Collective

  Input Parameters:
+ ts  - time integration context
- dt2 - second post event step

  Options Database Key:
. -ts_event_post_event_second_step <dt2> - second time step after the event

  Level: advanced

  Notes:
  `TSSetPostEventSecondStep()` allows one to set the second time step after the event.

  The post-event time steps should be selected based on the post-event dynamics.
  If the dynamics are stiff, or a significant jump in the equations or the state vector has taken place at the event,
  conservative (small) steps should be employed. If not, then larger time steps may be appropriate.

  This function accepts either a numerical value for `dt2`, or `PETSC_DECIDE` (default).

  To describe the way `PETSC_DECIDE` affects the post-event steps, consider a trajectory of time points t1 -> t2 -> t3 -> t4.
  Suppose the `TS` has reached and calculated the solution at point t3, and has planned the next move: t3 -> t4.
  At this moment, an event between t2 and t3 is detected, and after a few iterations it is resolved at point `te`, t2 < te < t3.
  After event `te`, two post-event steps can be specified: the first one dt1 (`TSSetPostEventStep()`),
  and the second one dt2 (`TSSetPostEventSecondStep()`). Both post-event steps can be either `PETSC_DECIDE`, or a number.
  Four different combinations are possible\:

  1. dt1 = `PETSC_DECIDE`, dt2 = `PETSC_DECIDE`. Then, after `te` `TS` goes to t3, and then to t4. This is the all-default behaviour.

  2. dt1 = `PETSC_DECIDE`, dt2 = x2 (numerical). Then, after `te` `TS` goes to t3, and then to t3+x2.

  3. dt1 = x1 (numerical), dt2 = x2 (numerical). Then, after `te` `TS` goes to te+x1, and then to te+x1+x2.

  4. dt1 = x1 (numerical), dt2 = `PETSC_DECIDE`. Then, after `te` `TS` goes to te+x1, and event handler does not interfere to the subsequent steps.

  In the special case when `te` == t3 with a good precision, the post-event step te -> t3 is not performed, so behaviour of (1) and (2) becomes\:

  1a. After `te` `TS` goes to t4, and event handler does not interfere to the subsequent steps.

  2a. After `te` `TS` goes to t4, and then to t4+x2.

  Warning! When the second post-event step (either `PETSC_DECIDE` or a numerical value) is managed by the event handler, i.e. in cases 1, 2, 3 and 2a,
  `TSAdapt` will never analyse (and never do a reasonable rejection of) the first post-event step. The first post-event step will always be accepted.
  In this situation, it is the user's responsibility to make sure the step size is appropriate!
  In cases 4 and 1a, however, `TSAdapt` will analyse the first post-event step, and is allowed to reject it.

  This function can be called not only in the initial setup, but also inside the `postevent()` callback set with `TSSetEventHandler()`,
  affecting the post-event steps for the current event, and the subsequent ones.

  The default value is `PETSC_DECIDE`.

.seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`, `TSSetPostEventStep()`
@*/
PetscErrorCode TSSetPostEventSecondStep(TS ts, PetscReal dt2)
{
  PetscFunctionBegin;
  ts->event->timestep_2nd_postevent = dt2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetEventTolerances - Set tolerances for event (indicator function) zero crossings

  Logically Collective

  Input Parameters:
+ ts   - time integration context
. tol  - tolerance, `PETSC_CURRENT` to leave the current value
- vtol - array of tolerances or `NULL`, used in preference to `tol` if present

  Options Database Key:
. -ts_event_tol <tol> - tolerance for event (indicator function) zero crossing

  Level: beginner

  Notes:
  One must call `TSSetEventHandler()` before setting the tolerances.

  The size of `vtol` should be equal to the number of events on the given process.

  This function can be also called from the `postevent()` callback set with `TSSetEventHandler()`,
  to adjust the tolerances on the fly.

.seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSSetEventTolerances(TS ts, PetscReal tol, PetscReal vtol[])
{
  TSEvent event;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (vtol) PetscAssertPointer(vtol, 3);
  PetscCheck(ts->event, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set the events first by calling TSSetEventHandler()");

  event = ts->event;
  if (vtol) {
    for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = vtol[i];
  } else {
    if (tol != (PetscReal)PETSC_CURRENT) {
      for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = tol;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSSetEventHandler - Sets functions and parameters used for indicating events and handling them

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
. nevents   - number of local events (i.e. managed by the given MPI process)
. direction - direction of zero crossing to be detected (one for each local event).
              `-1` => zero crossing in negative direction,
              `+1` => zero crossing in positive direction, `0` => both ways
. terminate - flag to indicate whether time stepping should be terminated after
              an event is detected (one for each local event)
. indicator - callback defininig the user indicator functions whose sign changes (see `direction`) mark presence of the events
. postevent - [optional] user post-event callback; it can change the solution, ODE etc at the time of the event
- ctx       - [optional] user-defined context for private data for the
              `indicator()` and `postevent()` routines (use `NULL` if no
              context is desired)

  Calling sequence of `indicator`:
+ ts     - the `TS` context
. t      - current time
. U      - current solution
. fvalue - output array with values of local indicator functions (length == `nevents`) for time t and state-vector U
- ctx    - the context passed as the final argument to `TSSetEventHandler()`

  Calling sequence of `postevent`:
+ ts           - the `TS` context
. nevents_zero - number of triggered local events (whose indicator function is marked as crossing zero, and direction is appropriate)
. events_zero  - indices of the triggered local events
. t            - current time
. U            - current solution
. forwardsolve - flag to indicate whether `TS` is doing a forward solve (`PETSC_TRUE`) or adjoint solve (`PETSC_FALSE`)
- ctx          - the context passed as the final argument to `TSSetEventHandler()`

  Options Database Keys:
+ -ts_event_tol <tol>                       - tolerance for zero crossing check of indicator functions
. -ts_event_monitor                         - print choices made by event handler
. -ts_event_recorder_initial_size <recsize> - initial size of event recorder
. -ts_event_post_event_step <dt1>           - first time step after event
. -ts_event_post_event_second_step <dt2>    - second time step after event
- -ts_event_dt_min <dt>                     - minimum time step considered for TSEvent

  Level: intermediate

  Notes:
  The indicator functions should be defined in the `indicator` callback using the components of solution `U` and/or time `t`.
  Note that `U` is `PetscScalar`-valued, and the indicator functions are `PetscReal`-valued. It is the user's responsibility to
  properly handle this difference, e.g. by applying `PetscRealPart()` or other appropriate conversion means.

  The full set of events is distributed (by the user design) across MPI processes, with each process defining its own local sub-set of events.
  However, the `postevent()` callback invocation is performed synchronously on all processes, including
  those processes which have not currently triggered any events.

.seealso: [](ch_ts), `TSEvent`, `TSCreate()`, `TSSetTimeStep()`, `TSSetConvergedReason()`
@*/
PetscErrorCode TSSetEventHandler(TS ts, PetscInt nevents, PetscInt direction[], PetscBool terminate[], PetscErrorCode (*indicator)(TS ts, PetscReal t, Vec U, PetscReal fvalue[], void *ctx), PetscErrorCode (*postevent)(TS ts, PetscInt nevents_zero, PetscInt events_zero[], PetscReal t, Vec U, PetscBool forwardsolve, void *ctx), void *ctx)
{
  TSAdapt   adapt;
  PetscReal hmin;
  TSEvent   event;
  PetscBool flg;
#if defined PETSC_USE_REAL_SINGLE
  PetscReal tol = 1e-4;
#else
  PetscReal tol = 1e-6;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (nevents) {
    PetscAssertPointer(direction, 3);
    PetscAssertPointer(terminate, 4);
  }
  PetscCall(PetscNew(&event));
  PetscCall(PetscMalloc1(nevents, &event->fvalue_prev));
  PetscCall(PetscMalloc1(nevents, &event->fvalue));
  PetscCall(PetscMalloc1(nevents, &event->fvalue_right));
  PetscCall(PetscMalloc1(nevents, &event->fsign_prev));
  PetscCall(PetscMalloc1(nevents, &event->fsign));
  PetscCall(PetscMalloc1(nevents, &event->fsign_right));
  PetscCall(PetscMalloc1(nevents, &event->side));
  PetscCall(PetscMalloc1(nevents, &event->side_prev));
  PetscCall(PetscMalloc1(nevents, &event->justrefined_AB));
  PetscCall(PetscMalloc1(nevents, &event->gamma_AB));
  PetscCall(PetscMalloc1(nevents, &event->direction));
  PetscCall(PetscMalloc1(nevents, &event->terminate));
  PetscCall(PetscMalloc1(nevents, &event->events_zero));
  PetscCall(PetscMalloc1(nevents, &event->vtol));
  for (PetscInt i = 0; i < nevents; i++) {
    event->direction[i]      = direction[i];
    event->terminate[i]      = terminate[i];
    event->justrefined_AB[i] = PETSC_FALSE;
    event->gamma_AB[i]       = 1;
    event->side[i]           = 2;
    event->side_prev[i]      = 0;
  }
  event->iterctr                = 0;
  event->processing             = PETSC_FALSE;
  event->revisit_right          = PETSC_FALSE;
  event->nevents                = nevents;
  event->indicator              = indicator;
  event->postevent              = postevent;
  event->ctx                    = ctx;
  event->timestep_postevent     = PETSC_DECIDE;
  event->timestep_2nd_postevent = PETSC_DECIDE;
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptGetStepLimits(adapt, &hmin, NULL));
  event->timestep_min = hmin;

  event->recsize = 8; /* Initial size of the recorder */
  PetscOptionsBegin(((PetscObject)ts)->comm, ((PetscObject)ts)->prefix, "TS Event options", "TS");
  {
    PetscCall(PetscOptionsReal("-ts_event_tol", "Tolerance for zero crossing check of indicator functions", "TSSetEventTolerances", tol, &tol, NULL));
    PetscCall(PetscOptionsName("-ts_event_monitor", "Print choices made by event handler", "", &flg));
    PetscCall(PetscOptionsInt("-ts_event_recorder_initial_size", "Initial size of event recorder", "", event->recsize, &event->recsize, NULL));
    PetscCall(PetscOptionsDeprecated("-ts_event_post_eventinterval_step", "-ts_event_post_event_second_step", "3.21", NULL));
    PetscCall(PetscOptionsReal("-ts_event_post_event_step", "First time step after event", "", event->timestep_postevent, &event->timestep_postevent, NULL));
    PetscCall(PetscOptionsReal("-ts_event_post_event_second_step", "Second time step after event", "", event->timestep_2nd_postevent, &event->timestep_2nd_postevent, NULL));
    PetscCall(PetscOptionsReal("-ts_event_dt_min", "Minimum time step considered for TSEvent", "", event->timestep_min, &event->timestep_min, NULL));
  }
  PetscOptionsEnd();

  PetscCall(PetscMalloc1(event->recsize, &event->recorder.time));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.stepnum));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.nevents));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.eventidx));
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscMalloc1(event->nevents, &event->recorder.eventidx[i]));
  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = tol;
  if (flg) PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)ts), "stdout", &event->monitor));

  PetscCall(TSEventDestroy(&ts->event));
  ts->event        = event;
  ts->event->refct = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEventRecorderResize - Resizes (2X) the event recorder arrays whenever the recording limit (event->recsize)
                          is reached.
*/
static PetscErrorCode TSEventRecorderResize(TSEvent event)
{
  PetscReal *time;
  PetscInt  *stepnum, *nevents;
  PetscInt **eventidx;
  PetscInt   fact = 2;

  PetscFunctionBegin;
  /* Create larger arrays */
  PetscCall(PetscMalloc1(fact * event->recsize, &time));
  PetscCall(PetscMalloc1(fact * event->recsize, &stepnum));
  PetscCall(PetscMalloc1(fact * event->recsize, &nevents));
  PetscCall(PetscMalloc1(fact * event->recsize, &eventidx));
  for (PetscInt i = 0; i < fact * event->recsize; i++) PetscCall(PetscMalloc1(event->nevents, &eventidx[i]));

  /* Copy over data */
  PetscCall(PetscArraycpy(time, event->recorder.time, event->recsize));
  PetscCall(PetscArraycpy(stepnum, event->recorder.stepnum, event->recsize));
  PetscCall(PetscArraycpy(nevents, event->recorder.nevents, event->recsize));
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscArraycpy(eventidx[i], event->recorder.eventidx[i], event->recorder.nevents[i]));

  /* Destroy old arrays */
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscFree(event->recorder.eventidx[i]));
  PetscCall(PetscFree(event->recorder.eventidx));
  PetscCall(PetscFree(event->recorder.nevents));
  PetscCall(PetscFree(event->recorder.stepnum));
  PetscCall(PetscFree(event->recorder.time));

  /* Set pointers */
  event->recorder.time     = time;
  event->recorder.stepnum  = stepnum;
  event->recorder.nevents  = nevents;
  event->recorder.eventidx = eventidx;

  /* Update the size */
  event->recsize *= fact;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Helper routine to handle user postevents and recording
*/
static PetscErrorCode TSPostEvent(TS ts, PetscReal t, Vec U)
{
  TSEvent   event        = ts->event;
  PetscBool restart      = PETSC_FALSE;
  PetscBool terminate    = PETSC_FALSE;
  PetscBool statechanged = PETSC_FALSE;
  PetscInt  ctr, stepnum;
  PetscBool inflag[3], outflag[3];
  PetscBool forwardsolve = PETSC_TRUE; // Flag indicating that TS is doing a forward solve

  PetscFunctionBegin;
  if (event->postevent) {
    PetscObjectState state_prev, state_post;
    PetscCall(PetscObjectStateGet((PetscObject)U, &state_prev));
    PetscCallBack("TSEvent post-event processing", (*event->postevent)(ts, event->nevents_zero, event->events_zero, t, U, forwardsolve, event->ctx)); // TODO update 'restart' here?
    PetscCall(PetscObjectStateGet((PetscObject)U, &state_post));
    if (state_prev != state_post) {
      restart      = PETSC_TRUE;
      statechanged = PETSC_TRUE;
    }
  }

  // Handle termination events and step restart
  for (PetscInt i = 0; i < event->nevents_zero; i++)
    if (event->terminate[event->events_zero[i]]) terminate = PETSC_TRUE;
  inflag[0] = restart;
  inflag[1] = terminate;
  inflag[2] = statechanged;
  PetscCallMPI(MPIU_Allreduce(inflag, outflag, 3, MPI_C_BOOL, MPI_LOR, ((PetscObject)ts)->comm));
  restart      = outflag[0];
  terminate    = outflag[1];
  statechanged = outflag[2];
  if (restart) PetscCall(TSRestartStep(ts));
  if (terminate) PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_EVENT));

  /*
    Recalculate the indicator functions and signs if the state has been changed by the user postevent callback.
    Note! If the state HAS NOT changed, the existing event->fsign (equal to zero) is kept, which:
    - might have been defined using the previous (now-possibly-overridden) event->vtol,
    - might have been set to zero on reaching a small time step rather than using the vtol criterion.
    This will enforce keeping event->fsign = 0 where the zero-crossings were actually marked,
    resulting in a more consistent behaviour of fsign's.
  */
  if (statechanged) {
    if (event->monitor) PetscCall(PetscPrintf(((PetscObject)ts)->comm, "TSEvent: at time %g the vector state has been changed by PostEvent, recalculating fvalues and signs\n", (double)t));
    PetscCall(VecLockReadPush(U));
    PetscCallBack("TSEvent indicator", (*event->indicator)(ts, t, U, event->fvalue, event->ctx));
    PetscCall(VecLockReadPop(U));
    TSEventCalcSigns(event->nevents, event->fvalue, event->vtol, event->fsign); // note, event->vtol might have been changed by the postevent()
  }

  // Record the event in the event recorder
  PetscCall(TSGetStepNumber(ts, &stepnum));
  ctr = event->recorder.ctr;
  if (ctr == event->recsize) PetscCall(TSEventRecorderResize(event));
  event->recorder.time[ctr]    = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (PetscInt i = 0; i < event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-sowing-chars
/*
  (modified) Anderson-Bjorck variant of regula falsi method, refines [tleft, t] or [t, tright] based on 'side' (-1 or +1).
  The scaling parameter is defined based on the 'justrefined' flag, the history of repeats of 'side', and the threshold.
  To escape certain failure modes, the algorithm may drift towards the bisection rule.
  The value pointed to by 'side_prev' gets updated.
  This function returns the new time step.

  The underlying pure Anderson-Bjorck algorithm was taken as described in
  J.M. Fernandez-Diaz, C.O. Menendez-Perez "A common framework for modified Regula Falsi methods and new methods of this kind", 2023.
  The modifications subsequently introduced have little effect on the behaviour for simple cases requiring only a few iterations
  (some minor convergence slowdown may take place though), but the effect becomes more pronounced for tough cases requiring many iterations.
  For the latter situation the speed-up may be order(s) of magnitude compared to the classical Anderson-Bjorck.
  The modifications (the threshold trick, and the drift towards bisection) were tested and tweaked
  based on a number of test functions from the mentioned paper.
*/
static inline PetscReal RefineAndersonBjorck(PetscReal tleft, PetscReal t, PetscReal tright, PetscReal fleft, PetscReal f, PetscReal fright, PetscInt side, PetscInt *side_prev, PetscBool justrefined, PetscReal *gamma)
{
  PetscReal      new_dt, scal = 1.0, scalB = 1.0, threshold = 0.0, power;
  PetscInt       reps     = 0;
  const PetscInt REPS_CAP = 8; // an upper bound to be imposed on 'reps' (set to 8, somewhat arbitrary number, found after some tweaking)

  // Preparations
  if (justrefined) {
    if (*side_prev * side > 0) *side_prev += side;     // the side keeps repeating -> increment the side counter (-ve or +ve)
    else *side_prev = side;                            // reset the counter
    reps      = PetscMin(*side_prev * side, REPS_CAP); // the length of the recent side-repeat series, including the current 'side'
    threshold = PetscPowReal(0.5, reps) * 0.1;         // ad-hoc strategy for threshold calculation (involved some tweaking)
  } else *side_prev = side;                            // initial reset of the counter

  // Time step calculation
  if (side == -1) {
    if (justrefined && fright != 0.0 && fleft != 0.0) {
      scal  = (fright - f) / fright;
      scalB = -f / fleft;
    }
  } else { // must be side == +1
    if (justrefined && fleft != 0.0 && fright != 0.0) {
      scal  = (fleft - f) / fleft;
      scalB = -f / fright;
    }
  }

  if (scal < threshold) scal = 0.5;
  if (reps > 1) *gamma *= scal; // side did not switch since the last time, accumulate gamma
  else *gamma = 1.0;            // side switched -> reset gamma
  power = PetscMax(0.0, (reps - 2.0) / (REPS_CAP - 2.0));
  scal  = PetscPowReal(scalB / *gamma, power) * (*gamma); // mix the Anderson-Bjorck scaling and Bisection scaling

  if (side == -1) new_dt = scal * fleft / (scal * fleft - f) * (t - tleft);
  else new_dt = f / (f - scal * fright) * (tright - t);
  /*
    In tough cases (e.g. a polynomial of high order), there is a failure mode for the standard Anderson-Bjorck,
    when the new proposed point jumps from one end-point of the bracket to the other, however the bracket
    is contracting very slowly. A larger threshold for 'scal' prevents entering this mode.
    On the other hand, if the iteration gets stuck near one end-point of the bracket, and the 'side' does not switch for a while,
    the 'scal' drifts towards the bisection approach (via scalB), ensuring stable convergence.
  */
  return new_dt;
}

/*
  Checks if the current point (t) is the zero-crossing location, based on the indicator function signs and direction[]:
  - using the dt_min criterion,
  - using the vtol criterion.
  The situation (fsign_prev, fsign) = (0, 0) is treated as staying in the near-zero-zone of the previous zero-crossing,
  and is not marked as a new zero-crossing.
  This function may update event->side[].
*/
static PetscErrorCode TSEventTestZero(TS ts, PetscReal t)
{
  TSEvent event = ts->event;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < event->nevents; i++) {
    const PetscBool bracket_is_left = (event->fsign_prev[i] * event->fsign[i] < 0 && event->fsign[i] * event->direction[i] >= 0) ? PETSC_TRUE : PETSC_FALSE;

    if (bracket_is_left && ((t - event->ptime_prev <= event->timestep_min) || event->revisit_right)) event->side[i] = 0;          // mark zero-crossing from dt_min; 'bracket_is_left' accounts for direction
    if (event->fsign[i] == 0 && event->fsign_prev[i] != 0 && event->fsign_prev[i] * event->direction[i] <= 0) event->side[i] = 0; // mark zero-crossing from vtol
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Checks if [fleft, f] or [f, fright] are 'brackets', i.e. intervals with the sign change, satisfying the 'direction'.
  The right interval is only checked if iterctr > 0 (i.e. Anderson-Bjorck refinement has started).
  The intervals like [0, x] and [x, 0] are not counted as brackets, i.e. intervals with the sign change.
  The function returns the 'side' value: -1 (left, or both are brackets), +1 (only right one), +2 (neither).
*/
static inline PetscInt TSEventTestBracket(PetscInt fsign_left, PetscInt fsign, PetscInt fsign_right, PetscInt direction, PetscInt iterctr)
{
  PetscInt side = 2;
  if (fsign_left * fsign < 0 && fsign * direction >= 0) side = -1;
  if (side != -1 && iterctr > 0 && fsign * fsign_right < 0 && fsign_right * direction >= 0) side = 1;
  return side;
}

/*
  Caps the time steps, accounting for evaluation time points.
  It uses 'event->timestep_cache' as a time step to calculate the tolerance for eval_times points detection. This
  is done since the event resolution may result in significant time step refinement, and we don't use these small steps for tolerances.
  To enhance the consistency of eval_times points detection, tolerance 'eval_times->worktol' is reused later in the TSSolve iteration.
  If a user-defined step is cut by this function, the input uncut step is saved to adapt->dt_span_cached.
  Flag 'user_dt' indicates if the step was defined by user.
*/
static inline PetscReal TSEvent_dt_cap(TS ts, PetscReal t, PetscReal dt, PetscBool user_dt)
{
  PetscReal res = dt;
  if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    PetscReal maxdt    = ts->max_time - t; // this may be overridden by eval_times
    PetscBool cut_made = PETSC_FALSE;
    PetscReal eps      = 10 * PETSC_MACHINE_EPSILON;
    if (ts->eval_times) {
      PetscInt   idx = ts->eval_times->time_point_idx;
      PetscInt   Ns  = ts->eval_times->num_time_points;
      PetscReal *st  = ts->eval_times->time_points;

      if (ts->eval_times->worktol == 0) ts->eval_times->worktol = ts->eval_times->reltol * ts->event->timestep_cache + ts->eval_times->abstol; // in case TSAdaptChoose() has not defined it
      if (idx < Ns && PetscIsCloseAtTol(t, st[idx], ts->eval_times->worktol, 0)) {                                                             // just hit a evaluation time point
        if (idx + 1 < Ns) maxdt = st[idx + 1] - t;                                                                                             // ok to use the next evaluation time point
        else maxdt = ts->max_time - t;                                                                                                         // can't use the next evaluation time point: they have finished
      } else if (idx < Ns) maxdt = st[idx] - t;                                                                                                // haven't hit a evaluation time point, use the nearest one
    }
    maxdt = PetscMin(maxdt, ts->max_time - t);
    PetscCheck((maxdt > eps) || (PetscAbsReal(maxdt) <= eps && PetscIsCloseAtTol(t, ts->max_time, eps, 0)), PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected state: bad maxdt (%g) in TSEvent_dt_cap()", (double)maxdt);

    if (PetscIsCloseAtTol(dt, maxdt, eps, 0)) res = maxdt; // no cut
    else {
      if (dt > maxdt) {
        res      = maxdt; // yes cut
        cut_made = PETSC_TRUE;
      } else res = dt; // no cut
    }
    if (ts->adapt && user_dt) { // only update dt_span_cached for the user-defined step
      if (cut_made) ts->adapt->dt_eval_times_cached = dt;
      else ts->adapt->dt_eval_times_cached = 0;
    }
  }
  return res;
}

/*
  Updates the left-end values
*/
static inline void TSEvent_update_left(TSEvent event, PetscReal t)
{
  for (PetscInt i = 0; i < event->nevents; i++) {
    event->fvalue_prev[i] = event->fvalue[i];
    event->fsign_prev[i]  = event->fsign[i];
  }
  event->ptime_prev = t;
}

/*
  Updates the right-end values
*/
static inline void TSEvent_update_right(TSEvent event, PetscReal t)
{
  for (PetscInt i = 0; i < event->nevents; i++) {
    event->fvalue_right[i] = event->fvalue[i];
    event->fsign_right[i]  = event->fsign[i];
  }
  event->ptime_right = t;
}

/*
  Updates the current values from the right-end values
*/
static inline PetscReal TSEvent_update_from_right(TSEvent event)
{
  for (PetscInt i = 0; i < event->nevents; i++) {
    event->fvalue[i] = event->fvalue_right[i];
    event->fsign[i]  = event->fsign_right[i];
  }
  return event->ptime_right;
}

static inline PetscBool Not_PETSC_DECIDE(PetscReal dt)
{
  return dt == PETSC_DECIDE ? PETSC_FALSE : PETSC_TRUE;
}

// PetscClangLinter pragma disable: -fdoc-section-spacing
// PetscClangLinter pragma disable: -fdoc-section-header-unknown
// PetscClangLinter pragma disable: -fdoc-section-header-spelling
// PetscClangLinter pragma disable: -fdoc-section-header-missing
// PetscClangLinter pragma disable: -fdoc-section-header-fishy-header
// PetscClangLinter pragma disable: -fdoc-param-list-func-parameter-documentation
// PetscClangLinter pragma disable: -fdoc-synopsis-missing-description
// PetscClangLinter pragma disable: -fdoc-sowing-chars
/*
  TSEventHandler - the main function to perform a single iteration of event detection.

  Developer notes:
  A) The 'event->iterctr > 0' is used as an indicator that Anderson-Bjorck refinement has started.
  B) If event->iterctr == 0, then justrefined_AB[i] is always false.
  C) The right-end quantities: ptime_right, fvalue_right[i] and fsign_right[i] are only guaranteed to be valid
  for event->iterctr > 0.
  D) If event->iterctr > 0, then event->processing is PETSC_TRUE; the opposite may not hold.
  E) When event->processing == PETSC_TRUE and event->iterctr == 0, the event handler iterations are complete, but
  the event handler continues managing the 1st and 2nd post-event steps. In this case the 1st post-event step
  proposed by the event handler is not checked by TSAdapt, and is always accepted (beware!).
  However, if the 2nd post-event step is not managed by the event handler (e.g. 1st = numerical, 2nd = PETSC_DECIDE),
  condition "E" does not hold, and TSAdapt may reject/adjust the 1st post-event step.
  F) event->side[i] may take values: 0 <=> point t is a zero-crossing for indicator function i (via vtol/dt_min criterion);
  -1/+1 <=> detected a bracket to the left/right of t for indicator function i; +2 <=> no brackets/zero-crossings.
  G) The signs event->fsign[i] (with values 0/-1/+1) are calculated for each new point. Zero sign is set if the function value is
  smaller than the tolerance. Besides, zero sign is enforced after marking a zero-crossing due to small bracket size criterion.

  The intervals with the indicator function sign change (i.e. containing the potential zero-crossings) are called 'brackets'.
  To find a zero-crossing, the algorithm first locates a bracket, and then sequentially subdivides it, generating a sequence
  of brackets whose length tends to zero. The bracket subdivision involves the (modified) Anderson-Bjorck method.

  Apart from the comments scattered throughout the code to clarify different lines and blocks,
  a few tricky aspects of the algorithm (and the underlying reasoning) are discussed in detail below:

  =Sign tracking=
  When a zero-crossing is found, the sign variable (event->fsign[i]) is set to zero for the current point t.
  This happens both for zero-crossings triggered via the vtol criterion, and those triggered via the dt_min
  criterion. After the event, as the TS steps forward, the current sign values are handed over to event->fsign_prev[i].
  The recalculation of signs is avoided if possible: e.g. if a 'vtol' criterion resulted in a zero-crossing at point t,
  but the subsequent call to postevent() handler decreased 'vtol', making the indicator function no longer "close to zero"
  at point t, the fsign[i] will still consistently keep the zero value. This allows avoiding the erroneous duplication
  of events:

  E.g. consider a bracket [t0, t2], where f0 < 0, f2 > 0, which resulted in a zero-crossing t1 with f1 < 0, abs(f1) < vtol.
  Suppose the postevent() handler changes vtol to vtol*, such that abs(f1) > vtol*. The TS makes a step t1 -> t3, where
  again f1 < 0, f3 > 0, and the event handler will find a new event near t1, which is actually a duplication of the
  original event at t1. The duplications are avoided by NOT counting the sign progressions 0 -> +1, or 0 -> -1
  as brackets. Tracking (instead of recalculating) the sign values makes this procedure work more consistently.

  The sign values are however recalculated if the postevent() callback has changed the current solution vector U
  (such a change resets everything).
  The sign value is also set to zero if the dt_min criterion has triggered the event. This allows the algorithm to
  work consistently, irrespective of the type of criterion involved (vtol/dt_min).

  =Event from min bracket=
  When the event handler ends up with a bracket [t0, t1] with size <= dt_min, a zero crossing is reported at t1,
  and never at t0. If such a bracket is discovered when TS is staying at t0, one more step forward (to t1) is necessary
  to mark the found event. This is the situation of revisiting t1, which is described below (see Revisiting).

  Why t0 is not reported as event location? Suppose it is, and let f0 < 0, f1 > 0. Also suppose that the
  postevent() handler has slightly changed the solution U, so the sign at t0 is recalculated: it equals -1. As the TS steps
  further: t0 -> t2, with sign0 == -1, and sign2 == +1, the event handler will locate the bracket [t0, t2], eventually
  resolving a new event near t1, i.e. finding a duplicate event.
  This situation is avoided by reporting the event at t1 in the first place.

  =Revisiting=
  When handling the situation with small bracket size, the TS solver may happen to visit the same point twice,
  but with different results.

  E.g. originally it discovered a bracket with sign change [t0, t10], and started resolving the zero-crossing,
  visiting the points t1,...,t9 : t0 < t1 < ... < t9 < t10. Suppose that at t9 the algorithm discovers
  that [t9, t10] is a bracket with the sign change it was looking for, and that |t10 - t9| is too small.
  So point t10 should be revisited and marked as the zero crossing (by the minimum bracket size criterion).
  On re-visiting t10, via the refined sequence of steps t0,...,t10, the TS solver may arrive at a solution U*
  different from the solution U it found at t10 originally. Hence, the indicator functions at t10 may become different,
  and the condition of the sign change, which existed originally, may disappear, breaking the logic of the algorithm.

  To handle such (-=unlikely=-, but possible) situations, two strategies can be considered:
  1) [not used here] Allow the brackets with sign change to disappear during iterations. The algorithm should be able
  to cleanly exit the iteration and leave all the objects/variables/caches involved in a valid state.
  2) [ADOPTED HERE!] On revisiting t10, the event handler reuses the indicator functions previously calculated for the
  original solution U. This U may be less precise than U*, but this trick does not allow the algorithm logic to break down.
  HOWEVER, the original U is not stored anywhere, it is essentially lost since the TS performed the rollback from it.
  On revisiting t10, the updated solution U* will inevitably be found and used everywhere EXCEPT the current
  indicator functions calculation, e.g. U* will be used in the postevent() handler call. Since t10 is the event location,
  the appropriate indicator-function-signs will be enforced to be 0 (regardless if the solution was U or U*).
  If the solution is then changed by the postevent(), the indicator-function-signs will be recalculated.

  Whether the algorithm is revisiting a point in the current TSEventHandler() call is flagged by 'event->revisit_right'.
*/
PetscErrorCode TSEventHandler(TS ts)
{
  TSEvent   event;
  PetscReal t, dt_next = 0.0;
  Vec       U;
  PetscInt  minsidein = 2, minsideout = 2; // minsideout is sync on all ranks
  PetscBool finished = PETSC_FALSE;        // should stay sync on all ranks
  PetscBool revisit_right_cache;           // [sync] flag for inner consistency checks

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);

  if (!ts->event) PetscFunctionReturn(PETSC_SUCCESS);
  event               = ts->event;
  event->nevents_zero = 0;
  revisit_right_cache = event->revisit_right;
  for (PetscInt i = 0; i < event->nevents; i++) event->side[i] = 2; // side's are reset on each new iteration
  if (event->iterctr == 0)
    for (PetscInt i = 0; i < event->nevents; i++) event->justrefined_AB[i] = PETSC_FALSE;

  PetscCall(TSGetTime(ts, &t));
  if (!event->processing) { // update the caches
    PetscReal dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    event->ptime_cache    = t;
    event->timestep_cache = dt; // the next TS move is planned to be: t -> t+dt
  }
  if (event->processing && event->iterctr == 0 && Not_PETSC_DECIDE(event->timestep_2nd_postevent)) { // update the caches while processing the post-event steps
    event->ptime_cache    = t;
    event->timestep_cache = event->timestep_2nd_postevent;
  }

  PetscCall(TSGetSolution(ts, &U)); // if revisiting, this will be the updated U* (see discussion on "Revisiting" in the Developer notes above)
  if (event->revisit_right) {
    PetscReal tr = TSEvent_update_from_right(event);
    PetscCheck(PetscAbsReal(tr - t) < PETSC_SMALL, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Inconsistent time value when performing 'revisiting' in TSEventHandler()");
  } else {
    PetscCall(VecLockReadPush(U));
    PetscCallBack("TSEvent indicator", (*event->indicator)(ts, t, U, event->fvalue, event->ctx)); // fill fvalue's at point 't'
    PetscCall(VecLockReadPop(U));
    TSEventCalcSigns(event->nevents, event->fvalue, event->vtol, event->fsign); // fill fvalue signs
  }
  PetscCall(TSEventTestZero(ts, t)); // check if the current point 't' is the event location; event->side[] may get updated

  for (PetscInt i = 0; i < event->nevents; i++) { // check for brackets on the left/right of 't'
    if (event->side[i] != 0) event->side[i] = TSEventTestBracket(event->fsign_prev[i], event->fsign[i], event->fsign_right[i], event->direction[i], event->iterctr);
    minsidein = PetscMin(minsidein, event->side[i]);
  }
  PetscCallMPI(MPIU_Allreduce(&minsidein, &minsideout, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)ts)));
  /*
    minsideout (sync on all ranks) indicates the minimum of the following states:
    -1 : [ptime_prev, t] is a bracket for some indicator-function-i
    +1 : [t, ptime_right] is a bracket for some indicator-function-i
     0 : t is a zero-crossing for some indicator-function-i
     2 : none of the above
  */
  PetscCheck(!event->revisit_right || minsideout == 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "minsideout != 0 when performing 'revisiting' in TSEventHandler()");

  if (minsideout == -1 || minsideout == +1) {                                                           // this if-branch will refine the left/right bracket
    const PetscReal bracket_size = (minsideout == -1) ? t - event->ptime_prev : event->ptime_right - t; // sync on all ranks

    if (minsideout == +1 && bracket_size <= event->timestep_min) { // check if the bracket (right) is small
      // [--------------------|-]
      dt_next              = bracket_size; // need one more step to get to event->ptime_right
      event->revisit_right = PETSC_TRUE;
      TSEvent_update_left(event, t);
      if (event->monitor)
        PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%d] TSEvent: iter %" PetscInt_FMT " - reached too small bracket [%g - %g], next stepping to its right end %g (revisiting)\n", PetscGlobalRank, event->iterctr, (double)event->ptime_prev,
                                         (double)event->ptime_right, (double)(event->ptime_prev + dt_next)));
    } else { // the bracket is not very small -> refine it
      // [--------|-------------]
      if (bracket_size <= 2 * event->timestep_min) dt_next = bracket_size / 2; // the bracket is almost small -> bisect it
      else {                                                                   // the bracket is not small -> use Anderson-Bjorck
        PetscReal dti_min = PETSC_MAX_REAL;
        for (PetscInt i = 0; i < event->nevents; i++) {
          if (event->side[i] == minsideout) { // only refine the appropriate brackets
            PetscReal dti = RefineAndersonBjorck(event->ptime_prev, t, event->ptime_right, event->fvalue_prev[i], event->fvalue[i], event->fvalue_right[i], event->side[i], &event->side_prev[i], event->justrefined_AB[i], &event->gamma_AB[i]);
            dti_min       = PetscMin(dti_min, dti);
          }
        }
        PetscCallMPI(MPIU_Allreduce(&dti_min, &dt_next, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)ts)));
        if (dt_next < event->timestep_min) dt_next = event->timestep_min;
        if (bracket_size - dt_next < event->timestep_min) dt_next = bracket_size - event->timestep_min;
      }

      if (minsideout == -1) { // minsideout == -1, update the right-end values, retain the left-end values
        TSEvent_update_right(event, t);
        PetscCall(TSRollBack(ts));
        PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING)); // e.g. to override TS_CONVERGED_TIME on reaching ts->max_time
      } else TSEvent_update_left(event, t);                          // minsideout == +1, update the left-end values, retain the right-end values

      for (PetscInt i = 0; i < event->nevents; i++) { // update the "Anderson-Bjorck" flags
        if (event->side[i] == minsideout) {
          event->justrefined_AB[i] = PETSC_TRUE; // only for these i's Anderson-Bjorck was invoked
          if (event->monitor)
            PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%d] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " refining the bracket with sign change [%g - %g], next stepping to %g\n", PetscGlobalRank, event->iterctr, i, (double)event->ptime_prev,
                                             (double)event->ptime_right, (double)(event->ptime_prev + dt_next)));
        } else event->justrefined_AB[i] = PETSC_FALSE; // for these i's Anderson-Bjorck was not invoked
      }
    }
    event->iterctr++;
    event->processing = PETSC_TRUE;
  } else if (minsideout == 0) { // found the appropriate zero-crossing (and no brackets to the left), finishing!
    // [--------0-------------]
    finished             = PETSC_TRUE;
    event->revisit_right = PETSC_FALSE;
    for (PetscInt i = 0; i < event->nevents; i++)
      if (event->side[i] == minsideout) {
        event->events_zero[event->nevents_zero++] = i;
        if (event->fsign[i] == 0) { // vtol was engaged
          if (event->monitor)
            PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%d] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " zero crossing located at time %g (tol=%g)\n", PetscGlobalRank, event->iterctr, i, (double)t, (double)event->vtol[i]));
        } else {               // dt_min was engaged
          event->fsign[i] = 0; // sign = 0 is enforced further
          if (event->monitor)
            PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%d] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " accepting time %g as event location, due to reaching too small bracket [%g - %g]\n", PetscGlobalRank, event->iterctr, i, (double)t,
                                             (double)event->ptime_prev, (double)t));
        }
      }
    event->iterctr++;
    event->processing = PETSC_TRUE;
  } else { // minsideout == 2: no brackets, no zero-crossings
    // [----------------------]
    PetscCheck(event->iterctr == 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected state (event->iterctr != 0) in TSEventHandler()");
    if (event->processing) {
      PetscReal dt2;
      if (event->timestep_2nd_postevent == PETSC_DECIDE) dt2 = event->timestep_cache;                            // (1)
      else dt2 = event->timestep_2nd_postevent;                                                                  // (2), (2a), (3)
      PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, dt2, Not_PETSC_DECIDE(event->timestep_2nd_postevent)))); // set the second post-event step
    }
    event->processing = PETSC_FALSE;
  }

  // if 'revisit_right' was flagged before the current iteration started, the iteration is expected to finish
  PetscCheck(!revisit_right_cache || finished, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected state of 'revisit_right_cache' in TSEventHandler()");

  if (finished) { // finished handling the current event
    PetscCall(TSPostEvent(ts, t, U));

    PetscReal dt1;
    if (event->timestep_postevent == PETSC_DECIDE) { // (1), (2)
      dt1               = event->ptime_cache - t;
      event->processing = PETSC_TRUE;
      if (PetscAbsReal(dt1) < PETSC_SMALL) { // (1a), (2a): the cached post-event point == event point
        dt1               = event->timestep_cache;
        event->processing = Not_PETSC_DECIDE(event->timestep_2nd_postevent);
      }
    } else {                                         // (3), (4)
      dt1               = event->timestep_postevent; // 1st post-event dt = user-provided value
      event->processing = Not_PETSC_DECIDE(event->timestep_2nd_postevent);
    }

    PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, dt1, Not_PETSC_DECIDE(event->timestep_postevent)))); // set the first post-event step
    event->iterctr = 0;
  } // if-finished

  if (event->iterctr == 0) TSEvent_update_left(event, t); // not found an event, or finished the event
  else {
    PetscCall(TSGetTime(ts, &t));                                              // update 't' to account for potential rollback
    PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, dt_next, PETSC_FALSE))); // continue resolving the event
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSAdjointEventHandler(TS ts)
{
  TSEvent   event;
  PetscReal t;
  Vec       U;
  PetscInt  ctr;
  PetscBool forwardsolve = PETSC_FALSE; // Flag indicating that TS is doing an adjoint solve

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (!ts->event) PetscFunctionReturn(PETSC_SUCCESS);
  event = ts->event;

  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetSolution(ts, &U));

  ctr = event->recorder.ctr - 1;
  if (ctr >= 0 && PetscAbsReal(t - event->recorder.time[ctr]) < PETSC_SMALL) {
    // Call the user post-event function
    if (event->postevent) {
      PetscCallBack("TSEvent post-event processing", (*event->postevent)(ts, event->recorder.nevents[ctr], event->recorder.eventidx[ctr], t, U, forwardsolve, event->ctx));
      event->recorder.ctr--;
    }
  }
  PetscCall(PetscBarrier((PetscObject)ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSGetNumEvents - Get the number of events defined on the given MPI process

  Logically Collective

  Input Parameter:
. ts - the `TS` context

  Output Parameter:
. nevents - the number of local events on each MPI process

  Level: intermediate

.seealso: [](ch_ts), `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSGetNumEvents(TS ts, PetscInt *nevents)
{
  PetscFunctionBegin;
  *nevents = ts->event->nevents;
  PetscFunctionReturn(PETSC_SUCCESS);
}
