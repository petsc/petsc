#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the older PETSc versions (without TSSetPostEventStep())

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank (with zeros: 1.05, 9.05[terminating])\n"
                     "- one event function on rank = '1%size', equal to sin(pi*t), zeros = 1,...,10\n"
                     "TimeSpan = [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 9.21] plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined D\n"

                     "Options:\n"
                     "-dir     d : zero-crossing direction for events: 0 (default), 1, -1\n"
                     "-flg       : additional output in Postevent (default: nothing)\n"
                     "-errtol  e : error tolerance, for printing 'pass/fail' for located events (1e-5 by default)\n"
                     "-restart   : flag for TSRestartStep() in PostEvent (default: no)\n"
                     "-term      : flag to terminate at 9.05 event (true by default)\n"
                     "-dtpost  x : if x > 0, then on even PostEvent calls 1st-post-event-step = x is set,\n"
                     "                             on odd PostEvent calls 1st-post-event-step = PETSC_DECIDE is set,\n"
                     "             if x == 0, nothing happens (default)\n"
                     "-D       z : a small real number to define additional TimeSpan points (default = 0.02)\n"
                     "-dt2_at6 t : second time step set after event at t=6 (if nothing is specified, no action is done)\n"
                     "-mult7   m : after event at t=7, the linear system coeffs '0.2' are multiplied by m (default = 1.0)\n";

#define MAX_NFUNC 100  // max event functions per rank
#define MAX_NEV   5000 // max zero crossings for each rank

typedef struct {
  PetscMPIInt rank, size;
  PetscReal   pi;
  PetscReal   fvals[MAX_NFUNC]; // helper array for reporting the residuals
  PetscReal   evres[MAX_NEV];   // times of found zero-crossings
  PetscReal   ref[MAX_NEV];     // reference times of zero-crossings, for checking
  PetscInt    cnt;              // counter
  PetscInt    cntref;           // actual length of 'ref' on the given rank
  PetscBool   flg;              // flag for additional print in PostEvent
  PetscReal   errtol;           // error tolerance, for printing 'pass/fail' for located events (1e-5 by default)
  PetscBool   restart;          // flag for TSRestartStep() in PostEvent
  PetscBool   term;             // flag to terminate at 9.05 event
  PetscReal   dtpost;           // first post-event step
  PetscReal   dt2_at6;          // second time step set after event at t=6
  PetscReal   mult7;            // multiplier for coeffs at t=7
  PetscInt    postcnt;          // counter for PostEvent calls
  Mat         A;                // system matrix
  PetscInt    m;                // local size of A
} AppCtx;

PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);
PetscErrorCode Fill_mat(PetscReal coeff, PetscInt m, Mat A); // Fills the system matrix (2*2)

int main(int argc, char **argv)
{
  TS                ts;
  Vec               sol;
  PetscInt          n, dir0;
  PetscReal         tol = 1e-7, D = 0.02;
  PetscInt          dir[MAX_NFUNC];
  PetscBool         term[MAX_NFUNC], match;
  PetscScalar      *x;
  PetscReal         tspan[28], dtlast, tlast, tlast_expected, maxtime;
  PetscInt          tspan_size = PETSC_STATIC_ARRAY_LENGTH(tspan);
  AppCtx            ctx;
  TSConvergedReason reason;
  TSAdapt           adapt;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  setbuf(stdout, NULL);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  ctx.pi      = PetscAcosReal(-1.0);
  ctx.cnt     = 0;
  ctx.cntref  = 0;
  ctx.flg     = PETSC_FALSE;
  ctx.errtol  = 1e-5;
  ctx.restart = PETSC_FALSE;
  ctx.term    = PETSC_TRUE;
  ctx.dtpost  = 0;
  ctx.dt2_at6 = -2;
  ctx.mult7   = 1.0;
  ctx.postcnt = 0;
  ctx.m       = 0;

  // The linear problem has a 2*2 matrix. The matrix is constant
  if (ctx.rank == 0) ctx.m = 2;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, ctx.m, ctx.m, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 0, NULL, &ctx.A));
  PetscCallBack("Fill_mat", Fill_mat(0.2, ctx.m, ctx.A));
  PetscCall(MatCreateVecs(ctx.A, &sol, NULL));
  PetscCall(VecGetArray(sol, &x));
  if (ctx.rank == 0) { // initial conditions
    x[0] = 0;          // sin(0)
    x[1] = 1;          // cos(0)
  }
  PetscCall(VecRestoreArray(sol, &x));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_LINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, NULL));
  PetscCall(TSSetRHSJacobian(ts, ctx.A, ctx.A, TSComputeRHSJacobianConstant, NULL));

  PetscCall(TSSetTimeStep(ts, 0.099));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  // Set the event handling
  dir0 = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));             // desired zero-crossing direction
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));               // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));   // error tolerance for located events
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-restart", &ctx.restart, NULL)); // flag for TSRestartStep()
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-term", &ctx.term, NULL));       // flag to terminate at 9.05 event
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dtpost", &ctx.dtpost, NULL));   // post-event step
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt2_at6", &ctx.dt2_at6, NULL)); // second time step set after event at t=6
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mult7", &ctx.mult7, NULL));     // multiplier for coeffs at t=7
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));                 // small number for tspan

  n = 0;               // event counter
  if (ctx.rank == 0) { // first event -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.05;
  }
  if (ctx.rank == ctx.size - 1) { // second event (with optional termination) -- on last rank
    dir[n]    = dir0;
    term[n++] = ctx.term;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 9.05;
  }
  if (ctx.rank == 1 % ctx.size) { // third event -- on rank = 1%ctx.size
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;

    for (PetscInt i = 1; i < MAX_NEV - 2; i++) {
      if (i % 2 == 1 && dir0 <= 0) ctx.ref[ctx.cntref++] = i;
      if (i % 2 == 0 && dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));
  PetscCall(TSSetEventTolerances(ts, tol, NULL));

  // Set the time span
  for (PetscInt i = 0; i < 10; i++) {
    tspan[2 * i]     = 0.01 + i + (i == 7 ? -0.02 : 0);
    tspan[2 * i + 1] = 0.21 + i;
  }
  tspan[20] = 3;
  tspan[21] = 4;
  tspan[22] = 4 + D;
  tspan[23] = 5 - D;
  tspan[24] = 5;
  tspan[25] = 6 - D;
  tspan[26] = 6;
  tspan[27] = 6 + D;
  PetscCall(PetscSortReal(tspan_size, tspan));
  PetscCall(TSSetTimeSpan(ts, tspan_size, tspan));
  PetscCall(TSSetFromOptions(ts));

  // Solution
  PetscCall(TSSolve(ts, sol));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CONVERGED REASON: %" PetscInt_FMT " (TS_CONVERGED_EVENT == %" PetscInt_FMT ")\n", (PetscInt)reason, (PetscInt)TS_CONVERGED_EVENT));

  // The 4 columns printed are: [RANK] [time of event] [error w.r.t. reference] ["pass"/"fail"]
  for (PetscInt j = 0; j < ctx.cnt; j++) {
    PetscReal err = 10.0;
    if (j < ctx.cntref) err = PetscAbsReal(ctx.evres[j] - ctx.ref[j]);
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%g\t%g\t%s\n", ctx.rank, (double)ctx.evres[j], (double)err, err < ctx.errtol ? "pass" : "fail"));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  { // Verify evaluated solutions
    PetscInt         num_sols;
    Vec             *sols;
    const PetscReal *sol_times;
    PetscCall(TSGetEvaluationSolutions(ts, &num_sols, &sol_times, &sols));
    for (PetscInt i = 0; i < num_sols; i++) {
      PetscCheck(PetscIsCloseAtTol(tspan[i], sol_times[i], 1e-6, 1e2 * PETSC_MACHINE_EPSILON), PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Requested solution at time %g, but received time at %g", (double)tspan[i], (double)sol_times[i]);
    }
  }

  // print the final time and step
  PetscCall(TSGetTime(ts, &tlast));
  PetscCall(TSGetTimeStep(ts, &dtlast));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &match));

  PetscCall(TSGetMaxTime(ts, &maxtime));
  tlast_expected = ((dir0 == 1 || !ctx.term) ? maxtime : PetscMin(maxtime, 9.05));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final time = %g, max time = %g, %s\n", (double)tlast, (double)maxtime, PetscAbsReal(tlast - tlast_expected) < ctx.errtol ? "pass" : "fail"));

  if (match) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Adapt = none\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Last dt = %g\n", (double)dtlast));
  }

  PetscCall(MatDestroy(&ctx.A));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&sol));

  PetscCall(PetscFinalize());
  return 0;
}

/*
  User callback for defining the event-functions
*/
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx)
{
  PetscInt n   = 0;
  AppCtx  *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  // for the test purposes, event-functions are defined based on t
  // first event -- on rank-0
  if (Ctx->rank == 0) {
    if (t < 2.05) gval[n++] = 0.5 * (1 - PetscPowReal(t - 2.05, 12));
    else gval[n++] = 0.5;
  }

  // second event -- on last rank
  if (Ctx->rank == Ctx->size - 1) {
    if (t > 8.05) gval[n++] = 0.25 * (1 - PetscPowReal(t - 8.05, 12));
    else gval[n++] = 0.25;
  }

  // third event -- on rank = 1%ctx.size
  if (Ctx->rank == 1 % Ctx->size) gval[n++] = PetscSinReal(Ctx->pi * t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  User callback for the post-event stuff
*/
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx)
{
  AppCtx   *Ctx         = (AppCtx *)ctx;
  PetscBool mat_changed = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (Ctx->flg) {
    PetscCallBack("EventFunction", EventFunction(ts, t, U, Ctx->fvals, ctx));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] At t = %20.16g : %" PetscInt_FMT " events triggered, fvalues =", Ctx->rank, (double)t, nev_zero));
    for (PetscInt j = 0; j < nev_zero; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\t%g", (double)Ctx->fvals[evs_zero[j]]));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }

  if (Ctx->cnt + nev_zero < MAX_NEV)
    for (PetscInt i = 0; i < nev_zero; i++) Ctx->evres[Ctx->cnt++] = t; // save the repeating zeros separately for easier/unified testing

#ifdef NEW_VERSION
  Ctx->postcnt++; // sync
  if (Ctx->dtpost > 0) {
    if (Ctx->postcnt % 2 == 0) PetscCall(TSSetPostEventStep(ts, Ctx->dtpost));
    else PetscCall(TSSetPostEventStep(ts, PETSC_DECIDE));
  }
#endif

  // t==6: set the second post-event step
  if (PetscAbsReal(t - (PetscReal)6.0) < 0.01 && Ctx->dt2_at6 != -2) PetscCall(TSSetPostEventSecondStep(ts, Ctx->dt2_at6));

  // t==7: change the system matrix
  if (PetscAbsReal(t - 7) < 0.01 && Ctx->mult7 != 1) {
    PetscCallBack("Fill_mat", Fill_mat(0.2 * Ctx->mult7, Ctx->m, Ctx->A));
    PetscCall(TSSetRHSJacobian(ts, Ctx->A, Ctx->A, TSComputeRHSJacobianConstant, NULL));
    mat_changed = PETSC_TRUE;
  }

  if (Ctx->restart || mat_changed) PetscCall(TSRestartStep(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Fills the system matrix (2*2)
*/
PetscErrorCode Fill_mat(PetscReal coeff, PetscInt m, Mat A)
{
  PetscInt    inds[2];
  PetscScalar vals[4];

  PetscFunctionBeginUser;
  inds[0] = 0;
  inds[1] = 1;
  vals[0] = 0;
  vals[1] = coeff;
  vals[2] = -coeff;
  vals[3] = 0;
  PetscCall(MatSetValues(A, m, inds, m, inds, vals, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*---------------------------------------------------------------------------------------------*/
/*
  Note, in the tests below, -ts_event_post_event_step is occasionally set to -1,
  which corresponds to PETSC_DECIDE in the API. It is not a very good practice to
  explicitly specify -1 in this option. Rather, if PETSC_DECIDE behaviour is needed,
  simply remove this option altogether. This will result in using the defaults
  (which is PETSC_DECIDE).
*/
/*TEST
  test:
    suffix: 1
    requires: !single
    output_file: output/ex3span_1.out
    args: -ts_monitor -ts_adapt_type none -restart
    args: -dtpost 0.1127 -D 0.0015 -dir 0 -ts_max_time 9.8 -ts_dt 0.18
    nsize: 1

  test:
    suffix: 1single
    requires: single
    output_file: output/ex3span_1single.out
    args: -ts_monitor -ts_adapt_type none -restart -ts_event_dt_min 1e-6
    args: -dtpost 0.1127 -D 0.0015 -dir 0 -ts_max_time 9.8 -ts_dt 0.18
    nsize: 1

  test:
    suffix: 2
    output_file: output/ex3span_2.out
    args: -ts_event_dt_min 1e-6 -dtpost 1 -term 0 -ts_max_time 9.61
    nsize: 1

  test:
    suffix: 3none
    output_file: output/ex3span_3none.out
    args: -ts_event_dt_min 1e-6 -ts_adapt_type none -dir 0
    args: -ts_event_post_event_step {{-1 0.11}}
    args: -ts_event_post_event_second_step 0.12
    args: -dt2_at6 {{-2 0.08 0.15}}
    nsize: 3

  test:
    suffix: 3basic
    output_file: output/ex3span_3basic.out
    args: -ts_event_dt_min 1e-6 -ts_adapt_type basic -dir 0
    args: -ts_event_post_event_step {{-1 0.11}}
    args: -ts_event_post_event_second_step 0.12
    args: -dt2_at6 {{-2 0.08 0.15}}
    args: -mult7 {{1 2}}
    nsize: 2

  test:
    suffix: fin
    output_file: output/ex3span_fin.out
    args: -ts_max_time {{8.21 8.99 9 9.04 9.05 9.06 9.21 9.99 12}}
    args: -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1125
    args: -D 0.0025
    args: -dir {{0 -1 1}}
    args: -ts_dt 0.3025
    args: -ts_type {{rk bdf}}
    filter: grep "Final time ="
    filter_output: grep "Final time ="
    nsize: 2

  test:
    suffix: adaptmonitor
    requires: !single
    output_file: output/ex3span_adaptmonitor.out
    args: -ts_adapt_monitor -dir 1
    nsize: 1
TEST*/
