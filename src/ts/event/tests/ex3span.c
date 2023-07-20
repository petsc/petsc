#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the old (current) TSEvent code

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank (with zeros: 1.05, 9.05[terminating])\n"
                     "- one event function on rank = '1%size', equal to sin(pi*t), zeros = 1,...,10\n"
                     "TimeSpan = [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 9.21] plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined D\n"

                     "Options:\n"
                     "-dir    d : zero-crossing direction for events: 0, 1, -1\n"
                     "-flg      : additional output in Postevent\n"
                     "-restart  : flag for TSRestartStep() in PostEvent\n"
                     "-term     : flag to terminate at 9.05 event (true by default)\n"
                     "-dtpost x : if x > 0, then on even PostEvent calls dt_postevent = x is set, on odd PostEvent calls dt_postevent = 0 is set,\n"
                     "            if x == 0, nothing happens\n"
                     "-D      z : a small number to define additional TimeSpan points\n";

#define MAX_NFUNC 100  // max event functions per rank
#define MAX_NEV   5000 // max zero crossings for each rank

typedef struct {
  PetscMPIInt rank, size;
  PetscReal   pi;
  PetscReal   fvals[MAX_NFUNC]; // helper array for reporting the residuals
  PetscReal   evres[MAX_NEV];   // times of found zero-crossings
  PetscInt    evnum[MAX_NEV];   // number of zero-crossings at each time
  PetscInt    cnt;              // counter
  PetscBool   flg;              // flag for additional print in PostEvent
  PetscBool   restart;          // flag for TSRestartStep() in PostEvent
  PetscBool   term;             // flag to terminate at 9.05 event
  PetscReal   dtpost;           // post-event step
  PetscInt    postcnt;          // counter for PostEvent calls
} AppCtx;

PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);

int main(int argc, char **argv)
{
  TS                ts;
  Mat               A;
  Vec               sol;
  PetscInt          n, dir0, m = 0;
  PetscReal         tol = 1e-7, D = 0.02;
  PetscInt          dir[MAX_NFUNC], inds[2];
  PetscBool         term[MAX_NFUNC], match;
  PetscScalar      *x, vals[4];
  PetscReal         tspan[28], dtlast;
  AppCtx            ctx;
  TSConvergedReason reason;
  TSAdapt           adapt;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  setbuf(stdout, NULL);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  ctx.pi      = PetscAcosReal(-1.0);
  ctx.cnt     = 0;
  ctx.flg     = PETSC_FALSE;
  ctx.restart = PETSC_FALSE;
  ctx.term    = PETSC_TRUE;
  ctx.dtpost  = 0;
  ctx.postcnt = 0;

  // The linear problem has a 2*2 matrix. The matrix is constant
  if (ctx.rank == 0) m = 2;
  inds[0] = 0;
  inds[1] = 1;
  vals[0] = 0;
  vals[1] = 0.2;
  vals[2] = -0.2;
  vals[3] = 0;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, m, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 0, NULL, &A));
  PetscCall(MatSetValues(A, m, inds, m, inds, vals, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));

  PetscCall(MatCreateVecs(A, &sol, NULL));
  PetscCall(VecGetArray(sol, &x));
  if (ctx.rank == 0) { // initial conditions
    x[0] = 0;          // sin(0)
    x[1] = 1;          // cos(0)
  }
  PetscCall(VecRestoreArray(sol, &x));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_LINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, NULL));
  PetscCall(TSSetRHSJacobian(ts, A, A, TSComputeRHSJacobianConstant, NULL));

  PetscCall(TSSetTimeStep(ts, 0.099));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  // Set the event handling
  dir0 = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));             // desired zero-crossing direction
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));               // flag for additional output
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-restart", &ctx.restart, NULL)); // flag for TSRestartStep()
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-term", &ctx.term, NULL));       // flag to terminate at 9.05 event
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dtpost", &ctx.dtpost, NULL));   // post-event step
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));                 // small number for tspan

  n = 0;               // event counter
  if (ctx.rank == 0) { // first event -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;
  }
  if (ctx.rank == ctx.size - 1) { // second event (with optional termination) -- on last rank
    dir[n]    = dir0;
    term[n++] = ctx.term;
  }
  if (ctx.rank == 1 % ctx.size) { // third event -- on rank = 1%ctx.size
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;
  }
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
  PetscCall(PetscSortReal(28, tspan));
  PetscCall(TSSetTimeSpan(ts, 28, tspan));
  PetscCall(TSSetFromOptions(ts));

  // Solution
  PetscCall(TSSolve(ts, sol));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CONVERGED REASON: %" PetscInt_FMT " (TS_CONVERGED_EVENT == %" PetscInt_FMT ")\n", (PetscInt)reason, (PetscInt)TS_CONVERGED_EVENT));

  // The 3 columns printed are: [RANK] [num. of events at the given time] [time of event]
  for (PetscInt j = 0; j < ctx.cnt; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%" PetscInt_FMT "\t%g\n", ctx.rank, ctx.evnum[j], (double)ctx.evres[j]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // print the final time step
  PetscCall(TSGetTimeStep(ts, &dtlast));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &match));
  if (match) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Adapt = none\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Last dt = %g\n", (double)dtlast));
  }

  PetscCall(MatDestroy(&A));
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
  if (Ctx->rank == 1 % Ctx->size) { gval[n++] = PetscSinReal(Ctx->pi * t); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  User callback for the post-event stuff
*/
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx)
{
  AppCtx *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (Ctx->flg) {
    PetscCallBack("EventFunction", EventFunction(ts, t, U, Ctx->fvals, ctx));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] At t = %20.16g : %" PetscInt_FMT " events triggered, fvalues =", Ctx->rank, (double)t, nev_zero));
    for (PetscInt j = 0; j < nev_zero; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\t%g", (double)Ctx->fvals[evs_zero[j]]));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }

  if (Ctx->cnt < MAX_NEV && nev_zero > 0) {
    Ctx->evres[Ctx->cnt]   = t;
    Ctx->evnum[Ctx->cnt++] = nev_zero;
  }

#ifdef NEW_VERSION
  Ctx->postcnt++; // sync
  if (Ctx->dtpost > 0) {
    if (Ctx->postcnt % 2 == 0) PetscCall(TSSetPostEventStep(ts, Ctx->dtpost));
    else PetscCall(TSSetPostEventStep(ts, 0));
  }
#endif

  if (Ctx->restart) PetscCall(TSRestartStep(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*---------------------------------------------------------------------------------------------*/
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
    suffix: fin8
    output_file: output/ex3span_fin8.out
    args: -ts_max_time 8
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1127
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin81
    output_file: output/ex3span_fin8.1.out
    args: -ts_max_time 8.1
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1125
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin821
    output_file: output/ex3span_fin8.21.out
    args: -ts_max_time 8.21
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin899
    output_file: output/ex3span_fin8.99.out
    args: -ts_max_time 8.99
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1127
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 4

  test:
    suffix: fin9
    output_file: output/ex3span_fin9.out
    args: -ts_max_time 9
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.202
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin901
    output_file: output/ex3span_fin9.01.out
    args: -ts_max_time 9.01
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1127
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.6
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin904
    output_file: output/ex3span_fin9.04.out
    args: -ts_max_time 9.04
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.102
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin905
    output_file: output/ex3span_fin9.05.out
    args: -ts_max_time 9.05
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1127
    args: -D {{0.0015 0.03}}
    args: -dir {{0 1 -1}}
    args: -ts_dt 0.402
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin905etc
    output_file: output/ex3span_fin9.05.out
    args: -ts_max_time {{9.06 9.07 9.1 9.21 9.5 9.99 10 11}}
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.11
    args: -D 0.0025
    args: -dir {{0 -1}}
    args: -ts_dt 0.3025
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin906
    output_file: output/ex3span_fin9.06.out
    args: -ts_max_time 9.06
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.1121
    args: -D {{0.0015 0.02}}
    args: -dir 1
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin91
    output_file: output/ex3span_fin9.1.out
    args: -ts_max_time 9.1
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.09
    args: -D {{0.0015 0.03}}
    args: -dir 1
    args: -ts_dt 0.302
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin921
    output_file: output/ex3span_fin9.21.out
    args: -ts_max_time 9.21
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.7
    args: -D {{0.0015 0.02}}
    args: -dir 1
    args: -ts_dt 0.22
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 1

  test:
    suffix: fin95
    output_file: output/ex3span_fin9.5.out
    args: -ts_max_time 9.5
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -D {{0.0015 0.0135}}
    args: -dir 1
    args: -ts_dt 0.502
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 4

  test:
    suffix: fin10
    output_file: output/ex3span_fin10.out
    args: -ts_max_time 10
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.17
    args: -D {{0.0015 0.02}}
    args: -dir 1
    args: -ts_dt 0.422
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 2

  test:
    suffix: fin11
    output_file: output/ex3span_fin11.out
    args: -ts_max_time 11
    args: -ts_monitor -ts_event_dt_min 1e-6
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -D {{0.0015 0.0016}}
    args: -dir 1
    args: -ts_dt 0.2718281828
    args: -ts_type {{beuler rk alpha rosw bdf}}
    filter: grep "TS dt" | tail -n 1 | cut -f6 -d " "
    nsize: 4

TEST*/
