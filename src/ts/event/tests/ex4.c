#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the old (current) TSEvent code

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "Using one or several event functions (on rank-0)\n"
                     "This program is mostly intended to test the Anderson-Bjorck iteration\n"
                     "Options:\n"
                     "-dir    d : zero-crossing direction for events\n"
                     "-flg      : additional output in Postevent\n"
                     "-restart  : flag for TSRestartStep() in PostEvent\n"
                     "-dtpost x : if x > 0, then on even PostEvent calls dt_postevent = x is set, on odd PostEvent calls dt_postevent = 0 is set,\n"
                     "            if x == 0, nothing happens\n"
                     "-func   F : selects the event function [0, ..., 11], if F == -1 (default) is set, all event functions are taken\n";

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
  PetscReal   dtpost;           // post-event step
  PetscInt    postcnt;          // counter for PostEvent calls
  PetscInt    F;                // event-function index
  PetscInt    Fnum;             // total available event functions
} AppCtx;

PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);

int main(int argc, char **argv)
{
  TS           ts;
  Mat          A;
  Vec          sol;
  PetscInt     n, dir0, m = 0;
  PetscInt     dir[MAX_NFUNC], inds[2];
  PetscBool    term[MAX_NFUNC];
  PetscScalar *x, vals[4];
  AppCtx       ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  setbuf(stdout, NULL);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  ctx.pi      = PetscAcosReal(-1.0);
  ctx.cnt     = 0;
  ctx.flg     = PETSC_FALSE;
  ctx.restart = PETSC_FALSE;
  ctx.dtpost  = 0;
  ctx.postcnt = 0;
  ctx.F       = -1;
  ctx.Fnum    = 12;

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

  PetscCall(TSSetTime(ts, 0.03));
  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 4.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  // Set the event handling
  dir0 = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));             // desired zero-crossing direction
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));               // flag for additional output
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-restart", &ctx.restart, NULL)); // flag for TSRestartStep()
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dtpost", &ctx.dtpost, NULL));   // post-event step
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-F", &ctx.F, NULL));              // event-function index
  PetscCheck(ctx.F >= -1 && ctx.F < ctx.Fnum, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Value of 'F' is out of range");

  n = 0;               // event counter
  if (ctx.rank == 0) { // all events -- on rank-0
    if (ctx.F == -1)
      for (n = 0; n < ctx.Fnum; n++) { // all event-functions
        dir[n]  = dir0;
        term[n] = PETSC_FALSE;
      }
    else { // single event-function
      dir[n]    = dir0;
      term[n++] = PETSC_FALSE;
    }
  }
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));

  // Solution
  PetscCall(TSSolve(ts, sol));

  // The 3 columns printed are: [RANK] [num. of events at the given time] [time of event]
  for (PetscInt j = 0; j < ctx.cnt; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%" PetscInt_FMT "\t%.5g\n", ctx.rank, ctx.evnum[j], (double)ctx.evres[j]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

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
  // all events -- on rank-0
  if (Ctx->rank == 0) {
    if (Ctx->F == 0 || Ctx->F == -1) gval[n++] = PetscSinReal(Ctx->pi * t) / Ctx->pi; // FUNC-0, roots 1, 2, 3, 4
    if (Ctx->F == 1 || Ctx->F == -1) gval[n++] = PetscLogReal(t);                     // FUNC-2, root 1
    if (Ctx->F == 2 || Ctx->F == -1) {                                                // FUNC-3, root 1
      if (t < 2) gval[n++] = (1 - PetscPowReal(t - 2, 12)) / 12.0;
      else gval[n++] = 1 / 12.0;
    }
    if (Ctx->F == 3 || Ctx->F == -1) gval[n++] = t - PetscExpReal(PetscSinReal(t)) + 1;                                                          // FUNC-5, root 1.69681
    if (Ctx->F == 4 || Ctx->F == -1) gval[n++] = (1e10 * PetscPowReal(t, 1 / t) - 1) / 100;                                                      // FUNC-6, root 0.1
    if (Ctx->F == 5 || Ctx->F == -1) gval[n++] = PetscLogReal(t - 0.02) * PetscLogReal(t - 0.02) * PetscSignReal(t - 1.02) * 1e7;                // FUNC-7, root 1.02
    if (Ctx->F == 6 || Ctx->F == -1) gval[n++] = 4 * PetscCosReal(t) - PetscExpReal(t);                                                          // FUNC-14, root 0.904788
    if (Ctx->F == 7 || Ctx->F == -1) gval[n++] = (20.0 * t - 1) / (19.0 * t) / 10;                                                               // FUNC-15, root 0.05
    if (Ctx->F == 8 || Ctx->F == -1) gval[n++] = ((t - 1) * PetscExpReal(-20 * t) + PetscPowReal(t, 20)) * 1e4;                                  // FUNC-16, root 0.552
    if (Ctx->F == 9 || Ctx->F == -1) gval[n++] = (t * t * (t * t / 3.0 + PetscSqrtReal(2.0) * PetscSinReal(t)) - PetscSqrtReal(3.0) / 18) * 10;  // FUNC-17, root 0.399
    if (Ctx->F == 10 || Ctx->F == -1) gval[n++] = ((t * t + 1) * PetscSinReal(t) - PetscExpReal(PetscSqrtReal(t)) * (t - 1) * (t * t - 5)) / 10; // FUNC-18, roots 0.87, 2.388
    if (Ctx->F == 11 || Ctx->F == -1) gval[n++] = 2 * t - 5;                                                                                     // FUNC-21, root 2.5
  }
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

  if (Ctx->cnt + nev_zero < MAX_NEV) {
    for (PetscInt i = 0; i < nev_zero; i++) { // save the repeating zeros separately for easier/unified testing
      Ctx->evres[Ctx->cnt]   = t;
      Ctx->evnum[Ctx->cnt++] = 1;
    }
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
    suffix: 0
    output_file: output/ex4_0.out
    args: -dir 0
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-6
    args: -ts_dt 0.4
    args: -restart 1
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -ts_adapt_type {{none basic}}
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step {{0 0.35}}
    args: -ts_type {{beuler rk}}
    nsize: {{1 4}}
    filter: sort
    filter_output: sort

  test:
    suffix: F7
    output_file: output/ex4_F7.out
    args: -dir 0
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-6
    args: -ts_dt 0.4
    args: -F 7
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -ts_adapt_type {{none basic}}
    args: -ts_type {{beuler rk}}
    nsize: 1

  test:
    suffix: F7revisit
    output_file: output/ex4_F7revisit.out
    args: -ts_event_monitor -F 7 -ts_dt 0.04 -ts_event_dt_min 0.016
    nsize: 1

  test:
    suffix: pos
    output_file: output/ex4_pos.out
    args: -dir 1
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-6
    args: -ts_dt 0.4
    args: -restart 0
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -ts_adapt_type {{none basic}}
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step {{0 0.35}}
    args: -ts_type {{beuler rk}}
    nsize: {{1 4}}
    filter: sort
    filter_output: sort

  test:
    suffix: neg
    output_file: output/ex4_neg.out
    args: -dir -1
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-6
    args: -ts_dt 0.4
    args: -restart 1
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -ts_adapt_type {{none basic}}
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step {{0 0.35}}
    args: -ts_type {{beuler rk}}
    nsize: {{1 4}}
    filter: sort
    filter_output: sort
TEST*/
