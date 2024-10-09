#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the older PETSc versions (without TSSetPostEventStep())

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "Using one or several event functions (on rank-0)\n"
                     "This program is mostly intended to test the Anderson-Bjorck iteration with challenging event-functions\n"
                     "Options:\n"
                     "-dir    d : zero-crossing direction for events\n"
                     "-flg      : additional output in Postevent\n"
                     "-errtol e : error tolerance, for printing 'pass/fail' for located events (1e-5 by default)\n"
                     "-restart  : flag for TSRestartStep() in PostEvent\n"
                     "-dtpost x : if x > 0, then on even PostEvent calls 1st-post-event-step = x is set,\n"
                     "                            on odd PostEvent calls 1st-post-event-step = PETSC_DECIDE is set,\n"
                     "            if x == 0, nothing happens\n"
                     "-func   F : selects the event function [0, ..., 11], if F == -1 (default) is set, all event functions are taken\n";

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
  PetscBool    pass = PETSC_TRUE;
  PetscScalar *x, vals[4];
  AppCtx       ctx;

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
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));   // error tolerance for located events
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

    // set the reference values
    if (ctx.F == 0 || ctx.F == -1) {
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 2.0;
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 4.0;
      if (dir0 <= 0) ctx.ref[ctx.cntref++] = 1.0;
      if (dir0 <= 0) ctx.ref[ctx.cntref++] = 3.0;
    }
    if (ctx.F == 1 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.0;
    if (ctx.F == 2 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.0;
    if (ctx.F == 3 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.696812386809752;
    if (ctx.F == 4 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 0.1;
    if (ctx.F == 5 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.02;
    if (ctx.F == 6 || ctx.F == -1)
      if (dir0 <= 0) ctx.ref[ctx.cntref++] = 0.90478821787302;
    if (ctx.F == 7 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 0.05;
    if (ctx.F == 8 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 0.552704666678489;
    if (ctx.F == 9 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 0.399422291710969;
    if (ctx.F == 10 || ctx.F == -1) {
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 0.874511220376091;
      if (dir0 <= 0) ctx.ref[ctx.cntref++] = 2.388359335869107;
    }
    if (ctx.F == 11 || ctx.F == -1)
      if (dir0 >= 0) ctx.ref[ctx.cntref++] = 2.5;
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));

  // Solution
  PetscCall(TSSolve(ts, sol));

  // The 4 columns printed are: [RANK] [time of event] [error w.r.t. reference] ["pass"/"fail"]
  for (PetscInt j = 0; j < ctx.cnt; j++) {
    PetscReal err = 10.0;
    if (j < ctx.cntref) err = PetscAbsReal(ctx.evres[j] - ctx.ref[j]);
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%g\t%g\t%s\n", ctx.rank, (double)ctx.evres[j], (double)err, err < ctx.errtol ? "pass" : "fail"));
    pass = (pass && err < ctx.errtol ? PETSC_TRUE : PETSC_FALSE);
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "This test: %s\n", pass ? "PASSED" : "FAILED"));

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
    if (Ctx->F == 8 || Ctx->F == -1) gval[n++] = ((t - 1) * PetscExpReal(-20 * t) + PetscPowReal(t, 20)) * 1e4;                                  // FUNC-16, root 0.5527
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

  if (Ctx->cnt + nev_zero < MAX_NEV)
    for (PetscInt i = 0; i < nev_zero; i++) Ctx->evres[Ctx->cnt++] = t; // save the repeating zeros separately for easier/unified testing

#ifdef NEW_VERSION
  Ctx->postcnt++; // sync
  if (Ctx->dtpost > 0) {
    if (Ctx->postcnt % 2 == 0) PetscCall(TSSetPostEventStep(ts, Ctx->dtpost));
    else PetscCall(TSSetPostEventStep(ts, PETSC_DECIDE));
  }
#endif

  if (Ctx->restart) PetscCall(TSRestartStep(ts));
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
    suffix: 0
    requires: !single
    output_file: output/ex4_0.out
    args: -dir 0
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-10
    args: -ts_dt 0.25
    args: -restart 0
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -errtol 1e-7
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0
    args: -ts_event_post_event_step -1
    args: -ts_type rk
    nsize: 2
    filter: sort
    filter_output: sort

  test:
    suffix: 0single
    requires: single
    output_file: output/ex4_0single.out
    args: -dir 0
    args: -ts_adapt_dt_min 1e-6 -ts_event_dt_min 1e-6
    args: -ts_dt 0.3
    args: -ts_event_tol {{1e-7 1e-10}}
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.23
    args: -ts_event_post_event_step -1
    args: -ts_type beuler
    nsize: 3
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
    args: -ts_type rk
    nsize: 1

  test:
    suffix: F7revisit
    output_file: output/ex4_F7revisit.out
    args: -ts_event_monitor -F 7 -ts_dt 0.04 -ts_event_dt_min 0.016 -errtol 0.005
    nsize: 1

  test:
    suffix: 2all
    output_file: output/ex4_2.out
    args: -dir 0
    args: -F {{-1 0 1 2 3 4 5 6 7 8 9 10 11}}
    args: -ts_event_dt_min 1e-6 -ts_dt 0.4 -ts_event_tol 1e-8
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.35
    args: -ts_type rk
    filter: grep "This test"
    nsize: 1

  test:
    suffix: 2pos
    output_file: output/ex4_2.out
    args: -dir 1
    args: -F {{-1 0 1 2 3 4 5 7 8 9 10 11}}
    args: -ts_event_dt_min 1e-6 -ts_dt 0.4 -ts_event_tol 1e-8
    args: -ts_adapt_type none
    args: -dtpost 0.34
    args: -ts_type beuler
    filter: grep "This test"
    nsize: 1

  test:
    suffix: 2neg
    output_file: output/ex4_2.out
    args: -dir -1
    args: -F {{-1 0 6 10}}
    args: -ts_event_dt_min 1e-6 -ts_dt 0.4 -ts_event_tol 1e-8
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.33
    args: -ts_type rk
    filter: grep "This test"
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
    args: -dtpost 0.25
    args: -ts_event_post_event_step -1
    args: -ts_type {{beuler rk}}
    nsize: 1

  test:
    suffix: neg
    output_file: output/ex4_neg.out
    args: -dir -1
    args: -ts_adapt_dt_min 1e-10 -ts_event_dt_min 1e-6
    args: -ts_dt 0.4
    args: -restart 1
    args: -ts_event_tol {{1e-8 1e-15}}
    args: -ts_adapt_type {{none basic}}
    args: -dtpost 0.25
    args: -ts_event_post_event_step 0.35
    args: -ts_type rk
    nsize: 2
    filter: sort
    filter_output: sort
TEST*/
