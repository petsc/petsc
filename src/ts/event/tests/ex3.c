#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the older PETSc versions (without TSSetPostEventStep())

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank (with zeros: 1.05, 9.05[terminating])\n"
                     "- one event function on rank = '1%size', equal to V*sin(pi*t), zeros = 1,...,10\n"
                     "After each event location the tolerance for the sin() event is multiplied by 4\n"

                     "Options:\n"
                     "-dir    d : zero-crossing direction for events: 0, 1, -1\n"
                     "-flg      : additional output in Postevent\n"
                     "-errtol e : error tolerance, for printing 'pass/fail' for located events (1e-5 by default)\n"
                     "-restart  : flag for TSRestartStep() in PostEvent\n"
                     "-dtpost x : if x > 0, then on even PostEvent calls 1st-post-event-step = x is set,\n"
                     "                            on odd PostEvent calls 1st-post-event-step = PETSC_DECIDE is set,\n"
                     "            if x == 0, nothing happens\n"
                     "-v {float}: scaling of the sin() event function; for small v this event is triggered by the function values,\n"
                     "            for large v the event is triggered by the small step size\n"
                     "-change5  : flag to change the state vector at t=5 PostEvent\n";

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
  PetscReal   V;                // vertical scaling for sin()
  PetscReal   vtol[MAX_NFUNC];  // vtol array, with extra storage
  PetscBool   change5;          // flag to change the state vector at t=5 PostEvent
} AppCtx;

PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);

int main(int argc, char **argv)
{
  TS                ts;
  Mat               A;
  Vec               sol;
  PetscInt          n, dir0, m = 0;
  PetscReal         tol = 1e-7;
  PetscInt          dir[MAX_NFUNC], inds[2];
  PetscBool         term[MAX_NFUNC];
  PetscScalar      *x, vals[4];
  AppCtx            ctx;
  TSConvergedReason reason;

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
  ctx.V       = 1.0;
  ctx.change5 = PETSC_FALSE;

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

  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  // Set the event handling
  dir0 = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));             // desired zero-crossing direction
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));               // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));   // error tolerance for located events
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-restart", &ctx.restart, NULL)); // flag for TSRestartStep()
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dtpost", &ctx.dtpost, NULL));   // post-event step
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &ctx.V, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-change5", &ctx.change5, NULL)); // flag to change the state vector at t=5 PostEvent

  n = 0;               // event counter
  if (ctx.rank == 0) { // first event -- on rank-0
    ctx.vtol[n] = tol * 10;
    dir[n]      = dir0;
    term[n++]   = PETSC_FALSE;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.05;
  }
  if (ctx.rank == ctx.size - 1) { // second event (with termination) -- on last rank
    ctx.vtol[n] = tol * 10;
    dir[n]      = dir0;
    term[n++]   = PETSC_TRUE;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 9.05;
  }
  if (ctx.rank == 1 % ctx.size) { // third event -- on rank = 1%ctx.size
    ctx.vtol[n] = tol;
    dir[n]      = dir0;
    term[n++]   = PETSC_FALSE;

    for (PetscInt i = 1; i < MAX_NEV - 2; i++) {
      if (i % 2 == 1 && dir0 <= 0) ctx.ref[ctx.cntref++] = i;
      if (i % 2 == 0 && dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));
  PetscCall(TSSetEventTolerances(ts, tol, ctx.vtol));

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
  if (Ctx->rank == 1 % Ctx->size) gval[n++] = Ctx->V * PetscSinReal(Ctx->pi * t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  User callback for the post-event stuff
*/
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx)
{
  PetscInt     n = 0;
  PetscScalar *x;
  AppCtx      *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (Ctx->flg) {
    PetscCallBack("EventFunction", EventFunction(ts, t, U, Ctx->fvals, ctx));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] At t = %20.16g : %" PetscInt_FMT " events triggered, fvalues =", Ctx->rank, (double)t, nev_zero));
    for (PetscInt j = 0; j < nev_zero; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\t%g", (double)Ctx->fvals[evs_zero[j]]));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }

  // change the state vector near t=5.0
  if (PetscAbsReal(t - (PetscReal)5.0) < 0.01 && Ctx->change5) {
    PetscCall(VecGetArray(U, &x));
    if (Ctx->rank == 0) x[1] = -x[1];
    PetscCall(VecRestoreArray(U, &x));
  }

  // update vtol's
  if (Ctx->rank == 0) n++;             // first event -- on rank-0
  if (Ctx->rank == Ctx->size - 1) n++; // second event -- on last rank
  if (Ctx->rank == 1 % Ctx->size) {    // third event -- on rank = 1%ctx.size
    if (Ctx->flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "vtol for sin: %g -> ", (double)Ctx->vtol[n]));
    Ctx->vtol[n] *= 4;
    if (PetscAbsReal(t - (PetscReal)5.0) < 0.01) Ctx->vtol[n] /= 100; // one-off decrease
    if (Ctx->flg) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g\n", (double)Ctx->vtol[n]));
    n++;
  }
  PetscCall(TSSetEventTolerances(ts, 0, Ctx->vtol));

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
    suffix: V
    output_file: output/ex3_V.out
    args: -ts_type beuler
    args: -ts_adapt_type basic
    args: -v {{1e2 1e5 1e8}}
    args: -ts_adapt_dt_min 1e-6
    args: -change5 {{0 1}}
    nsize: 1

  test:
    suffix: neu1
    output_file: output/ex3_neu1.out
    args: -dir 0
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 1
    args: -dtpost 0.24
    args: -ts_event_post_event_step 0.31
    args: -ts_type {{beuler rk}}
    args: -ts_adapt_type {{none basic}}
    nsize: 1

  test:
    suffix: neu2
    output_file: output/ex3_neu2.out
    args: -dir 0
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 1
    args: -dtpost 0
    args: -ts_event_post_event_step {{-1 0.31}}
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 2
    filter: sort
    filter_output: sort

  test:
    suffix: neu4
    output_file: output/ex3_neu4.out
    args: -dir 0
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart {{0 1}}
    args: -dtpost 0.24
    args: -ts_event_post_event_step 0.21
    args: -ts_type beuler
    args: -ts_adapt_type {{none basic}}
    nsize: 4
    filter: sort
    filter_output: sort

  test:
    suffix: pos1
    output_file: output/ex3_pos1.out
    args: -dir 1
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 0
    args: -dtpost 0.24
    args: -ts_type {{beuler rk}}
    args: -ts_adapt_type {{none basic}}
    nsize: 1

  test:
    suffix: pos2
    output_file: output/ex3_pos2.out
    args: -dir 1
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 1
    args: -dtpost {{0 0.24}}
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 2
    filter: sort
    filter_output: sort

  test:
    suffix: pos4
    output_file: output/ex3_pos4.out
    args: -dir 1
    args: -v 1e9
    args: -ts_adapt_dt_min 1e-6
    args: -restart 0
    args: -dtpost 0
    args: -ts_event_post_event_step {{-1 0.32}}
    args: -ts_type beuler
    args: -ts_adapt_type {{none basic}}
    args: -change5 1
    nsize: 4
    filter: sort
    filter_output: sort

  test:
    suffix: neg1
    output_file: output/ex3_neg1.out
    args: -dir -1
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 1
    args: -dtpost {{0 0.24}}
    args: -ts_type {{beuler rk}}
    args: -ts_adapt_type basic
    nsize: 1

  test:
    suffix: neg2
    output_file: output/ex3_neg2.out
    args: -dir -1
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 0
    args: -dtpost {{0 0.24}}
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 2
    filter: sort
    filter_output: sort

  test:
    suffix: neg4
    output_file: output/ex3_neg4.out
    args: -dir -1
    args: -v 1e5
    args: -ts_adapt_dt_min 1e-6
    args: -restart 0
    args: -dtpost {{0 0.24}}
    args: -ts_event_post_event_step 0.3
    args: -ts_type beuler
    args: -ts_adapt_type {{none basic}}
    nsize: 4
    filter: sort
    filter_output: sort
TEST*/
