#include <petscts.h>
#include <stdio.h>

#define NEW_VERSION // Applicable for the new features; avoid this for the older PETSc versions (without TSSetPostEventStep())

static char help[] = "Simple linear problem with events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "Using 16 event functions:\n"
                     "7 polynomials (dir=+1) with zeros: 1+2^i,     i=-3,...3, on ranks=(i+3)%size\n"
                     "7 polynomials (dir=-1) with zeros: 1+(8-2^i), i=-3,...3, on ranks=(i+3)%size\n"
                     "(t-5)^2 * sin(pi*t), with zeros = 1,2,...10,      on rank-0\n"
                     "    0.5 * cos(pi*t), with zeros = 0.5,1.5,...9.5, on last-rank\n"
                     "Options:\n"
                     "-dir    d : zero-crossing direction for events\n"
                     "-flg      : additional output in Postevent\n"
                     "-errtol e : error tolerance, for printing 'pass/fail' for located events (1e-5 by default)\n"
                     "-restart  : flag for TSRestartStep() in PostEvent\n"
                     "-dtpost x : if x > 0, then on even PostEvent calls 1st-post-event-step = x is set,\n"
                     "                            on odd PostEvent calls 1st-post-event-step = PETSC_DECIDE is set,\n"
                     "            if x == 0, nothing happens\n";

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
  PetscReal   vtol[MAX_NFUNC];  // vtol array, with extra storage
  PetscInt    dir0;             // desired zero-crossing direction
} AppCtx;

PetscErrorCode     EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode     Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);
static inline void SetVtols(PetscMPIInt rank, PetscMPIInt size, PetscReal tol0, PetscReal tolsin, PetscReal *vtol); // helper function to fill vtol[]

int main(int argc, char **argv)
{
  TS           ts;
  Mat          A;
  Vec          sol;
  PetscInt     n, m = 0;
  PetscInt     dir[MAX_NFUNC], inds[2];
  PetscBool    term[MAX_NFUNC];
  PetscScalar *x, vals[4];
  PetscReal    aux;
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
  ctx.dir0 = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &ctx.dir0, NULL));         // desired zero-crossing direction
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));               // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));   // error tolerance for located events
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-restart", &ctx.restart, NULL)); // flag for TSRestartStep()
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dtpost", &ctx.dtpost, NULL));   // post-event step

  n   = 0; // event counter
  aux = 1.0 / 8.0;
  for (PetscInt i = -3; i <= 3; i++) { // pos-polynomials
    if (ctx.rank == (i + 3) % ctx.size) {
      dir[n]    = ctx.dir0;
      term[n++] = PETSC_FALSE;
      if (ctx.dir0 >= 0) ctx.ref[ctx.cntref++] = 1.0 + aux;
    }
    aux *= 2;
  }
  aux = 1.0 / 8.0;
  for (PetscInt i = -3; i <= 3; i++) { // neg-polynomials
    if (ctx.rank == (i + 3) % ctx.size) {
      dir[n]    = ctx.dir0;
      term[n++] = PETSC_FALSE;
      if (ctx.dir0 <= 0) ctx.ref[ctx.cntref++] = 9.0 - aux;
    }
    aux *= 2;
  }
  if (ctx.rank == 0) { // sin-event -- on rank-0
    dir[n]    = ctx.dir0;
    term[n++] = PETSC_FALSE;
    for (PetscInt i = 1; i < MAX_NEV / 2 - 10; i++) {
      if (i % 2 == 1 && ctx.dir0 <= 0) ctx.ref[ctx.cntref++] = i;
      if (i % 2 == 0 && ctx.dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.rank == ctx.size - 1) { // cos-event -- on last rank
    dir[n]    = ctx.dir0;
    term[n++] = PETSC_FALSE;
    for (PetscInt i = 1; i < MAX_NEV / 2 - 10; i++) {
      if (i % 2 == 1 && ctx.dir0 <= 0) ctx.ref[ctx.cntref++] = i - 0.5;
      if (i % 2 == 0 && ctx.dir0 >= 0) ctx.ref[ctx.cntref++] = i - 0.5;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));
  SetVtols(ctx.rank, ctx.size, 1e-8, 1e-8, ctx.vtol);
  PetscCall(TSSetEventTolerances(ts, PETSC_DECIDE, ctx.vtol));

  // Solution
  PetscCall(TSSolve(ts, sol));

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
  PetscInt  n   = 0;
  AppCtx   *Ctx = (AppCtx *)ctx;
  PetscReal P;

  PetscFunctionBeginUser;
  // for the test purposes, event-functions are defined based on t
  for (PetscInt i = -3; i <= 3; i++) { // pos-polynomials
    if (Ctx->rank == (i + 3) % Ctx->size) {
      P = PetscPowReal(2.0, i);
      if (t < 2 + P) gval[n++] = 1 - PetscPowReal(2 + P - t, i + 5);
      else gval[n++] = 1;
    }
  }
  for (PetscInt i = -3; i <= 3; i++) { // neg-polynomials
    if (Ctx->rank == (i + 3) % Ctx->size) {
      P = PetscPowReal(2.0, i);
      if (t > 8 - P) gval[n++] = 1 - PetscPowReal(t - 8 + P, i + 5);
      else gval[n++] = 1;
    }
  }
  if (Ctx->rank == 0) gval[n++] = (t - 5) * (t - 5) * PetscSinReal(Ctx->pi * t); // sin-event -- on rank-0
  if (Ctx->rank == Ctx->size - 1) gval[n++] = 0.5 * PetscCosReal(Ctx->pi * t);   // cos-event -- on last rank
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

  if ((Ctx->dir0 == 0 && PetscAbsReal(t - (PetscReal)4.0) < 0.01) || (Ctx->dir0 == -1 && PetscAbsReal(t - (PetscReal)3.0) < 0.01)) {
    SetVtols(Ctx->rank, Ctx->size, 1e-8, 1e-26, Ctx->vtol); // for better resolution of sin-event at t=5.0
    PetscCall(TSSetEventTolerances(ts, PETSC_DECIDE, Ctx->vtol));
  }
  if (PetscAbsReal(t - (PetscReal)5.0) < 0.01) {
    SetVtols(Ctx->rank, Ctx->size, 1e-8, 1e-8, Ctx->vtol); // back to normal
    PetscCall(TSSetEventTolerances(ts, PETSC_DECIDE, Ctx->vtol));
  }

  if (Ctx->restart) PetscCall(TSRestartStep(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// helper function to fill vtol[]
static inline void SetVtols(PetscMPIInt rank, PetscMPIInt size, PetscReal tol0, PetscReal tolsin, PetscReal *vtol)
{
  PetscInt n = 0;
  for (PetscInt i = -3; i <= 3; i++)
    if (rank == (i + 3) % size) vtol[n++] = tol0; // pos-polynomials
  for (PetscInt i = -3; i <= 3; i++)
    if (rank == (i + 3) % size) vtol[n++] = tol0; // neg-polynomials
  if (rank == 0) vtol[n++] = tolsin;              // sin-event -- on rank-0
  if (rank == size - 1) vtol[n++] = tol0;         // cos-event -- on last rank
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
    suffix: pos1
    output_file: output/ex5_pos1.out
    args: -dir 1 -ts_event_dt_min 1e-6
    args: -restart 1
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step 0.31
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 1

  test:
    suffix: pos4
    output_file: output/ex5_pos4.out
    args: -dir 1 -ts_event_dt_min 1e-6 -ts_dt 0.25
    args: -restart 0
    args: -dtpost 0
    args: -ts_event_post_event_step -1
    args: -ts_type {{beuler rk}}
    args: -ts_adapt_type {{none basic}}
    nsize: 4
    filter: sort
    filter_output: sort

  test:
    suffix: neu1
    output_file: output/ex5_neu1.out
    args: -dir 0 -ts_event_dt_min 1e-6
    args: -restart 0
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step -1
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 1

  test:
    suffix: neu4
    output_file: output/ex5_neu4.out
    args: -dir 0 -ts_event_dt_min 1e-6 -ts_dt 0.25
    args: -dtpost 0
    args: -ts_event_post_event_step {{-1 0.29}}
    args: -ts_event_post_event_second_step {{-1 0.31}}
    args: -ts_type rk
    args: -ts_adapt_type {{none basic}}
    nsize: 4
    filter: sort
    filter_output: sort

  test:
    suffix: neg2
    output_file: output/ex5_neg2.out
    args: -dir -1 -ts_event_dt_min 1e-6
    args: -restart 1
    args: -dtpost {{0 0.25}}
    args: -ts_event_post_event_step 0.31
    args: -ts_type beuler
    args: -ts_adapt_type {{none basic}}
    nsize: 2
    filter: sort
    filter_output: sort

  test:
    suffix: neg4
    output_file: output/ex5_neg4.out
    args: -dir -1 -ts_event_dt_min 1e-6 -ts_dt 0.25
    args: -restart 0
    args: -dtpost 0
    args: -ts_event_post_event_step -1
    args: -ts_type {{beuler rk}}
    args: -ts_adapt_type {{none basic}}
    nsize: 4
    filter: sort
    filter_output: sort

TEST*/
