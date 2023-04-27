static char help[] = "Landau collision operator with amnisotropic thermalization verification test as per Hager et al.\n 'A fully non-linear multi-species Fokker-Planck-Landau collision operator for simulation of fusion plasma', and "
                     "published as 'A performance portable, fully implicit Landau collision operator with batched linear solvers' https://arxiv.org/abs/2209.03228\n\n";

#include <petscts.h>
#include <petsclandau.h>
#include <petscdmcomposite.h>
#include <petscds.h>

/*
 call back method for DMPlexLandauAccess:

Input Parameters:
 .   dm - a DM for this field
 -   local_field - the local index in the grid for this field
 .   grid - the grid index
 +   b_id - the batch index
 -   vctx - a user context

 Input/Output Parameter:
 .   x - Vector to data to

 */
PetscErrorCode landau_field_print_access_callback(DM dm, Vec x, PetscInt local_field, PetscInt grid, PetscInt b_id, void *vctx)
{
  LandauCtx  *ctx;
  PetscScalar val;
  PetscInt    species;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  species = ctx->species_offset[grid] + local_field;
  val     = (PetscScalar)(LAND_PACK_IDX(b_id, grid) + (species + 1) * 10);
  PetscCall(VecSet(x, val));
  PetscCall(PetscInfo(dm, "DMPlexLandauAccess user 'add' method to grid %" PetscInt_FMT ", batch %" PetscInt_FMT " and local field %" PetscInt_FMT " with %" PetscInt_FMT " grids\n", grid, b_id, local_field, ctx->num_grids));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static const PetscReal alphai   = 1 / 1.3;
static const PetscReal kev_joul = 6.241506479963235e+15; /* 1/1000e */

// constants: [index of (anisotropic) direction of source, z x[1] shift
/* < v, n_s v_|| > */
static void f0_vz(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  if (dim == 2) f0[0] = u[0] * (2. * PETSC_PI * x[0]) * x[1]; /* n r v_|| */
  else f0[0] = u[0] * x[2];
}
/* < v, n (v-shift)^2 > */
static void f0_v2_par_shift(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal vz = PetscRealPart(constants[0]);
  if (dim == 2) *f0 = u[0] * (2. * PETSC_PI * x[0]) * (x[1] - vz) * (x[1] - vz); /* n r v^2_par|perp */
  else *f0 = u[0] * (x[2] - vz) * (x[2] - vz);
}
static void f0_v2_perp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  if (dim == 2) *f0 = u[0] * (2. * PETSC_PI * x[0]) * x[0] * x[0]; /* n r v^2_perp */
  else *f0 = u[0] * (x[0] * x[0] + x[1] * x[1]);
}
/* < v, n_e > */
static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  if (dim == 2) f0[0] = 2. * PETSC_PI * x[0] * u[0];
  else f0[0] = u[0];
}
static void f0_v2_shift(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal vz = PetscRealPart(constants[0]);
  if (dim == 2) f0[0] = u[0] * (2. * PETSC_PI * x[0]) * (x[0] * x[0] + (x[1] - vz) * (x[1] - vz));
  else f0[0] = u[0] * (x[0] * x[0] + x[1] * x[1] + (x[2] - vz) * (x[2] - vz));
}
/* Define a Maxwellian function for testing out the operator. */
typedef struct {
  PetscReal v_0;
  PetscReal kT_m;
  PetscReal n;
  PetscReal shift;
  PetscInt  species;
} MaxwellianCtx;

static PetscErrorCode maxwellian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  MaxwellianCtx *mctx  = (MaxwellianCtx *)actx;
  PetscReal      theta = 2 * mctx->kT_m / (mctx->v_0 * mctx->v_0); /* theta = 2kT/mc^2 */
  PetscFunctionBegin;
  /* evaluate the shifted Maxwellian */
  if (dim == 2) u[0] += alphai * mctx->n * PetscPowReal(PETSC_PI * theta, -1.5) * PetscExpReal(-(alphai * x[0] * x[0] + (x[1] - mctx->shift) * (x[1] - mctx->shift)) / theta);
  else u[0] += alphai * mctx->n * PetscPowReal(PETSC_PI * theta, -1.5) * PetscExpReal(-(alphai * (x[0] * x[0] + x[1] * x[1]) + (x[2] - mctx->shift) * (x[2] - mctx->shift)) / theta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], PetscInt grid, PetscReal shifts[], LandauCtx *ctx)
{
  PetscErrorCode (*initu[LANDAU_MAX_SPECIES])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  MaxwellianCtx *mctxs[LANDAU_MAX_SPECIES], data[LANDAU_MAX_SPECIES];
  PetscFunctionBegin;
  if (!ctx) PetscCall(DMGetApplicationContext(dm, &ctx));
  for (PetscInt ii = ctx->species_offset[grid], i0 = 0; ii < ctx->species_offset[grid + 1]; ii++, i0++) {
    mctxs[i0]        = &data[i0];
    data[i0].v_0     = ctx->v_0;                             // v_0 same for all grids
    data[i0].kT_m    = ctx->k * temps[ii] / ctx->masses[ii]; /* kT/m = v_th ^ 2*/
    data[i0].n       = ns[ii];
    initu[i0]        = maxwellian;
    data[i0].shift   = 0;
    data[i0].species = ii;
  }
  if (1) {
    data[0].shift = -((PetscReal)PetscSign(ctx->charges[ctx->species_offset[grid]])) * ctx->electronShift * ctx->m_0 / ctx->masses[ctx->species_offset[grid]];
  } else {
    shifts[0]     = 0.5 * PetscSqrtReal(ctx->masses[0] / ctx->masses[1]);
    shifts[1]     = 50 * (ctx->masses[0] / ctx->masses[1]);
    data[0].shift = ctx->electronShift * shifts[grid] * PetscSqrtReal(data[0].kT_m) / ctx->v_0; // shifts to not matter!!!!
  }
  PetscCall(DMProjectFunction(dm, time, initu, (void **)mctxs, INSERT_ALL_VALUES, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  E_PAR_IDX,
  E_PERP_IDX,
  I_PAR_IDX,
  I_PERP_IDX,
  NUM_TEMPS
} TemperatureIDX;

/* --------------------  Evaluate Function F(x) --------------------- */
static PetscReal n_cm3[2] = {0, 0};
PetscErrorCode   FormFunction(TS ts, PetscReal tdummy, Vec X, Vec F, void *ptr)
{
  LandauCtx         *ctx = (LandauCtx *)ptr; /* user-defined application context */
  PetscScalar       *f;
  const PetscScalar *x;
  const PetscReal    k_B = 1.6e-12, e_cgs = 4.8e-10, proton_mass = 9.1094e-28, m_cgs[2] = {proton_mass, proton_mass * ctx->masses[1] / ctx->masses[0]}; // erg/eV, e, m as per NRL;
  PetscReal          AA, v_bar_ab, vTe, t1, TeDiff, Te, Ti, Tdiff;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  Te = PetscRealPart(2 * x[E_PERP_IDX] + x[E_PAR_IDX]) / 3, Ti = PetscRealPart(2 * x[I_PERP_IDX] + x[I_PAR_IDX]) / 3;
  // thermalization from NRL Plasma formulary, assume Z = 1, mu = 2, n_i = n_e
  v_bar_ab = 1.8e-19 * PetscSqrtReal(m_cgs[0] * m_cgs[1]) * n_cm3[0] * ctx->lambdas[0][1] * PetscPowReal(m_cgs[0] * Ti + m_cgs[1] * Te, -1.5);
  PetscCall(VecGetArray(F, &f));
  for (PetscInt ii = 0; ii < 2; ii++) {
    PetscReal tPerp = PetscRealPart(x[2 * ii + E_PERP_IDX]), tPar = PetscRealPart(x[2 * ii + E_PAR_IDX]), ff;
    TeDiff = tPerp - tPar;
    AA     = tPerp / tPar - 1;
    if (AA < 0) ff = PetscAtanhReal(PetscSqrtReal(-AA)) / PetscSqrtReal(-AA);
    else ff = PetscAtanReal(PetscSqrtReal(AA)) / PetscSqrtReal(AA);
    t1 = (-3 + (AA + 3) * ff) / PetscSqr(AA);
    //PetscReal vTeB = 8.2e-7 * n_cm3[0] * ctx->lambdas[0][1] * PetscPowReal(Te, -1.5);
    vTe = 2 * PetscSqrtReal(PETSC_PI / m_cgs[ii]) * PetscSqr(PetscSqr(e_cgs)) * n_cm3[0] * ctx->lambdas[0][1] * PetscPowReal(PetscRealPart(k_B * x[E_PAR_IDX]), -1.5) * t1;
    t1  = vTe * TeDiff; // * 2; // scaling from NRL that makes it fit pretty good

    f[2 * ii + E_PAR_IDX]  = 2 * t1; // par
    f[2 * ii + E_PERP_IDX] = -t1;    // perp
    Tdiff                  = (ii == 0) ? (Ti - Te) : (Te - Ti);
    f[2 * ii + E_PAR_IDX] += v_bar_ab * Tdiff;
    f[2 * ii + E_PERP_IDX] += v_bar_ab * Tdiff;
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------  Form initial approximation ----------------- */
static PetscReal T0[4] = {300, 390, 200, 260};
PetscErrorCode   createVec_NRL(LandauCtx *ctx, Vec *vec)
{
  PetscScalar *x;
  Vec          Temps;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, NUM_TEMPS, &Temps));
  PetscCall(VecGetArray(Temps, &x));
  for (PetscInt i = 0; i < NUM_TEMPS; i++) x[i] = T0[i];
  PetscCall(VecRestoreArray(Temps, &x));
  *vec = Temps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode createTS_NRL(LandauCtx *ctx, Vec Temps)
{
  TSAdapt adapt;
  TS      ts;

  PetscFunctionBeginUser;
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  ctx->data = (void *)ts; // 'data' is for applications (eg, monitors)
  PetscCall(TSSetApplicationContext(ts, ctx));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, FormFunction, ctx));
  PetscCall(TSSetSolution(ts, Temps));
  PetscCall(TSRKSetType(ts, TSRK2A));
  PetscCall(TSSetOptionsPrefix(ts, "nrl_"));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetType(adapt, TSADAPTNONE));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetTime(ts, 0));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor_nrl(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  const PetscScalar *x;
  LandauCtx         *ctx = (LandauCtx *)actx; /* user-defined application context */

  PetscFunctionBeginUser;
  if (stepi % 100 == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "nrl-step %d time= %g ", (int)stepi, (double)(time / ctx->t_0)));
    PetscCall(VecGetArrayRead(X, &x));
    for (PetscInt i = 0; i < NUM_TEMPS; i++) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g ", (double)PetscRealPart(x[i]))); }
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  LandauCtx *ctx      = (LandauCtx *)actx; /* user-defined application context */
  TS         ts_nrl   = (TS)ctx->data;
  PetscInt   printing = 0, logT;

  PetscFunctionBeginUser;
  if (ctx->verbose > 0) { // hacks to generate sparse data (eg, use '-dm_landau_verbose 1' and '-dm_landau_verbose -1' to get all steps printed)
    PetscReal dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    logT = (PetscInt)PetscLog2Real(time / dt);
    if (logT < 0) logT = 0;
    ctx->verbose = PetscPowInt(2, logT) / 2;
    if (ctx->verbose == 0) ctx->verbose = 1;
  }
  if (ctx->verbose) {
    TSConvergedReason reason;
    PetscCall(TSGetConvergedReason(ts, &reason));
    if (stepi % ctx->verbose == 0 || reason || stepi == 1 || ctx->verbose < 0) {
      PetscInt nDMs, id;
      DM       pack;
      Vec     *XsubArray = NULL;
      printing           = 1;
      PetscCall(TSGetDM(ts, &pack));
      PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
      PetscCall(DMGetOutputSequenceNumber(ctx->plex[0], &id, NULL));
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[0], id + 1, time));
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[1], id + 1, time));
      PetscCall(PetscInfo(pack, "ex1 plot step %" PetscInt_FMT ", time = %g\n", id, (double)time));
      PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
      PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
      PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 0)], NULL, "-ex1_vec_view_e"));
      PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 1)], NULL, "-ex1_vec_view_i"));
      // temps
      for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
        PetscDS     prob;
        DM          dm      = ctx->plex[grid];
        PetscScalar user[2] = {0, 0}, tt[1];
        PetscReal   vz_0 = 0, n, energy, e_perp, e_par, m_s = ctx->masses[ctx->species_offset[grid]];
        Vec         Xloc = XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, grid)];
        PetscCall(DMGetDS(dm, &prob));
        /* get n */
        PetscCall(PetscDSSetObjective(prob, 0, &f0_n));
        PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, NULL));
        n = PetscRealPart(tt[0]);
        /* get vz */
        PetscCall(PetscDSSetObjective(prob, 0, &f0_vz));
        PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, NULL));
        user[0] = vz_0 = PetscRealPart(tt[0]) / n;
        /* energy temp */
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_shift));
        PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
        energy = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n / 3; // scale?
        energy *= kev_joul * 1000;                                         // T eV
        /* energy temp - par */
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_par_shift));
        PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
        e_par = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n;
        e_par *= kev_joul * 1000; // eV
        /* energy temp - perp */
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_perp));
        PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
        e_perp = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n / 2;
        e_perp *= kev_joul * 1000; // eV
        if (grid == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step %4d) time= %e temperature (eV): ", (int)stepi, (double)time));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s T= %9.4e T_par= %9.4e T_perp= %9.4e ", (grid == 0) ? "electron:" : ";ion:", (double)energy, (double)e_par, (double)e_perp));
        if (n_cm3[grid] == 0) n_cm3[grid] = ctx->n_0 * n * 1e-6; // does not change m^3 --> cm^3
      }
      // cleanup
      PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
      PetscCall(PetscFree(XsubArray));
    }
  }
  /* evolve NRL data, end line */
  if (n_cm3[NUM_TEMPS / 2 - 1] < 0 && ts_nrl) {
    PetscCall(TSDestroy(&ts_nrl));
    ctx->data = NULL;
    if (printing) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nSTOP printing NRL Ts\n"));
  } else if (ts_nrl) {
    const PetscScalar *x;
    PetscReal          dt_real, dt;
    Vec                U;
    PetscCall(TSGetTimeStep(ts, &dt)); // dt for NEXT time step
    dt_real = dt * ctx->t_0;
    PetscCall(TSGetSolution(ts_nrl, &U));
    if (printing) {
      PetscCall(VecGetArrayRead(U, &x));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NRL_i_par= %9.4e NRL_i_perp= %9.4e ", (double)PetscRealPart(x[I_PAR_IDX]), (double)PetscRealPart(x[I_PERP_IDX])));
      if (n_cm3[0] > 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NRL_e_par= %9.4e NRL_e_perp= %9.4e\n", (double)PetscRealPart(x[E_PAR_IDX]), (double)PetscRealPart(x[E_PERP_IDX])));
      } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
      PetscCall(VecRestoreArrayRead(U, &x));
    }
    // we have the next time step, so need to advance now
    PetscCall(TSSetTimeStep(ts_nrl, dt_real));
    PetscCall(TSSetMaxSteps(ts_nrl, stepi + 1)); // next step
    PetscCall(TSSolve(ts_nrl, NULL));
  } else if (printing) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  if (printing) { PetscCall(DMPlexLandauPrintNorms(X, stepi /*id + 1*/)); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          pack;
  Vec         X;
  PetscInt    dim = 2, nDMs;
  TS          ts, ts_nrl = NULL;
  Mat         J;
  Vec        *XsubArray = NULL;
  LandauCtx  *ctx;
  PetscMPIInt rank;
  PetscBool   use_nrl   = PETSC_TRUE;
  PetscBool   print_nrl = PETSC_FALSE;
  PetscReal   dt0;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) { /* turn off output stuff for duplicate runs */
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_dm_view_e"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_dm_view_i"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_vec_view_e"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_vec_view_i"));
    PetscCall(PetscOptionsClearValue(NULL, "-info"));
    PetscCall(PetscOptionsClearValue(NULL, "-snes_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-pc_bjkokkos_ksp_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-ksp_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-ts_adapt_monitor"));
    PetscCall(PetscOptionsClearValue(NULL, "-ts_monitor"));
    PetscCall(PetscOptionsClearValue(NULL, "-snes_monitor"));
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_nrl", &use_nrl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_nrl", &print_nrl, NULL));
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx->num_grids == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must have two grids: use '-dm_landau_num_species_grid 1,1'");
  PetscCheck(ctx->num_species == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must have two species: use '-dm_landau_num_species_grid 1,1'");
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
  /* output plot names */
  PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 0)], 0 == 0 ? "ue" : "ui"));
  PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 1)], 1 == 0 ? "ue" : "ui"));
  /* add bimaxwellian anisotropic test */
  for (PetscInt b_id = 0; b_id < ctx->batch_sz; b_id++) {
    for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
      PetscReal shifts[2];
      PetscCall(SetMaxwellians(ctx->plex[grid], XsubArray[LAND_PACK_IDX(b_id, grid)], 0.0, ctx->thermal_temps, ctx->n, grid, shifts, ctx));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
  PetscCall(PetscFree(XsubArray));
  /* plot */
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[0], -1, 0.0));
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[1], -1, 0.0));
  PetscCall(DMViewFromOptions(ctx->plex[0], NULL, "-ex1_dm_view_e"));
  PetscCall(DMViewFromOptions(ctx->plex[1], NULL, "-ex1_dm_view_i"));
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSMonitorSet(ts, Monitor, ctx, NULL));
  /* Create NRL timestepping */
  if (use_nrl || print_nrl) {
    Vec NRL_vec;
    PetscCall(createVec_NRL(ctx, &NRL_vec));
    PetscCall(createTS_NRL(ctx, NRL_vec));
    PetscCall(VecDestroy(&NRL_vec));
  } else ctx->data = NULL;
  /* solve */
  PetscCall(TSGetTimeStep(ts, &dt0));
  PetscCall(TSSetTime(ts, dt0 / 2));
  PetscCall(TSSolve(ts, X));
  /* test add field method & output */
  PetscCall(DMPlexLandauAccess(pack, X, landau_field_print_access_callback, NULL));
  //PetscCall(Monitor(ts, -1, 1.0, X, ctx));
  /* clean up */
  ts_nrl = (TS)ctx->data;
  if (print_nrl) {
    PetscReal    finalTime, dt_real, tstart = dt0 * ctx->t_0 / 2; // hack
    Vec          U;
    PetscScalar *x;
    PetscInt     nsteps;
    dt_real = dt0 * ctx->t_0;
    PetscCall(TSSetTimeStep(ts_nrl, dt_real));
    PetscCall(TSGetTime(ts, &finalTime));
    finalTime *= ctx->t_0;
    PetscCall(TSSetMaxTime(ts_nrl, finalTime));
    nsteps = (PetscInt)(finalTime / dt_real) + 1;
    PetscCall(TSSetMaxSteps(ts_nrl, nsteps));
    PetscCall(TSSetStepNumber(ts_nrl, 0));
    PetscCall(TSSetTime(ts_nrl, tstart));
    PetscCall(TSGetSolution(ts_nrl, &U));
    PetscCall(VecGetArray(U, &x));
    for (PetscInt i = 0; i < NUM_TEMPS; i++) x[i] = T0[i];
    PetscCall(VecRestoreArray(U, &x));
    PetscCall(TSMonitorSet(ts_nrl, Monitor_nrl, ctx, NULL));
    PetscCall(TSSolve(ts_nrl, NULL));
  }
  PetscCall(TSDestroy(&ts));
  PetscCall(TSDestroy(&ts_nrl));
  PetscCall(VecDestroy(&X));
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    requires: p4est !complex double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex1_0.out
    filter: grep -v "DM"
    args: -dm_landau_amr_levels_max 0,2 -dm_landau_amr_post_refine 0 -dm_landau_amr_re_levels 2 -dm_landau_domain_radius 6,6 -dm_landau_electron_shift 1.5 -dm_landau_ion_charges 1 -dm_landau_ion_masses 2 -dm_landau_n 1,1 -dm_landau_n_0 1e20 -dm_landau_num_cells 2,4 -dm_landau_num_species_grid 1,1 -dm_landau_re_radius 2 -use_nrl true -print_nrl false -dm_landau_thermal_temps .3,.2 -dm_landau_type p4est -dm_landau_verbose -1 -dm_preallocate_only false -ex1_dm_view_e -ksp_type preonly -pc_type lu -petscspace_degree 3 -snes_converged_reason -snes_rtol 1.e-14 -snes_stol 1.e-14 -ts_adapt_clip .5,1.5 -ts_adapt_dt_max 5 -ts_adapt_monitor -ts_adapt_scale_solve_failed 0.5 -ts_arkimex_type 1bee -ts_dt .01 -ts_max_snes_failures -1 -ts_max_steps 1 -ts_max_time 8 -ts_monitor -ts_rtol 1e-2 -ts_type arkimex
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu -dm_landau_use_relativistic_corrections
    test:
      suffix: kokkos
      requires: kokkos_kernels !defined(PETSC_HAVE_CUDA_CLANG)
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda
      requires: cuda !defined(PETSC_HAVE_CUDA_CLANG)
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -mat_cusparse_use_cpu_solve

  testset:
    requires: !complex defined(PETSC_USE_DMLANDAU_2D) p4est
    args: -dm_landau_type p4est -dm_landau_amr_levels_max 3,3 -dm_landau_num_species_grid 1,1 -dm_landau_n 1,1 -dm_landau_thermal_temps 1,1 -dm_landau_ion_charges 1 -dm_landau_ion_masses 2 -petscspace_degree 2 -ts_type beuler -ts_dt .1 -ts_max_steps 0 -dm_landau_verbose 2 -ksp_type preonly -pc_type lu -dm_landau_device_type cpu -use_nrl false -print_nrl -snes_rtol 1.e-14 -snes_stol 1.e-14 -dm_landau_device_type cpu
    nsize: 1
    test:
      suffix: sphere
      args: -dm_landau_sphere -ts_max_steps 1 -dm_landau_amr_post_refine 0
    test:
      suffix: re
      args: -dm_landau_num_cells 4,4 -dm_landau_amr_levels_max 0,2 -dm_landau_z_radius_pre 2.5 -dm_landau_z_radius_post 3.75 -dm_landau_amr_z_refine_pre 1 -dm_landau_amr_z_refine_post 1 -dm_landau_electron_shift 1.25 -ts_max_steps 1 -snes_converged_reason -info :vec

TEST*/
