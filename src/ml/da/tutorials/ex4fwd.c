static char help[] = "2D Shallow water equations forward model with MMS verification.\n"
                     "Implements 2D shallow water equations with 3 DOF per grid point (h, hu, hv).\n\n"
                     "Usage:\n"
                     "  ./ex4fwd -steps 500 -nx 80 -ny 80\n"
                     "  ./ex4fwd -verify_mms -steps 100 -nx 40 -ny 40 -verification_freq 10\n"
                     "  ./ex4fwd -test_mms_spatial_order -steps 5 -dt 1e-4\n"
                     "  ./ex4fwd -test_mms_spatial_order -conv_nx_coarse 20 -conv_ny_coarse 20 -conv_refine 2 -steps 5 -dt 1e-4\n\n";

#include <petscdmda.h>
#include <petscts.h>

/* Default parameter values */
#define DEFAULT_NX                40
#define DEFAULT_NY                40
#define DEFAULT_STEPS             100
#define DEFAULT_G                 9.81
#define DEFAULT_DT                0.02
#define DEFAULT_LX                80.0
#define DEFAULT_LY                80.0
#define DEFAULT_H0                1.5
#define DEFAULT_AX                0.2
#define DEFAULT_AY                0.2
#define DEFAULT_PROGRESS_FREQ     10
#define DEFAULT_VERIFICATION_FREQ 10
#define DEFAULT_CONV_NX_COARSE    12
#define DEFAULT_CONV_NY_COARSE    12
#define DEFAULT_CONV_REFINE       2

#include "ex4.h"

static PetscErrorCode ComputeManufacturedError(Vec numerical, DM da, PetscReal time, PetscReal Lx, PetscReal Ly, PetscReal h0, PetscReal A, PetscReal *L1_error, PetscReal *L2_error, PetscReal *Linf_error)
{
  const PetscScalar ***x_num;
  PetscInt             xs, ys, xm, ym, i, j;
  PetscInt             nx, ny;
  PetscReal            dx, dy, dA;
  PetscReal            L1_local = 0.0, L2_local = 0.0, Linf_local = 0.0;
  PetscReal            sums[2];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  dx = Lx / nx;
  dy = Ly / ny;
  dA = dx * dy;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAVecGetArrayDOFRead(da, numerical, (void *)&x_num));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal x = ((PetscReal)i + 0.5) * dx;
      PetscReal y = ((PetscReal)j + 0.5) * dy;
      PetscReal h_exact, hu_exact, hv_exact;
      PetscReal error;

      ManufacturedSolution2D(Lx, Ly, x, y, time, h0, A, &h_exact, &hu_exact, &hv_exact);
      error = PetscAbsReal(PetscRealPart(x_num[j][i][0]) - h_exact);
      L1_local += error * dA;
      L2_local += error * error * dA;
      Linf_local = PetscMax(Linf_local, error);
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da, numerical, (void *)&x_num));
  sums[0] = L1_local;
  sums[1] = L2_local;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums, 2, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  *L1_error = sums[0];
  *L2_error = PetscSqrtReal(sums[1]);
  PetscCallMPI(MPIU_Allreduce(&Linf_local, Linf_error, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)da)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunManufacturedCase(PetscInt nx, PetscInt ny, PetscInt steps, PetscReal g, PetscReal dt, PetscReal Lx, PetscReal Ly, PetscReal h0, PetscReal A, Ex4FluxType flux_type, PetscReal *L1_err, PetscReal *L2_err, PetscReal *Linf_err)
{
  DM                 da_state;
  ShallowWater2DCtx *sw_ctx;
  Vec                x_numerical;

  PetscFunctionBeginUser;
  PetscCall(SetupForwardProblem(nx, ny, Lx, Ly, g, dt, h0, A, A, PETSC_TRUE, flux_type, &da_state, &sw_ctx, &x_numerical));
  PetscCall(SetInitialCondition(da_state, x_numerical, sw_ctx, PETSC_TRUE));
  for (PetscInt step = 0; step < steps; step++) PetscCall(ShallowWaterStep2DVec(sw_ctx, step * dt, x_numerical));
  PetscCall(ComputeManufacturedError(x_numerical, da_state, steps * dt, Lx, Ly, h0, A, L1_err, L2_err, Linf_err));
  PetscCall(VecDestroy(&x_numerical));
  PetscCall(ShallowWater2DContextDestroy(&sw_ctx));
  PetscCall(DMDestroy(&da_state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt           nx = DEFAULT_NX, ny = DEFAULT_NY, steps = DEFAULT_STEPS, progress_freq = DEFAULT_PROGRESS_FREQ, verification_freq = DEFAULT_VERIFICATION_FREQ;
  PetscReal          g = DEFAULT_G, dt = DEFAULT_DT, Lx = DEFAULT_LX, Ly = DEFAULT_LY, h0 = DEFAULT_H0, Ax = DEFAULT_AX, Ay = DEFAULT_AY;
  PetscBool          verify_mms = PETSC_FALSE, test_mms_spatial_order = PETSC_FALSE;
  PetscInt           conv_nx_coarse = DEFAULT_CONV_NX_COARSE, conv_ny_coarse = DEFAULT_CONV_NY_COARSE, conv_refine = DEFAULT_CONV_REFINE;
  Ex4FluxType        flux_type = EX4_FLUX_RUSANOV;
  DM                 da_state;
  ShallowWater2DCtx *sw_ctx = NULL;
  Vec                x_numerical;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "2D Shallow Water Forward Model", NULL);
  PetscCall(PetscOptionsInt("-nx", "Number of grid points in x", "", nx, &nx, NULL));
  PetscCall(PetscOptionsInt("-ny", "Number of grid points in y", "", ny, &ny, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsReal("-g", "Gravitational constant", "", g, &g, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-Lx", "Domain length in x", "", Lx, &Lx, NULL));
  PetscCall(PetscOptionsReal("-Ly", "Domain length in y", "", Ly, &Ly, NULL));
  PetscCall(PetscOptionsReal("-h0", "Mean water height", "", h0, &h0, NULL));
  PetscCall(PetscOptionsReal("-Ax", "Wave amplitude in x", "", Ax, &Ax, NULL));
  PetscCall(PetscOptionsReal("-Ay", "Wave amplitude in y", "", Ay, &Ay, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsInt("-verification_freq", "Frequency for MMS error output", "", verification_freq, &verification_freq, NULL));
  PetscCall(PetscOptionsBool("-verify_mms", "Enable manufactured-solution verification forcing", "", verify_mms, &verify_mms, NULL));
  PetscCall(PetscOptionsBool("-test_mms_spatial_order", "Run a three-grid manufactured-solution spatial-order check and exit", "", test_mms_spatial_order, &test_mms_spatial_order, NULL));
  PetscCall(PetscOptionsInt("-conv_nx_coarse", "Coarse-grid nx for manufactured-solution spatial-order check", "", conv_nx_coarse, &conv_nx_coarse, NULL));
  PetscCall(PetscOptionsInt("-conv_ny_coarse", "Coarse-grid ny for manufactured-solution spatial-order check", "", conv_ny_coarse, &conv_ny_coarse, NULL));
  PetscCall(PetscOptionsInt("-conv_refine", "Grid refinement factor for manufactured-solution spatial-order check", "", conv_refine, &conv_refine, NULL));
  PetscCall(PetscOptionsEnum("-ex4_flux", "Flux scheme (rusanov/mc)", "", Ex4FluxTypes, (PetscEnum)flux_type, (PetscEnum *)&flux_type, NULL));
  PetscOptionsEnd();

  PetscCheck(nx > 0 && ny > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid dimensions must be positive");
  PetscCheck(steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative");
  PetscCheck(dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive");
  PetscCheck(!test_mms_spatial_order || conv_refine >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-conv_refine must be >= 2 for spatial-order check (got %" PetscInt_FMT ")", conv_refine);
  PetscCheck((!verify_mms && !test_mms_spatial_order) || PetscAbsReal(Ax - Ay) <= PETSC_MACHINE_EPSILON * PetscMax(PetscAbsReal(Ax), PetscAbsReal(Ay)), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "MMS verification requires Ax == Ay (isotropic amplitude); got Ax=%g, Ay=%g", (double)Ax, (double)Ay);

  if (test_mms_spatial_order) {
    PetscInt  medium_nx = conv_refine * conv_nx_coarse, medium_ny = conv_refine * conv_ny_coarse, fine_nx = conv_refine * medium_nx, fine_ny = conv_refine * medium_ny;
    PetscReal coarse_L1, coarse_L2, coarse_Linf, medium_L1, medium_L2, medium_Linf, fine_L1, fine_L2, fine_Linf;
    PetscReal order_cm_L1, order_cm_L2, order_cm_Linf;
    PetscReal order_mf_L1, order_mf_L2, order_mf_Linf;

    PetscCall(RunManufacturedCase(conv_nx_coarse, conv_ny_coarse, PetscMax(1, steps), g, dt, Lx, Ly, h0, Ax, flux_type, &coarse_L1, &coarse_L2, &coarse_Linf));
    PetscCall(RunManufacturedCase(medium_nx, medium_ny, PetscMax(1, steps), g, dt, Lx, Ly, h0, Ax, flux_type, &medium_L1, &medium_L2, &medium_Linf));
    PetscCall(RunManufacturedCase(fine_nx, fine_ny, PetscMax(1, steps), g, dt, Lx, Ly, h0, Ax, flux_type, &fine_L1, &fine_L2, &fine_Linf));
    PetscCheck(coarse_L1 > 0.0 && medium_L1 > 0.0 && fine_L1 > 0.0 && coarse_L2 > 0.0 && medium_L2 > 0.0 && fine_L2 > 0.0 && coarse_Linf > 0.0 && medium_Linf > 0.0 && fine_Linf > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "MMS error norm collapsed to zero on at least one grid (truncation below print precision); increase -steps or -dt to obtain a measurable error before the order check");
    order_cm_L1   = PetscLogReal(coarse_L1 / medium_L1) / PetscLogReal((PetscReal)conv_refine);
    order_cm_L2   = PetscLogReal(coarse_L2 / medium_L2) / PetscLogReal((PetscReal)conv_refine);
    order_cm_Linf = PetscLogReal(coarse_Linf / medium_Linf) / PetscLogReal((PetscReal)conv_refine);
    order_mf_L1   = PetscLogReal(medium_L1 / fine_L1) / PetscLogReal((PetscReal)conv_refine);
    order_mf_L2   = PetscLogReal(medium_L2 / fine_L2) / PetscLogReal((PetscReal)conv_refine);
    order_mf_Linf = PetscLogReal(medium_Linf / fine_Linf) / PetscLogReal((PetscReal)conv_refine);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "MMS spatial-order check\n"
                          "  coarse grid : %" PetscInt_FMT " x %" PetscInt_FMT ", dt=%.5e, L1=%.5e, L2=%.5e, Linf=%.5e\n"
                          "  medium grid : %" PetscInt_FMT " x %" PetscInt_FMT ", dt=%.5e, L1=%.5e, L2=%.5e, Linf=%.5e\n"
                          "  fine grid   : %" PetscInt_FMT " x %" PetscInt_FMT ", dt=%.5e, L1=%.5e, L2=%.5e, Linf=%.5e\n"
                          "  observed p (coarse->medium) : L1=%.2f, L2=%.2f, Linf=%.2f\n"
                          "  observed p (medium->fine)   : L1=%.2f, L2=%.2f, Linf=%.2f\n",
                          conv_nx_coarse, conv_ny_coarse, (double)dt, (double)coarse_L1, (double)coarse_L2, (double)coarse_Linf, medium_nx, medium_ny, (double)dt, (double)medium_L1, (double)medium_L2, (double)medium_Linf, fine_nx, fine_ny, (double)dt, (double)fine_L1, (double)fine_L2, (double)fine_Linf, (double)order_cm_L1, (double)order_cm_L2, (double)order_cm_Linf, (double)order_mf_L1, (double)order_mf_L2, (double)order_mf_Linf));
  } else {
    PetscCall(SetupForwardProblem(nx, ny, Lx, Ly, g, dt, h0, Ax, Ay, verify_mms, flux_type, &da_state, &sw_ctx, &x_numerical));
    PetscCall(SetInitialCondition(da_state, x_numerical, sw_ctx, verify_mms));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D Shallow Water Forward Example\n==============================\n"));

    for (PetscInt step = 1; step <= steps; step++) {
      PetscReal time          = step * dt;
      PetscBool emit_progress = (PetscBool)(step == steps || (progress_freq > 0 && step % progress_freq == 0));
      PetscCall(ShallowWaterStep2DVec(sw_ctx, (step - 1) * dt, x_numerical));
      if (verify_mms && (step % verification_freq == 0 || step == steps)) {
        PetscReal L1_err, L2_err, Linf_err;
        PetscCall(ComputeManufacturedError(x_numerical, da_state, time, Lx, Ly, h0, Ax, &L1_err, &L2_err, &Linf_err));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  L1=%.5e  L2=%.5e  Linf=%.5e\n", step, (double)time, (double)L1_err, (double)L2_err, (double)Linf_err));
      } else if (emit_progress) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  Forward step complete\n", step, (double)time));
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, verify_mms ? "\nMMS forward run complete.\n" : "\nForward simulation complete.\n"));
    PetscCall(VecDestroy(&x_numerical));
    PetscCall(ShallowWater2DContextDestroy(&sw_ctx));
    PetscCall(DMDestroy(&da_state));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: verification
    requires: !complex
    args: -verify_mms -steps 50 -nx 40 -ny 40 -verification_freq 10
    output_file: output/ex4fwd_verification.out

  test:
    suffix: verification_parallel
    requires: !complex
    nsize: 2
    args: -verify_mms -steps 20 -nx 30 -ny 30 -verification_freq 5
    output_file: output/ex4fwd_verification_parallel.out

  test:
    suffix: mms_spatial
    requires: !complex !single
    args: -test_mms_spatial_order -steps 5 -dt 1e-4
    filter: grep -E "MMS spatial-order check|observed p" | sed -E "s/=([0-9]+)\.([0-9])[0-9]/=\\1.\\2/g"

TEST*/
