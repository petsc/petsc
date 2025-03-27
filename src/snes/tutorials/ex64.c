static char help[] = "Biot consolidation model discretized with finite elements,\n\
using a parallel unstructured mesh (DMPLEX) to represent the domain.\n\
We follow https://arxiv.org/pdf/1507.03199.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  PetscScalar mu;
  PetscScalar lam;
  PetscScalar alpha;
  PetscScalar kappa;
} Parameter;

static void u_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal mu = PetscRealPart(constants[0]);

  for (PetscInt c = 0; c < dim; ++c) {
    for (PetscInt d = 0; d < dim; ++d) f[c * dim + d] = mu * (u_x[c * dim + d] + u_x[d * dim + c]);
    f[c * dim + c] -= u[uOff[1]];
  }
}

/* Jfunction_testtrial */
static void Ju_1_u1p0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  for (PetscInt d = 0; d < dim; ++d) J[d * dim + d] = -1.0;
}

static void Ju_1_u1u1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal mu = PetscRealPart(constants[0]);
  const PetscInt  Nc = uOff[1] - uOff[0];

  for (PetscInt c = 0; c < Nc; ++c) {
    for (PetscInt d = 0; d < dim; ++d) {
      J[((c * Nc + c) * dim + d) * dim + d] += mu;
      J[((c * Nc + d) * dim + d) * dim + c] += mu;
    }
  }
}

static void p_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  f[0] = (-u[uOff[1]] + alpha * u[uOff[2]]) / lam;
  for (PetscInt d = 0; d < dim; ++d) f[0] -= u_x[d * dim + d];
}

static void Jp_0_p0u1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  for (PetscInt d = 0; d < dim; ++d) J[d * dim + d] = -1.0;
}

static void Jp_0_p0p0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal lam = PetscRealPart(constants[1]);

  J[0] = -1.0 / lam;
}

static void pJp_0_p0p0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal mu = PetscRealPart(constants[0]);

  J[0] = 1.0 / mu;
}

static void Jp_0_p0w0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  J[0] = alpha / lam;
}

static void w_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  f[0] = alpha / lam * (-2 * alpha * u[uOff[2]] + u[uOff[1]]);
}

static void Jw_0_w0p0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  J[0] = alpha / lam;
}

static void Jw_0_w0w0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  J[0] = -2 * PetscSqr(alpha) / lam;
}

static void pJw_0_w0w0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal lam   = PetscRealPart(constants[1]);
  const PetscReal alpha = PetscRealPart(constants[2]);

  J[0] = 2 * PetscSqr(alpha) / lam;
}

static void w_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal kappa = PetscRealPart(constants[3]);
  for (PetscInt d = 0; d < dim; ++d) f[d] = -kappa * u_x[uOff_x[2] + d];
}

static void Jw_1_w1w1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal kappa = PetscRealPart(constants[3]);

  for (PetscInt d = 0; d < dim; ++d) J[d * dim + d] = -kappa;
}

static void pJw_1_w1w1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal kappa = PetscRealPart(constants[3]);

  for (PetscInt d = 0; d < dim; ++d) J[d * dim + d] = kappa;
}

typedef struct {
  PetscScalar E;
  PetscScalar nu;
  PetscScalar alpha;
  PetscScalar kappa;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  PetscFunctionBeginUser;
  user->E     = 1.0;
  user->nu    = 0.3;
  user->alpha = 0.5;
  user->kappa = 1.0;
  PetscOptionsBegin(comm, "", "Biot Problem Options", "DMPLEX");
  PetscCall(PetscOptionsScalar("-E", "Young modulus", NULL, user->E, &user->E, NULL));
  PetscCall(PetscOptionsScalar("-nu", "Poisson ratio", NULL, user->nu, &user->nu, NULL));
  PetscCall(PetscOptionsScalar("-alpha", "Alpha", NULL, user->alpha, &user->alpha, NULL));
  PetscCall(PetscOptionsScalar("-kappa", "kappa", NULL, user->kappa, &user->kappa, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupEqn(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscScalar    constants[4];

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, u_1));
  PetscCall(PetscDSSetResidual(ds, 1, p_0, NULL));
  PetscCall(PetscDSSetResidual(ds, 2, w_0, w_1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, Ju_1_u1u1));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, Ju_1_u1p0, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, Jp_0_p0u1, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, Jp_0_p0p0, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 2, Jp_0_p0w0, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 2, 1, Jw_0_w0p0, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 2, 2, Jw_0_w0w0, NULL, NULL, Jw_1_w1w1));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 0, 0, NULL, NULL, NULL, Ju_1_u1u1));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 0, 1, NULL, NULL, Ju_1_u1p0, NULL));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 1, 0, NULL, Jp_0_p0u1, NULL, NULL));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 1, 1, pJp_0_p0p0, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobianPreconditioner(ds, 2, 2, pJw_0_w0w0, NULL, NULL, pJw_1_w1w1));

  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall-d", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall-p", label, 1, &id, 2, 0, NULL, NULL, NULL, NULL, NULL));

  constants[0] = user->E / (2.0 * (1.0 + user->nu));
  constants[1] = user->nu * user->E / ((1.0 + user->nu) * (1.0 - 2.0 * user->nu));
  constants[2] = user->alpha;
  constants[3] = user->kappa;
  PetscCall(PetscDSSetConstants(ds, 4, constants));

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "E = %g, nu = %g\n", (double)PetscRealPart(user->E), (double)PetscRealPart(user->nu)));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "mu = %g, lambda = %g\n", (double)PetscRealPart(constants[0]), (double)PetscRealPart(constants[1])));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "alpha = %g, kappa = %g\n", (double)PetscRealPart(constants[2]), (double)PetscRealPart(constants[3])));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupProblem(DM dm, PetscErrorCode (*setupEqn)(DM, AppCtx *), AppCtx *user)
{
  DM              cdm = dm;
  PetscQuadrature q   = NULL;
  PetscBool       simplex;
  PetscInt        dim, Nf = 3, f, Nc[3];
  const char     *name[3]   = {"displacement", "totalpressure", "pressure"};
  const char     *prefix[3] = {"displacement_", "totalpressure_", "pressure_"};

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  Nc[0] = dim;
  Nc[1] = 1;
  Nc[2] = 1;
  for (f = 0; f < Nf; ++f) {
    PetscFE fe;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc[f], simplex, prefix[f], -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, name[f]));
    if (!q) PetscCall(PetscFEGetQuadrature(fe, &q));
    PetscCall(PetscFESetQuadrature(fe, q));
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setupEqn)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES   snes;
  DM     dm;
  Vec    u;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupProblem(dm, SetupEqn, &user));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESSetFromOptions(snes));

  /* Solve with random rhs */
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSetRandom(u, NULL));
  PetscCall(SNESSolve(snes, NULL, u));

  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p2_p1_p2
    args: -displacement_petscspace_degree 2 -totalpressure_petscspace_degree 1 -pressure_petscspace_degree 2 \
          -dm_plex_box_faces 5,5 -dm_plex_separate_marker -dm_plex_simplex 0 \
          -snes_error_if_not_converged -ksp_error_if_not_converged \
          -ksp_type minres -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_totalpressure_pc_type jacobi -fieldsplit_displacement_pc_type gamg -fieldsplit_pressure_pc_type gamg

  test:
    nsize: 4
    suffix: 2d_p2_p1_p2_fetidp
    args: -displacement_petscspace_degree 2 -totalpressure_petscspace_degree 1 -pressure_petscspace_degree 2 \
          -petscpartitioner_type simple -dm_mat_type is -dm_plex_box_faces 3,3 -dm_refine 2 -dm_plex_separate_marker -dm_plex_simplex 0 \
          -snes_error_if_not_converged -ksp_error_if_not_converged \
          -ksp_type fetidp -ksp_fetidp_saddlepoint -ksp_fetidp_pressure_field 1,2 -ksp_fetidp_pressure_schur -fetidp_ksp_type cg -fetidp_ksp_norm_type natural \
          -fetidp_fieldsplit_lag_ksp_type preonly \
          -fetidp_bddc_pc_bddc_detect_disconnected -fetidp_bddc_pc_bddc_corner_selection -fetidp_bddc_pc_bddc_symmetric -fetidp_bddc_pc_bddc_coarse_redundant_pc_type cholesky \
          -fetidp_pc_discrete_harmonic 1 -fetidp_harmonic_pc_type cholesky \
          -fetidp_fieldsplit_p_pc_type bddc -fetidp_fieldsplit_p_ksp_type preonly

TEST*/
