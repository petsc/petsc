static const char help[] = "Tests for determining whether a new finite element works";

/*
  Use -interpolation_view and -l2_projection_view to look at the interpolants.
*/

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscsnes.h>

static void constant(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) f0[c] += 5.;
}

static void linear(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) f0[c] += 5.*x[c];
}

static void quadratic(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) f0[c] += 5.*x[c]*x[c];
}

static void trig(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) f0[c] += PetscCosReal(2.*PETSC_PI*x[c]);
}

/*
 The prime basis for the Wheeler-Yotov-Xue prism.
 */
static void prime(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal x = X[0], y = X[1], z = X[2], b = 1 + x + y + z;
  f0[0] += b + 2.0*x*z + 2.0*y*z + x*y + x*x;
  f0[1] += b + 2.0*x*z + 2.0*y*z + x*y + y*y;
  f0[2] += b - 3.0*x*z - 3.0*y*z -   2.0*z*z;
}

static const char    *names[]     = {"constant", "linear", "quadratic", "trig", "prime"};
static PetscPointFunc functions[] = { constant,   linear,   quadratic,   trig,   prime };

typedef struct {
  PetscPointFunc exactSol;
  PetscReal shear,flatten;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  char           name[PETSC_MAX_PATH_LEN] = "constant";
  PetscInt       Nfunc = sizeof(names)/sizeof(char *), i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->exactSol = NULL;
  options->shear    = 0.;
  options->flatten  = 1.;

  ierr = PetscOptionsBegin(comm, "", "FE Test Options", "PETSCFE");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-func", "Function to project into space", "", name, name, PETSC_MAX_PATH_LEN, NULL));
  CHKERRQ(PetscOptionsReal("-shear", "Factor by which to shear along the x-direction", "", options->shear, &(options->shear), NULL));
  CHKERRQ(PetscOptionsReal("-flatten", "Factor by which to flatten", "", options->flatten, &(options->flatten), NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  for (i = 0; i < Nfunc; ++i) {
    PetscBool flg;

    CHKERRQ(PetscStrcmp(name, names[i], &flg));
    if (flg) {options->exactSol = functions[i]; break;}
  }
  PetscCheck(options->exactSol, comm, PETSC_ERR_ARG_WRONG, "Invalid test function %s", name);
  PetscFunctionReturn(0);
}

/* The exact solution is the negative of the f0 contribution */
static PetscErrorCode exactSolution(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user    = (AppCtx *) ctx;
  PetscInt uOff[2] = {0, Nc};

  user->exactSol(dim, 1, 0, uOff, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, time, x, 0, NULL, u);
  for (PetscInt c = 0; c < Nc; ++c) u[c] *= -1.;
  return 0;
}

static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) f0[c] += u[c];
}

static void g0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  for (PetscInt c = 0; c < Nc; ++c) g0[c*Nc+c] = 1.0;
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSGetWeakForm(ds, &wf));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, f0, 0, NULL));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 1, user->exactSol, 0, NULL));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, NULL, 0, 0, 0, 0, 0, g0, 0, NULL, 0, NULL, 0, NULL));
  CHKERRQ(PetscDSSetExactSolution(ds, 0, exactSolution, user));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(DMCreateFEDefault(dm, 1, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name ? name : "Solution"));
  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

/* This test tells us whether the given function is contained in the approximation space */
static PetscErrorCode CheckInterpolation(DM dm, AppCtx *user)
{
  PetscSimplePointFunc exactSol[1];
  void                *exactCtx[1];
  PetscDS              ds;
  Vec                  u;
  PetscReal            error, tol = PETSC_SMALL;
  MPI_Comm             comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]));
  CHKERRQ(DMProjectFunction(dm, 0.0, exactSol, exactCtx, INSERT_ALL_VALUES, u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-interpolation_view"));
  CHKERRQ(DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  if (error > tol) CHKERRQ(PetscPrintf(comm, "Interpolation tests FAIL at tolerance %g error %g\n", (double) tol, (double) error));
  else             CHKERRQ(PetscPrintf(comm, "Interpolation tests pass at tolerance %g\n", (double) tol));
  PetscFunctionReturn(0);
}

/* This test tells us whether the element is unisolvent (the mass matrix has full rank), and what rate of convergence we achieve */
static PetscErrorCode CheckL2Projection(DM dm, AppCtx *user)
{
  PetscSimplePointFunc exactSol[1];
  void                *exactCtx[1];
  SNES                 snes;
  PetscDS              ds;
  Vec                  u;
  PetscReal            error, tol = PETSC_SMALL;
  MPI_Comm             comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]));
  CHKERRQ(SNESCreate(comm, &snes));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(VecSet(u, 0.0));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "solution"));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm, user, user, user));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(DMSNESCheckFromOptions(snes, u));
  CHKERRQ(SNESSolve(snes, NULL, u));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(VecViewFromOptions(u, NULL, "-l2_projection_view"));
  CHKERRQ(DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  if (error > tol) CHKERRQ(PetscPrintf(comm, "L2 projection tests FAIL at tolerance %g error %g\n", (double) tol, (double) error));
  else             CHKERRQ(PetscPrintf(comm, "L2 projection tests pass at tolerance %g\n", (double) tol));
  PetscFunctionReturn(0);
}

/* Distorts the mesh by shearing in the x-direction and flattening, factors provided in the options. */
static PetscErrorCode DistortMesh(DM dm, AppCtx *user)
{
  Vec            coordinates;
  PetscScalar   *ca;
  PetscInt       dE, n, i;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetCoordinateDim(dm, &dE));
  CHKERRQ(DMGetCoordinates(dm, &coordinates));
  CHKERRQ(VecGetLocalSize(coordinates, &n));
  CHKERRQ(VecGetArray(coordinates, &ca));
  for (i = 0; i < (n/dE); ++i) {
    ca[i*dE+0] += user->shear*ca[i*dE+0];
    ca[i*dE+1] *= user->flatten;
  }
  CHKERRQ(VecRestoreArray(coordinates, &ca));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DistortMesh(dm,&user));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(SetupDiscretization(dm, NULL, &user));

  CHKERRQ(CheckInterpolation(dm, &user));
  CHKERRQ(CheckL2Projection(dm, &user));

  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -petscspace_degree 1\
          -snes_error_if_not_converged -ksp_error_if_not_converged -pc_type lu

    test:
      suffix: p1_0
      args: -func {{constant linear}}

    # Using -dm_refine 2 -convest_num_refine 4 gives convergence rate 2.0
    test:
      suffix: p1_1
      args: -func {{quadratic trig}} \
            -snes_convergence_estimate -convest_num_refine 2

  testset:
    requires: !complex double
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism \
            -petscspace_type sum \
            -petscspace_variables 3 \
            -petscspace_components 3 \
            -petscspace_sum_spaces 2 \
            -petscspace_sum_concatenate false \
              -sumcomp_0_petscspace_variables 3 \
              -sumcomp_0_petscspace_components 3 \
              -sumcomp_0_petscspace_degree 1 \
              -sumcomp_1_petscspace_variables 3 \
              -sumcomp_1_petscspace_components 3 \
              -sumcomp_1_petscspace_type wxy \
            -petscdualspace_form_degree 0 \
            -petscdualspace_order 1 \
            -petscdualspace_components 3 \
          -snes_error_if_not_converged -ksp_error_if_not_converged -pc_type lu

    test:
      suffix: wxy_0
      args: -func constant

    test:
      suffix: wxy_1
      args: -func linear

    test:
      suffix: wxy_2
      args: -func prime

    test:
      suffix: wxy_3
      args: -func linear -shear 1 -flatten 1e-5

TEST*/
