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

  ierr = PetscOptionsBegin(comm, "", "FE Test Options", "PETSCFE");PetscCall(ierr);
  PetscCall(PetscOptionsString("-func", "Function to project into space", "", name, name, PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscOptionsReal("-shear", "Factor by which to shear along the x-direction", "", options->shear, &(options->shear), NULL));
  PetscCall(PetscOptionsReal("-flatten", "Factor by which to flatten", "", options->flatten, &(options->flatten), NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  for (i = 0; i < Nfunc; ++i) {
    PetscBool flg;

    PetscCall(PetscStrcmp(name, names[i], &flg));
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
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, f0, 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 1, user->exactSol, 0, NULL));
  PetscCall(PetscWeakFormSetIndexJacobian(wf, NULL, 0, 0, 0, 0, 0, g0, 0, NULL, 0, NULL, 0, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 0, exactSolution, user));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(DMCreateFEDefault(dm, 1, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name ? name : "Solution"));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
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
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]));
  PetscCall(DMProjectFunction(dm, 0.0, exactSol, exactCtx, INSERT_ALL_VALUES, u));
  PetscCall(VecViewFromOptions(u, NULL, "-interpolation_view"));
  PetscCall(DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  if (error > tol) PetscCall(PetscPrintf(comm, "Interpolation tests FAIL at tolerance %g error %g\n", (double) tol, (double) error));
  else             PetscCall(PetscPrintf(comm, "Interpolation tests pass at tolerance %g\n", (double) tol));
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
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]));
  PetscCall(SNESCreate(comm, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "solution"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, user, user, user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecViewFromOptions(u, NULL, "-l2_projection_view"));
  PetscCall(DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  if (error > tol) PetscCall(PetscPrintf(comm, "L2 projection tests FAIL at tolerance %g error %g\n", (double) tol, (double) error));
  else             PetscCall(PetscPrintf(comm, "L2 projection tests pass at tolerance %g\n", (double) tol));
  PetscFunctionReturn(0);
}

/* Distorts the mesh by shearing in the x-direction and flattening, factors provided in the options. */
static PetscErrorCode DistortMesh(DM dm, AppCtx *user)
{
  Vec            coordinates;
  PetscScalar   *ca;
  PetscInt       dE, n, i;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetCoordinates(dm, &coordinates));
  PetscCall(VecGetLocalSize(coordinates, &n));
  PetscCall(VecGetArray(coordinates, &ca));
  for (i = 0; i < (n/dE); ++i) {
    ca[i*dE+0] += user->shear*ca[i*dE+0];
    ca[i*dE+1] *= user->flatten;
  }
  PetscCall(VecRestoreArray(coordinates, &ca));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DistortMesh(dm,&user));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(SetupDiscretization(dm, NULL, &user));

  PetscCall(CheckInterpolation(dm, &user));
  PetscCall(CheckL2Projection(dm, &user));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
