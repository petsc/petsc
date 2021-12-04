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

static const char    *names[]     = {"constant", "linear", "quadratic", "trig"};
static PetscPointFunc functions[] = { constant,   linear,   quadratic,   trig};

typedef struct {
  PetscPointFunc exactSol;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  char           name[PETSC_MAX_PATH_LEN] = "constant";
  PetscInt       Nfunc = sizeof(names)/sizeof(char *), i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->exactSol = NULL;

  ierr = PetscOptionsBegin(comm, "", "FE Test Options", "PETSCFE");CHKERRQ(ierr);
  ierr = PetscOptionsString("-func", "Function to project into space", "", name, name, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  for (i = 0; i < Nfunc; ++i) {
    PetscBool flg;

    ierr = PetscStrcmp(name, names[i], &flg);CHKERRQ(ierr);
    if (flg) {options->exactSol = functions[i]; break;}
  }
  if (!options->exactSol) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Invalid test function %s", name);
  PetscFunctionReturn(0);
}

/* The exact solution is the negative of the f0 contribution */
static PetscErrorCode exactSolution(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  AppCtx  *user    = (AppCtx *) ctx;
  PetscInt uOff[2] = {0., Nc};

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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, f0, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 1, user->exactSol, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexJacobian(wf, NULL, 0, 0, 0, 0, 0, g0, 0, NULL, 0, NULL, 0, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 0, exactSolution, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = DMCreateFEDefault(dm, 1, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name ? name : "Solution");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
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
  PetscErrorCode       ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, exactSol, exactCtx, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-interpolation_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  if (error > tol) {ierr = PetscPrintf(comm, "Interpolation tests FAIL at tolerance %g error %g\n", (double) tol, (double) error);CHKERRQ(ierr);}
  else             {ierr = PetscPrintf(comm, "Interpolation tests pass at tolerance %g\n", (double) tol);CHKERRQ(ierr);}
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
  PetscErrorCode       ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscDSGetExactSolution(ds, 0, &exactSol[0], &exactCtx[0]);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, user, user, user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-l2_projection_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  if (error > tol) {ierr = PetscPrintf(comm, "L2 projection tests FAIL at tolerance %g error %g\n", (double) tol, (double) error);CHKERRQ(ierr);}
  else             {ierr = PetscPrintf(comm, "L2 projection tests pass at tolerance %g\n", (double) tol);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, NULL, &user);CHKERRQ(ierr);

  ierr = CheckInterpolation(dm, &user);CHKERRQ(ierr);
  ierr = CheckL2Projection(dm, &user);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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

TEST*/
