static char help[] = "Benchmark Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular domain\n\
using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  PetscInt  nit;    /* Number of benchmark iterations */
  PetscBool strong; /* Do not integrate the Laplacian by parts */
} AppCtx;

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode quadratic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = PetscSqr(x[0]) + PetscSqr(x[1]);
  return 0;
}

static void f0_strong_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] -= u_x[dim + d*dim+d];
  f0[0] += 4.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->nit    = 10;
  options->strong = PETSC_FALSE;
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-benchmark_it", "Solve the benchmark problem this many times", "ex13.c", options->nit, &options->nit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-strong", "Do not integrate the Laplacian by parts", "ex13.c", options->strong, &options->strong, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  if (user->strong) {
    ierr = PetscDSSetResidual(ds, 0, f0_strong_u, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(ds, 0, quadratic_u, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quadratic_u, NULL, user, NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  /* Benchmark system */
  if (user.nit) {
#if defined(PETSC_USE_LOG)
    PetscLogStage kspstage,pcstage;
#endif
    KSP       ksp;
    PC        pc;
    Mat       A,P;
    Vec       b;
    PetscInt  i;
    ierr = PetscOptionsClearValue(NULL,"-ksp_monitor");CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &A, &P, NULL, NULL);CHKERRQ(ierr);
    ierr = VecSet(u, 0.0);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes, &b, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, u, b);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(snes, u, A, P);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("PCSetUp", &pcstage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(pcstage);CHKERRQ(ierr);
    ierr = PCSetUp(pc);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscLogStageRegister("KSP Solve only", &kspstage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(kspstage);CHKERRQ(ierr);
    for (i=0;i<user.nit;i++) {
      ierr = VecZeroEntries(u);CHKERRQ(ierr);
      ierr = KSPSolve(ksp, b, u);CHKERRQ(ierr);
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: strong
    requires: triangle
    args: -dm_plex_dim 2 -dm_refine 1 -benchmark_it 0 -dmsnes_check \
          -potential_petscspace_degree 2 -dm_ds_jet_degree 2 -strong

  test:
    suffix: bench
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,8 -dm_refine 1 -dm_distribute \
          -petscpartitioner_type simple -petscpartitioner_simple_process_grid 1,1,2 -petscpartitioner_simple_node_grid 1,1,2 \
          -potential_petscspace_degree 2 -ksp_type cg -pc_type gamg -benchmark_it 1 -dm_view -snes_rtol 1.e-4

  test:
    suffix: comparison
    nsize: 4
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -dm_distribute -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4

  test:
    suffix: cuda
    nsize: 4
    requires: cuda
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -dm_distribute -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijcusparse -dm_vec_type cuda

  test:
    suffix: kokkos_comp
    nsize: 4
    requires: kokkos_kernels
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -dm_distribute -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijkokkos -dm_vec_type kokkos

  test:
    nsize: 4
    requires: kokkos_kernels
    suffix: kokkos
    args: -dm_plex_dim 2 -dm_plex_box_faces 2,8 -dm_distribute -petscpartitioner_type simple -petscpartitioner_simple_process_grid 2,1 \
          -petscpartitioner_simple_node_grid 2,1 -dm_plex_simplex 0 -potential_petscspace_degree 1 -dm_refine 1 -ksp_type cg -pc_type gamg -ksp_norm_type unpreconditioned \
          -mg_levels_esteig_ksp_type cg -mg_levels_pc_type jacobi -ksp_converged_reason -snes_monitor_short -snes_rtol 1.e-4 -dm_view -dm_mat_type aijkokkos -dm_vec_type kokkos

  test:
    suffix: aijmkl_comp
    nsize: 4
    requires: mkl
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -dm_distribute -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijmkl

  test:
    suffix: aijmkl_seq
    nsize: 1
    requires: mkl
    TODO: broken (INDEFINITE PC)
    args: -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_refine 1 -petscpartitioner_type simple -potential_petscspace_degree 1 -dm_distribute -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_sym_graph 0 -pc_gamg_threshold -1 -pc_gamg_square_graph 10 -pc_gamg_process_eq_limit 400 -pc_gamg_reuse_interpolation -pc_gamg_coarse_eq_limit 10 -mg_levels_esteig_ksp_type cg -mg_levels_pc_type jacobi -ksp_type cg -ksp_norm_type unpreconditioned -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijmkl -dm_vec_type standard

TEST*/
