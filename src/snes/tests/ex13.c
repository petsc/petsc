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
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-benchmark_it", "Solve the benchmark problem this many times", "ex13.c", options->nit, &options->nit, NULL));
  PetscCall(PetscOptionsBool("-strong", "Do not integrate the Laplacian by parts", "ex13.c", options->strong, &options->strong, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  if (user->strong) {
    PetscCall(PetscDSSetResidual(ds, 0, f0_strong_u, NULL));
    PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_u, user));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quadratic_u, NULL, user, NULL));
  } else {
    PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
    PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
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

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    /* TODO: Check whether the boundary of coarse meshes is marked */
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  /* Benchmark system */
  if (user.nit) {
#if defined(PETSC_USE_LOG)
    PetscLogStage kspstage,pcstage;
#endif
    KSP       ksp;
    PC        pc;
    Vec       b;
    PetscInt  i;
    PetscLogDouble time;
    PetscCall(PetscOptionsClearValue(NULL,"-ksp_monitor"));
    PetscCall(PetscOptionsClearValue(NULL,"-ksp_view"));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(VecSet(u, 0.0));
    PetscCall(SNESGetFunction(snes, &b, NULL, NULL));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscLogStageRegister("PCSetUp", &pcstage));
    PetscCall(PetscLogStagePush(pcstage));
    PetscCall(PCSetUp(pc));
    PetscCall(PetscLogStagePop());
    PetscCall(PetscLogStageRegister("KSP Solve only", &kspstage));
    PetscCall(PetscTime(&time));
    PetscCall(PetscLogStagePush(kspstage));
    for (i=0;i<user.nit;i++) {
      PetscCall(VecZeroEntries(u));
      PetscCall(KSPSolve(ksp, b, u));
    }
    PetscCall(PetscLogStagePop());
    PetscCall(PetscTimeSubtract(&time));
    // ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve time: %g\n",-time); // breaks CI
  }
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,8 -dm_refine 1 \
          -petscpartitioner_type simple -petscpartitioner_simple_process_grid 1,1,2 -petscpartitioner_simple_node_grid 1,1,2 \
          -potential_petscspace_degree 2 -ksp_type cg -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -benchmark_it 1 -dm_view -snes_rtol 1.e-4

  test:
    suffix: comparison
    nsize: 4
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4

  test:
    suffix: cuda
    nsize: 4
    requires: cuda
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijcusparse -dm_vec_type cuda

  test:
    suffix: kokkos_comp
    nsize: 4
    requires: !sycl kokkos_kernels
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijkokkos -dm_vec_type kokkos

  test:
    nsize: 4
    requires: !sycl kokkos_kernels
    suffix: kokkos
    args: -dm_plex_dim 2 -dm_plex_box_faces 2,8 -petscpartitioner_type simple -petscpartitioner_simple_process_grid 2,1 \
          -petscpartitioner_simple_node_grid 2,1 -dm_plex_simplex 0 -potential_petscspace_degree 1 -dm_refine 1 -ksp_type cg -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -ksp_norm_type unpreconditioned \
          -pc_gamg_esteig_ksp_type cg -ksp_converged_reason -snes_monitor_short -snes_rtol 1.e-4 -dm_view -dm_mat_type aijkokkos -dm_vec_type kokkos

  test:
    suffix: aijmkl_comp
    nsize: 4
    requires: mkl_sparse
    output_file: output/ex13_comparison.out
    args: -dm_plex_dim 2 -dm_plex_box_faces 4,4 -dm_refine 3 -petscpartitioner_simple_process_grid 2,2 \
      -petscpartitioner_simple_node_grid 1,1 -potential_petscspace_degree 2 -petscpartitioner_type simple \
      -dm_plex_simplex 0 -snes_monitor_short -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_process_eq_limit 400 -ksp_norm_type unpreconditioned \
      -pc_gamg_coarse_eq_limit 10 -snes_converged_reason -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijmkl

  test:
    suffix: aijmkl_seq
    nsize: 1
    requires: mkl_sparse
    TODO: broken (INDEFINITE PC)
    args: -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_refine 1 -petscpartitioner_type simple -potential_petscspace_degree 1 -dm_plex_simplex 0 -snes_monitor_short \
          -snes_type ksponly -dm_view -pc_type gamg -pc_gamg_sym_graph 0 -pc_gamg_threshold -1 -pc_gamg_square_graph 10 -pc_gamg_process_eq_limit 400 \
          -pc_gamg_reuse_interpolation -pc_gamg_coarse_eq_limit 10 -pc_gamg_esteig_ksp_type cg -ksp_type cg -ksp_norm_type unpreconditioned -snes_converged_reason \
          -ksp_converged_reason -snes_rtol 1.e-4 -dm_mat_type aijmkl -dm_vec_type standard

TEST*/
