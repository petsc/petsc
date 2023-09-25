#include <petscksp.h>

static char help[] = "Solves a saddle-point linear system using PCHPDDM.\n\n";

static PetscErrorCode MatAndISLoad(const char *prefix, const char *identifier, Mat A, IS is, Mat N, PetscMPIInt rank, PetscMPIInt size);

int main(int argc, char **args)
{
  Vec               b;               /* computed solution and RHS */
  Mat               A[4], aux[2], S; /* linear system matrix */
  KSP               ksp, *subksp;    /* linear solver context */
  PC                pc;
  IS                is[2];
  PetscMPIInt       rank, size;
  PetscInt          m, M, n, N, id = 0;
  PetscViewer       viewer;
  const char *const system[] = {"elasticity", "stokes"};
  /* "elasticity":
   *    2D linear elasticity with rubber-like and steel-like material coefficients, i.e., Poisson's ratio \in {0.4999, 0.35} and Young's modulus \in {0.01 GPa, 200.0 GPa}
   *      discretized by order 2 (resp. 0) Lagrange finite elements in displacements (resp. pressure) on a triangle mesh
   * "stokes":
   *    2D lid-driven cavity with constant viscosity
   *      discretized by order 2 (resp. 1) Lagrange finite elements, i.e., lowest-order Taylor--Hood finite elements, in velocities (resp. pressure) on a triangle mesh
   *      if the option -empty_A11 is not set (or set to false), a pressure with a zero mean-value is computed
   */
  char      dir[PETSC_MAX_PATH_LEN], prefix[PETSC_MAX_PATH_LEN];
  PetscBool flg[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_USER, "This example requires 4 processes");
  PetscCall(PetscOptionsGetEList(NULL, NULL, "-system", system, PETSC_STATIC_ARRAY_LENGTH(system), &id, NULL));
  if (id == 1) PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_A11", flg, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(MatCreate(PETSC_COMM_WORLD, A + (i ? 3 : 0)));
    PetscCall(ISCreate(PETSC_COMM_SELF, is + i));
    PetscCall(MatCreate(PETSC_COMM_SELF, aux + i));
  }
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));
  /* loading matrices and auxiliary data for the diagonal blocks */
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/%s", dir, id == 1 ? "B" : "A"));
  PetscCall(MatAndISLoad(prefix, "00", A[0], is[0], aux[0], rank, size));
  PetscCall(MatAndISLoad(prefix, "11", A[3], is[1], aux[1], rank, size));
  /* loading the off-diagonal block with a coherent row/column layout */
  PetscCall(MatCreate(PETSC_COMM_WORLD, A + 2));
  PetscCall(MatGetLocalSize(A[0], &n, NULL));
  PetscCall(MatGetSize(A[0], &N, NULL));
  PetscCall(MatGetLocalSize(A[3], &m, NULL));
  PetscCall(MatGetSize(A[3], &M, NULL));
  PetscCall(MatSetSizes(A[2], m, n, M, N));
  PetscCall(MatSetUp(A[2]));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/%s10.dat", dir, id == 1 ? "B" : "A"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, prefix, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A[2], viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* transposing the off-diagonal block */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", flg + 1, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-permute", flg + 2, NULL));
  if (flg[1]) {
    if (!flg[2]) PetscCall(MatCreateTranspose(A[2], A + 1));
    else {
      PetscCall(MatTranspose(A[2], MAT_INITIAL_MATRIX, A + 1));
      PetscCall(MatDestroy(A + 2));
      PetscCall(MatCreateTranspose(A[1], A + 2));
    }
  } else {
    if (!flg[2]) PetscCall(MatCreateHermitianTranspose(A[2], A + 1));
    else {
      PetscCall(MatHermitianTranspose(A[2], MAT_INITIAL_MATRIX, A + 1));
      PetscCall(MatDestroy(A + 2));
      PetscCall(MatCreateHermitianTranspose(A[1], A + 2));
    }
  }
  if (flg[0]) PetscCall(MatDestroy(A + 3));
  /* global coefficient matrix */
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, A, &S));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, S, S));
  PetscCall(KSPGetPC(ksp, &pc));
  /* outer preconditioner */
  PetscCall(PCSetType(pc, PCFIELDSPLIT));
  PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));
  PetscCall(PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELF, NULL));
  PetscCall(PCSetUp(pc));
  PetscCall(PCFieldSplitGetSubKSP(pc, &n, &subksp));
  PetscCall(KSPGetPC(subksp[0], &pc));
  /* inner preconditioner associated to top-left block */
  PetscCall(PCSetType(pc, PCHPDDM));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[0], aux[0], NULL, NULL));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(KSPGetPC(subksp[1], &pc));
  /* inner preconditioner associated to Schur complement, which will be set internally to a PCKSP */
  PetscCall(PCSetType(pc, PCHPDDM));
  if (!flg[0]) PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[1], aux[1], NULL, NULL));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PetscFree(subksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(MatCreateVecs(S, &b, NULL));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/rhs_%s.dat", dir, id == 1 ? "B" : "A"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, prefix, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(b, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(KSPSolve(ksp, b, b));
  flg[0] = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-viewer", flg, NULL));
  if (flg[0]) PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(A + 1));
  PetscCall(MatDestroy(A + 2));
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(MatDestroy(A + (i ? 3 : 0)));
    PetscCall(MatDestroy(aux + i));
    PetscCall(ISDestroy(is + i));
  }
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode MatAndISLoad(const char *prefix, const char *identifier, Mat A, IS is, Mat aux, PetscMPIInt rank, PetscMPIInt size)
{
  IS              sizes;
  const PetscInt *idx;
  PetscViewer     viewer;
  char            name[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_sizes_%d_%d.dat", prefix, identifier, rank, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, name, FILE_MODE_READ, &viewer));
  PetscCall(ISCreate(PETSC_COMM_SELF, &sizes));
  PetscCall(ISLoad(sizes, viewer));
  PetscCall(ISGetIndices(sizes, &idx));
  PetscCall(MatSetSizes(A, idx[0], idx[1], idx[2], idx[3]));
  PetscCall(MatSetUp(A));
  PetscCall(ISRestoreIndices(sizes, &idx));
  PetscCall(ISDestroy(&sizes));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s.dat", prefix, identifier));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_is_%d_%d.dat", prefix, identifier, rank, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, name, FILE_MODE_READ, &viewer));
  PetscCall(ISLoad(is, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_aux_%d_%d.dat", prefix, identifier, rank, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(aux, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: hpddm slepc double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)

   testset:
      requires: datafilespath
      nsize: 4
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -ksp_monitor -ksp_rtol 1e-4 -fieldsplit_ksp_max_it 100 -fieldsplit_pc_hpddm_levels_1_eps_nev 10 -fieldsplit_pc_hpddm_levels_1_st_share_sub_ksp -fieldsplit_pc_hpddm_has_neumann -fieldsplit_pc_hpddm_define_subdomains -fieldsplit_1_pc_hpddm_schur_precondition geneo -fieldsplit_pc_hpddm_coarse_pc_type redundant -fieldsplit_pc_hpddm_coarse_redundant_pc_type cholesky -fieldsplit_pc_hpddm_levels_1_sub_pc_type lu -fieldsplit_ksp_type fgmres -ksp_type fgmres -ksp_max_it 10 -fieldsplit_1_pc_hpddm_coarse_correction balanced -fieldsplit_1_pc_hpddm_levels_1_eps_gen_non_hermitian -fieldsplit_1_pc_hpddm_coarse_p 2
      test:
        requires: mumps
        suffix: 1
        args: -viewer -system {{elasticity stokes}separate output} -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_sub_mat_mumps_icntl_26 1
        filter: grep -v -e "action of " -e "                            " -e "block size" -e "total: nonzeros=" -e "using I-node" -e "aij" -e "transpose" -e "diagonal" -e "total number of" -e "                rows="
      test:
        requires: mumps
        suffix: 2
        output_file: output/ex87_1_system-stokes.out
        args: -viewer -system stokes -empty_A11 -transpose {{false true}shared output} -permute {{false true}shared output} -fieldsplit_1_pc_hpddm_ksp_pc_side right -fieldsplit_1_pc_hpddm_coarse_mat_type baij -fieldsplit_1_pc_hpddm_levels_1_sub_mat_mumps_icntl_26 1
        filter: grep -v -e "action of " -e "                            " -e "block size" -e "total: nonzeros=" -e "using I-node" -e "aij" -e "transpose" -e "diagonal" -e "total number of" -e "                rows=" | sed -e "s/      right preconditioning/      left preconditioning/g" -e "s/      using UNPRECONDITIONED/      using PRECONDITIONED/g"
      test:
        suffix: 1_petsc
        args: -system {{elasticity stokes}separate output} -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_mat_solver_type petsc -fieldsplit_1_pc_hpddm_levels_1_eps_threshold 0.3 -permute
      test:
        suffix: 2_petsc
        output_file: output/ex87_1_petsc_system-stokes.out
        args: -system stokes -empty_A11 -transpose -fieldsplit_1_pc_hpddm_ksp_pc_side right -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_mat_solver_type petsc -fieldsplit_1_pc_hpddm_coarse_mat_type baij -fieldsplit_1_pc_hpddm_levels_1_eps_threshold 0.3 -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_shift_type inblocks
        filter: sed -e "s/type: transpose/type: hermitiantranspose/g"
      test:
        suffix: threshold
        output_file: output/ex87_1_petsc_system-elasticity.out
        args: -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_eps_threshold 0.2 -fieldsplit_1_pc_hpddm_coarse_mat_type {{baij sbaij}shared output}

TEST*/
