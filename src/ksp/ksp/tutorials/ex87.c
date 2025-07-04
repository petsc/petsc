#include <petscksp.h>
#include <petsc/private/petscimpl.h>

static char help[] = "Solves a saddle-point linear system using PCHPDDM.\n\n";

static PetscErrorCode MatAndISLoad(const char *prefix, const char *identifier, Mat A, IS is, Mat N, PetscMPIInt size);

int main(int argc, char **args)
{
  Vec               b, x;            /* computed solution and RHS */
  Mat               A[4], aux[2], S; /* linear system matrix */
  KSP               ksp, *subksp;    /* linear solver context */
  PC                pc;
  IS                is[2];
  PetscMPIInt       size;
  PetscInt          m, M, n, N, id = 0;
  PetscViewer       viewer;
  const char *const system[] = {"elasticity", "stokes", "diffusion", "lagrange"};
  /* "elasticity":
   *    2D linear elasticity with rubber-like and steel-like material coefficients, i.e., Poisson's ratio \in {0.4999, 0.35} and Young's modulus \in {0.01 GPa, 200.0 GPa}
   *      discretized by order 2 (resp. 0) Lagrange finite elements in displacements (resp. pressure) on a triangle mesh
   * "stokes":
   *    2D lid-driven cavity with constant viscosity
   *      discretized by order 2 (resp. 1) Lagrange finite elements, i.e., lowest-order Taylor--Hood finite elements, in velocities (resp. pressure) on a triangle mesh
   *      if the option -empty_A11 is not set (or set to false), a pressure with a zero mean-value is computed
   * "diffusion":
   *    2D primal-dual nonsymmetric diffusion equation
   *      discretized by order 2 (resp. 1) Lagrange finite elements in primal (resp. dual) unknowns on a triangle mesh
   * "lagrange":
   *    2D linear elasticity with essential boundary conditions imposed through a Lagrange multiplier
   */
  char      dir[PETSC_MAX_PATH_LEN], prefix[PETSC_MAX_PATH_LEN];
  PCType    type;
  PetscBool flg[4] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetEList(NULL, NULL, "-system", system, PETSC_STATIC_ARRAY_LENGTH(system), &id, NULL));
  if (id == 1) PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_A11", flg, NULL));
  if (id != 3) PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example requires 4 processes");
  else PetscCheck(id == 3 && size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example requires 2 processes");
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(MatCreate(PETSC_COMM_WORLD, A + (i ? 3 : 0)));
    if (id < 2 || (id == 3 && i == 0)) {
      PetscCall(ISCreate(PETSC_COMM_SELF, is + i));
      PetscCall(MatCreate(PETSC_COMM_SELF, aux + i));
    } else {
      is[i]  = NULL;
      aux[i] = NULL;
    }
  }
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));
  /* loading matrices and auxiliary data for the diagonal blocks */
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/%s", dir, id == 3 ? "D" : (id == 2 ? "C" : (id == 1 ? "B" : "A"))));
  PetscCall(MatAndISLoad(prefix, "00", A[0], is[0], aux[0], size));
  PetscCall(MatAndISLoad(prefix, "11", A[3], is[1], aux[1], size));
  /* loading the off-diagonal block with a coherent row/column layout */
  PetscCall(MatCreate(PETSC_COMM_WORLD, A + 2));
  PetscCall(MatGetLocalSize(A[0], &n, NULL));
  PetscCall(MatGetSize(A[0], &N, NULL));
  PetscCall(MatGetLocalSize(A[3], &m, NULL));
  PetscCall(MatGetSize(A[3], &M, NULL));
  PetscCall(MatSetSizes(A[2], m, n, M, N));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/%s10.dat", dir, id == 3 ? "D" : (id == 2 ? "C" : (id == 1 ? "B" : "A"))));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, prefix, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A[2], viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  if (id != 2) {
    /* transposing the off-diagonal block */
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", flg + 1, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-permute", flg + 2, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit", flg + 3, NULL));
    if (flg[1]) {
      if (flg[2]) {
        PetscCall(MatTranspose(A[2], MAT_INITIAL_MATRIX, A + 1));
        PetscCall(MatDestroy(A + 2));
      }
      if (!flg[3]) PetscCall(MatCreateTranspose(A[2 - flg[2]], A + 1 + flg[2]));
      else PetscCall(MatTranspose(A[2 - flg[2]], MAT_INITIAL_MATRIX, A + 1 + flg[2]));
    } else {
      if (flg[2]) {
        PetscCall(MatHermitianTranspose(A[2], MAT_INITIAL_MATRIX, A + 1));
        PetscCall(MatDestroy(A + 2));
      }
      if (!flg[3]) PetscCall(MatCreateHermitianTranspose(A[2 - flg[2]], A + 1 + flg[2]));
      else PetscCall(MatHermitianTranspose(A[2 - flg[2]], MAT_INITIAL_MATRIX, A + 1 + flg[2]));
    }
  } else {
    PetscCall(MatCreate(PETSC_COMM_WORLD, A + 1));
    PetscCall(MatSetSizes(A[1], n, m, N, M));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/C01.dat", dir));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, prefix, FILE_MODE_READ, &viewer));
    PetscCall(MatLoad(A[1], viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  if (flg[0]) PetscCall(MatDestroy(A + 3));
  else {
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-diagonal_A11", flg, NULL));
    if (flg[0]) {
      PetscCall(MatDestroy(A + 3));
      PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, m, m, M, M, PETSC_SMALL, A + 3));
    }
  }
  flg[1] = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-all_transpose", flg + 1, NULL));
  if (flg[1] && flg[2]) {
    PetscCall(MatTranspose(A[1], MAT_INITIAL_MATRIX, &S));
    PetscCall(MatDestroy(A + 1));
    PetscCall(MatCreateHermitianTranspose(S, A + 1));
    PetscCall(MatDestroy(&S));
  }
  /* global coefficient matrix */
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, A, &S));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, S, S));
  PetscCall(KSPGetPC(ksp, &pc));
  /* outer preconditioner */
  PetscCall(PCSetType(pc, PCFIELDSPLIT));
  PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));
  PetscCall(PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELF, NULL));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCGetType(pc, &type));
  PetscCall(PetscStrcmp(type, PCFIELDSPLIT, flg + 1));
  if (flg[1]) {
    PetscCall(PCSetUp(pc));
    PetscCall(PCFieldSplitGetSubKSP(pc, &n, &subksp));
    PetscCall(KSPGetPC(subksp[0], &pc));
    /* inner preconditioner associated to top-left block */
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscCall(PCSetType(pc, PCHPDDM));
    PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[0], aux[0], NULL, NULL));
#endif
    PetscCall(PCSetFromOptions(pc));
    PetscCall(KSPGetPC(subksp[1], &pc));
    /* inner preconditioner associated to Schur complement, which will be set internally to PCKSP (or PCASM if the Schur complement is centralized on a single process) */
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscCall(PCSetType(pc, PCHPDDM));
    if (!flg[0]) PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[1], aux[1], NULL, NULL));
#endif
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscFree(subksp));
  } else PetscCall(MatSetBlockSize(A[0], 2));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(MatCreateVecs(S, &b, &x));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%s/rhs_%s.dat", dir, id == 3 ? "D" : (id == 2 ? "C" : (id == 1 ? "B" : "A"))));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, prefix, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(b, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(KSPSolve(ksp, b, x));
  flg[1] = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-viewer", flg + 1, NULL));
  if (flg[1]) PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));
  flg[1] = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-successive_solves", flg + 1, NULL));
  if (flg[1]) {
    KSPConvergedReason reason[2];
    PetscInt           iterations[2];
    PetscCall(KSPGetConvergedReason(ksp, reason));
    PetscCall(KSPGetTotalIterations(ksp, iterations));
    PetscCall(KSPMonitorCancel(ksp));
    PetscCall(PetscOptionsClearValue(NULL, "-ksp_monitor"));
    PetscCall(PetscObjectStateIncrease((PetscObject)S));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetUp(pc)); /* update PCFIELDSPLIT submatrices */
    PetscCall(PCFieldSplitGetSubKSP(pc, &n, &subksp));
    PetscCall(KSPGetPC(subksp[0], &pc));
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[0], aux[0], NULL, NULL));
#endif
    PetscCall(PCSetFromOptions(pc));
    PetscCall(KSPGetPC(subksp[1], &pc));
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscCall(PCSetType(pc, PCHPDDM)); /* may have been set to PCKSP internally (or PCASM if the Schur complement is centralized on a single process), so need to enforce the proper PCType */
    if (!flg[0]) PetscCall(PCHPDDMSetAuxiliaryMat(pc, is[1], aux[1], NULL, NULL));
#endif
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscFree(subksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPGetConvergedReason(ksp, reason + 1));
    PetscCall(KSPGetTotalIterations(ksp, iterations + 1));
    iterations[1] -= iterations[0];
    PetscCheck(reason[0] == reason[1] && PetscAbs(iterations[0] - iterations[1]) <= 3, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "Successive calls to KSPSolve() did not converge for the same reason (%s v. %s) or with the same number of iterations (+/- 3, %" PetscInt_FMT " v. %" PetscInt_FMT ")", KSPConvergedReasons[reason[0]], KSPConvergedReasons[reason[1]], iterations[0], iterations[1]);
  }
  PetscCall(VecDestroy(&x));
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

PetscErrorCode MatAndISLoad(const char *prefix, const char *identifier, Mat A, IS is, Mat aux, PetscMPIInt size)
{
  Mat             tmp[3];
  IS              sizes;
  const PetscInt *idx;
  PetscInt        m;
  PetscLayout     map;
  PetscViewer     viewer;
  char            name[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_sizes_%d.dat", prefix, identifier, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes));
  PetscCall(ISLoad(sizes, viewer));
  PetscCall(ISSetBlockSize(sizes, is && aux ? 5 : 4)); /* not mandatory but useful to check for proper sizes */
  PetscCall(ISGetIndices(sizes, &idx));
  PetscCall(MatSetSizes(A, idx[0], idx[1], idx[2], idx[3]));
  if (is && aux) {
    PetscCall(MatCreate(PETSC_COMM_WORLD, tmp));
    PetscCall(MatSetSizes(tmp[0], idx[4], idx[4], PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(tmp[0]));
  }
  PetscCall(ISRestoreIndices(sizes, &idx));
  PetscCall(ISDestroy(&sizes));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s.dat", prefix, identifier));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  if (is && aux) {
    PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes));
    PetscCall(MatGetLayouts(tmp[0], &map, NULL));
    PetscCall(ISSetLayout(sizes, map));
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_is_%d.dat", prefix, identifier, size));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(ISLoad(sizes, viewer));
    PetscCall(ISGetLocalSize(sizes, &m));
    PetscCall(ISGetIndices(sizes, &idx));
    PetscCall(ISSetType(is, ISGENERAL));
    PetscCall(ISGeneralSetIndices(is, m, idx, PETSC_COPY_VALUES));
    PetscCall(ISRestoreIndices(sizes, &idx));
    PetscCall(ISDestroy(&sizes));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s%s_aux_%d.dat", prefix, identifier, size));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(MatLoad(tmp[0], viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MatGetDiagonalBlock(tmp[0], tmp + 1));
    PetscCall(MatDuplicate(tmp[1], MAT_COPY_VALUES, tmp + 2));
    PetscCall(MatHeaderReplace(aux, tmp + 2));
    PetscCall(MatDestroy(tmp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   testset:
      requires: datafilespath hpddm slepc double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
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
        args: -viewer -system stokes -empty_A11 -transpose {{false true}shared output} -permute {{false true}shared output} -fieldsplit_1_pc_hpddm_ksp_pc_side right -fieldsplit_1_pc_hpddm_coarse_mat_type baij -fieldsplit_1_pc_hpddm_levels_1_sub_mat_mumps_icntl_26 1 -explicit {{false true}shared output}
        filter: grep -v -e "action of " -e "                            " -e "block size" -e "total: nonzeros=" -e "using I-node" -e "aij" -e "transpose" -e "diagonal" -e "total number of" -e "                rows=" | sed -e "s/      right preconditioning/      left preconditioning/g" -e "s/      using UNPRECONDITIONED/      using PRECONDITIONED/g"
      test:
        suffix: 1_petsc
        args: -system {{elasticity stokes}separate output} -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_mat_solver_type petsc -fieldsplit_1_pc_hpddm_levels_1_eps_threshold_absolute 0.3 -permute
      test:
        suffix: 2_petsc
        output_file: output/ex87_1_petsc_system-stokes.out
        args: -system stokes -empty_A11 -transpose -fieldsplit_1_pc_hpddm_ksp_pc_side right -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_mat_solver_type petsc -fieldsplit_1_pc_hpddm_coarse_mat_type baij -fieldsplit_1_pc_hpddm_levels_1_eps_threshold_absolute 0.3 -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_shift_type inblocks -successive_solves
        filter: sed -e "s/type: transpose/type: hermitiantranspose/g"
      test:
        suffix: threshold
        requires: !defined(PETSC_HAVE_MKL_SPARSE_SP2M_FEATURE)
        output_file: output/ex87_1_petsc_system-elasticity.out
        args: -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_eps_threshold_absolute 0.2 -fieldsplit_1_pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -successive_solves
   testset:
      requires: datafilespath hpddm slepc double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -ksp_monitor -ksp_rtol 1e-4 -fieldsplit_ksp_max_it 100 -fieldsplit_pc_hpddm_levels_1_st_share_sub_ksp -fieldsplit_pc_hpddm_define_subdomains -fieldsplit_1_pc_hpddm_schur_precondition geneo -fieldsplit_pc_hpddm_coarse_pc_type redundant -fieldsplit_pc_hpddm_coarse_redundant_pc_type cholesky -fieldsplit_pc_hpddm_levels_1_sub_pc_type lu -fieldsplit_ksp_type fgmres -ksp_type fgmres -ksp_max_it 10 -fieldsplit_1_pc_hpddm_coarse_correction balanced -fieldsplit_1_pc_hpddm_levels_1_eps_gen_non_hermitian -fieldsplit_1_pc_hpddm_coarse_p 2 -system stokes -fieldsplit_1_pc_hpddm_ksp_pc_side left -fieldsplit_1_pc_hpddm_levels_1_sub_pc_factor_mat_solver_type petsc -fieldsplit_1_pc_hpddm_levels_1_eps_threshold_absolute 0.3
      test:
        suffix: diagonal
        output_file: output/ex87_1_petsc_system-stokes.out
        args: -fieldsplit_pc_hpddm_levels_1_eps_nev 10 -fieldsplit_0_pc_hpddm_has_neumann -diagonal_A11 {{false true}shared output}
      test:
        suffix: harmonic_overlap_2
        output_file: output/ex87_1_petsc_system-stokes.out
        args: -fieldsplit_0_pc_hpddm_harmonic_overlap 2 -fieldsplit_0_pc_hpddm_levels_1_svd_nsv 20 -diagonal_A11 -permute {{false true}shared output} -all_transpose

   test:
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) !hpddm !memkind
      nsize: 4
      suffix: selfp
      output_file: output/empty.out
      filter: grep -v "CONVERGED_RTOL iterations"
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -system stokes -ksp_rtol 1e-4 -ksp_converged_reason -ksp_max_it 30 -pc_type fieldsplit -pc_fieldsplit_type schur -fieldsplit_ksp_type preonly -pc_fieldsplit_schur_precondition selfp -fieldsplit_pc_type bjacobi -fieldsplit_sub_pc_type lu -transpose {{false true}shared output} -fieldsplit_1_mat_schur_complement_ainv_type lump

   test:
      requires: datafilespath hpddm slepc double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      suffix: nonsymmetric_least_squares
      output_file: output/empty.out
      filter: grep -v "CONVERGED_RTOL iterations"
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -system diffusion -ksp_rtol 1e-4 -ksp_converged_reason -ksp_max_it 20 -pc_type fieldsplit -pc_fieldsplit_type schur -fieldsplit_ksp_type preonly -fieldsplit_0_pc_type jacobi -prefix_push fieldsplit_1_ -pc_hpddm_schur_precondition least_squares -pc_hpddm_define_subdomains -prefix_push pc_hpddm_levels_1_ -sub_pc_type lu -sub_pc_factor_shift_type nonzero -eps_nev 5 -st_share_sub_ksp -prefix_pop -prefix_pop

   test:
      requires: datafilespath hpddm slepc double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 2
      suffix: lagrange
      output_file: output/empty.out
      filter: grep -v "CONVERGED_RTOL iterations"
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -ksp_rtol 1e-4 -fieldsplit_ksp_max_it 100 -fieldsplit_0_pc_hpddm_has_neumann -fieldsplit_0_pc_hpddm_levels_1_eps_nev 10 -fieldsplit_0_pc_hpddm_levels_1_st_share_sub_ksp -fieldsplit_0_pc_hpddm_define_subdomains -fieldsplit_1_pc_hpddm_schur_precondition geneo -fieldsplit_0_pc_hpddm_coarse_pc_type redundant -fieldsplit_0_pc_hpddm_coarse_redundant_pc_type cholesky -fieldsplit_0_pc_hpddm_levels_1_sub_pc_type lu -fieldsplit_ksp_type fgmres -ksp_type fgmres -ksp_max_it 10 -system lagrange -transpose {{false true}shared output} -successive_solves

   test:
      requires: datafilespath mumps double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      suffix: mumps
      output_file: output/empty.out
      args: -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -ksp_type preonly -system elasticity -pc_type cholesky -mat_mumps_icntl_15 1

TEST*/
