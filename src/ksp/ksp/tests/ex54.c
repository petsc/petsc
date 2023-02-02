/*

     Tests MPIDENSE matrix operations MatMultTranspose() with processes with no rows or columns.
     As the matrix is rectangular, least square solution is computed, so KSPLSQR is also tested here.
*/

#include <petscksp.h>

PetscErrorCode fill(Mat m, Vec v)
{
  PetscInt    idxn[3]   = {0, 1, 2};
  PetscInt    localRows = 0;
  PetscMPIInt rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (rank == 1 || rank == 2) localRows = 4;
  if (size == 1) localRows = 8;
  PetscCall(MatSetSizes(m, localRows, PETSC_DECIDE, PETSC_DECIDE, 3));
  PetscCall(VecSetSizes(v, localRows, PETSC_DECIDE));

  PetscCall(MatSetFromOptions(m));
  PetscCall(VecSetFromOptions(v));
  PetscCall(MatSetUp(m));

  if (size == 1) {
    PetscInt    idxm1[4]    = {0, 1, 2, 3};
    PetscScalar values1[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};
    PetscInt    idxm2[4]    = {4, 5, 6, 7};
    PetscScalar values2[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    PetscCall(MatSetValues(m, 4, idxm1, 3, idxn, values1, INSERT_VALUES));
    PetscCall(VecSetValue(v, 0, 1.1, INSERT_VALUES));
    PetscCall(VecSetValue(v, 1, 2.5, INSERT_VALUES));
    PetscCall(VecSetValue(v, 2, 3, INSERT_VALUES));
    PetscCall(VecSetValue(v, 3, 4, INSERT_VALUES));

    PetscCall(MatSetValues(m, 4, idxm2, 3, idxn, values2, INSERT_VALUES));
    PetscCall(VecSetValue(v, 4, 5, INSERT_VALUES));
    PetscCall(VecSetValue(v, 5, 6, INSERT_VALUES));
    PetscCall(VecSetValue(v, 6, 7, INSERT_VALUES));
    PetscCall(VecSetValue(v, 7, 8, INSERT_VALUES));
  } else if (rank == 1) {
    PetscInt    idxm[4]    = {0, 1, 2, 3};
    PetscScalar values[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

    PetscCall(MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES));
    PetscCall(VecSetValue(v, 0, 1.1, INSERT_VALUES));
    PetscCall(VecSetValue(v, 1, 2.5, INSERT_VALUES));
    PetscCall(VecSetValue(v, 2, 3, INSERT_VALUES));
    PetscCall(VecSetValue(v, 3, 4, INSERT_VALUES));
  } else if (rank == 2) {
    PetscInt    idxm[4]    = {4, 5, 6, 7};
    PetscScalar values[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    PetscCall(MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES));
    PetscCall(VecSetValue(v, 4, 5, INSERT_VALUES));
    PetscCall(VecSetValue(v, 5, 6, INSERT_VALUES));
    PetscCall(VecSetValue(v, 6, 7, INSERT_VALUES));
    PetscCall(VecSetValue(v, 7, 8, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat       Q, C, V, A, B;
  Vec       v, a, b, se;
  KSP       QRsolver;
  PC        pc;
  PetscReal norm;
  PetscInt  m, n;
  PetscBool exact = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Q));
  PetscCall(MatSetType(Q, MATDENSE));
  PetscCall(fill(Q, v));

  PetscCall(MatCreateVecs(Q, &a, NULL));
  PetscCall(MatCreateNormalHermitian(Q, &C));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &QRsolver));
  PetscCall(KSPGetPC(QRsolver, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetType(QRsolver, KSPLSQR));
  PetscCall(KSPSetFromOptions(QRsolver));
  PetscCall(KSPSetOperators(QRsolver, Q, Q));
  PetscCall(MatViewFromOptions(Q, NULL, "-sys_view"));
  PetscCall(VecViewFromOptions(a, NULL, "-rhs_view"));
  PetscCall(KSPSolve(QRsolver, v, a));
  PetscCall(KSPLSQRGetStandardErrorVec(QRsolver, &se));
  if (se) PetscCall(VecViewFromOptions(se, NULL, "-se_view"));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-exact", &exact, NULL));
  if (exact) {
    PetscCall(KSPDestroy(&QRsolver));
    PetscCall(MatDestroy(&C));
    PetscCall(MatConvert(Q, MATAIJ, MAT_INPLACE_MATRIX, &Q));
    PetscCall(MatCreateNormalHermitian(Q, &C));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &QRsolver));
    PetscCall(KSPGetPC(QRsolver, &pc));
    PetscCall(PCSetType(pc, PCQR));
    PetscCall(KSPSetType(QRsolver, KSPLSQR));
    PetscCall(KSPSetFromOptions(QRsolver));
    PetscCall(KSPSetOperators(QRsolver, Q, C));
    PetscCall(VecZeroEntries(a));
    PetscCall(KSPSolve(QRsolver, v, a));
    PetscCall(MatGetLocalSize(Q, &m, &n));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &V));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &A));
    PetscCall(MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &B));
    PetscCall(MatSetRandom(V, NULL));
    PetscCall(KSPMatSolve(QRsolver, V, A));
    PetscCall(KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PCSetType(pc, PCCHOLESKY));
    PetscCall(MatDestroy(&C));
    if (!PetscDefined(USE_COMPLEX)) {
      PetscCall(MatTransposeMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
    } else {
      Mat Qc;
      PetscCall(MatHermitianTranspose(Q, MAT_INITIAL_MATRIX, &Qc));
      PetscCall(MatMatMult(Qc, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
      PetscCall(MatDestroy(&Qc));
    }
    PetscCall(KSPSetOperators(QRsolver, Q, C));
    PetscCall(KSPSetFromOptions(QRsolver));
    PetscCall(VecDuplicate(a, &b));
    PetscCall(KSPSolve(QRsolver, v, b));
    PetscCall(KSPMatSolve(QRsolver, V, B));
    PetscCall(KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecAXPY(a, -1.0, b));
    PetscCall(VecNorm(a, NORM_2, &norm));
    PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||a-b|| > PETSC_SMALL (%g)", (double)norm);
    PetscCall(MatAXPY(A, -1.0, B, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &norm));
    PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||A-B|| > PETSC_SMALL (%g)", (double)norm);
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&V));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
  }
  PetscCall(KSPDestroy(&QRsolver));
  PetscCall(VecDestroy(&a));
  PetscCall(VecDestroy(&v));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Q));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_monitor_true_residual -ksp_max_it 10 -sys_view -ksp_converged_reason -ksp_view -ksp_lsqr_compute_standard_error -ksp_lsqr_monitor

   test:
      suffix: 2
      nsize: 4
      args: -ksp_monitor_true_residual -ksp_max_it 10 -sys_view -ksp_converged_reason -ksp_view -ksp_lsqr_compute_standard_error -ksp_lsqr_monitor

   test:
      suffix: 3
      nsize: 2
      args: -ksp_monitor_true_residual -ksp_max_it 10 -sys_view -ksp_converged_reason -ksp_view -ksp_lsqr_monitor -ksp_convergence_test lsqr -ksp_lsqr_compute_standard_error -se_view -ksp_lsqr_exact_mat_norm 0

   test:
      suffix: 4
      nsize: 2
      args: -ksp_monitor_true_residual -ksp_max_it 10 -sys_view -ksp_converged_reason -ksp_view -ksp_lsqr_monitor -ksp_convergence_test lsqr -ksp_lsqr_compute_standard_error -se_view -ksp_lsqr_exact_mat_norm 1

   test:
      requires: suitesparse
      suffix: 5
      nsize: 1
      args: -exact

TEST*/
