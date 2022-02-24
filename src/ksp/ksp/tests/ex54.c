/*

     Tests MPIDENSE matrix operations MatMultTranspose() with processes with no rows or columns.
     As the matrix is rectangular, least square solution is computed, so KSPLSQR is also tested here.
*/

#include <petscksp.h>

PetscErrorCode fill(Mat m, Vec v)
{
  PetscInt       idxn[3] = {0, 1, 2};
  PetscInt       localRows = 0;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (rank == 1 || rank == 2) localRows = 4;
  if (size == 1) localRows = 8;
  CHKERRQ(MatSetSizes(m, localRows, PETSC_DECIDE, PETSC_DECIDE, 3));
  CHKERRQ(VecSetSizes(v, localRows, PETSC_DECIDE));

  CHKERRQ(MatSetFromOptions(m));
  CHKERRQ(VecSetFromOptions(v));
  CHKERRQ(MatSetUp(m));

  if (size == 1) {
    PetscInt    idxm1[4] = {0, 1, 2, 3};
    PetscScalar values1[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};
    PetscInt    idxm2[4] = {4, 5, 6, 7};
    PetscScalar values2[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    CHKERRQ(MatSetValues(m, 4, idxm1, 3, idxn, values1, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES));

    CHKERRQ(MatSetValues(m, 4, idxm2, 3, idxn, values2, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES));
  } else if (rank == 1) {
    PetscInt    idxm[4] = {0, 1, 2, 3};
    PetscScalar values[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

    CHKERRQ(MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES));
  } else if (rank == 2) {
    PetscInt    idxm[4] = {4, 5, 6, 7};
    PetscScalar values[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    CHKERRQ(MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES));
    CHKERRQ(VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  PetscFunctionReturn(0);
}

int main(int argc, char** argv)
{
  Mat            Q, C, V, A, B;
  Vec            v, a, b, se;
  KSP            QRsolver;
  PC             pc;
  PetscReal      norm;
  PetscInt       m, n;
  PetscBool      exact = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &v));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &Q));
  CHKERRQ(MatSetType(Q, MATDENSE));
  CHKERRQ(fill(Q, v));

  CHKERRQ(MatCreateVecs(Q, &a, NULL));
  CHKERRQ(MatCreateNormalHermitian(Q, &C));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &QRsolver));
  CHKERRQ(KSPGetPC(QRsolver, &pc));
  CHKERRQ(PCSetType(pc, PCNONE));
  CHKERRQ(KSPSetType(QRsolver, KSPLSQR));
  CHKERRQ(KSPSetFromOptions(QRsolver));
  CHKERRQ(KSPSetOperators(QRsolver, Q, Q));
  CHKERRQ(MatViewFromOptions(Q, NULL, "-sys_view"));
  CHKERRQ(VecViewFromOptions(a, NULL, "-rhs_view"));
  CHKERRQ(KSPSolve(QRsolver, v, a));
  CHKERRQ(KSPLSQRGetStandardErrorVec(QRsolver, &se));
  if (se) {
    CHKERRQ(VecViewFromOptions(se, NULL, "-se_view"));
  }
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-exact", &exact, NULL));
  if (exact) {
    CHKERRQ(KSPDestroy(&QRsolver));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(MatConvert(Q, MATAIJ, MAT_INPLACE_MATRIX, &Q));
    CHKERRQ(MatCreateNormalHermitian(Q, &C));
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &QRsolver));
    CHKERRQ(KSPGetPC(QRsolver, &pc));
    CHKERRQ(PCSetType(pc, PCQR));
    CHKERRQ(KSPSetType(QRsolver, KSPLSQR));
    CHKERRQ(KSPSetFromOptions(QRsolver));
    CHKERRQ(KSPSetOperators(QRsolver, Q, C));
    CHKERRQ(VecZeroEntries(a));
    CHKERRQ(KSPSolve(QRsolver, v, a));
    CHKERRQ(MatGetLocalSize(Q, &m, &n));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &V));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &A));
    CHKERRQ(MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &B));
    CHKERRQ(MatSetRandom(V, NULL));
    CHKERRQ(KSPMatSolve(QRsolver, V, A));
    CHKERRQ(KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PCSetType(pc, PCCHOLESKY));
    CHKERRQ(MatDestroy(&C));
    if (!PetscDefined(USE_COMPLEX)) {
      CHKERRQ(MatTransposeMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
    } else {
      Mat Qc;
      CHKERRQ(MatHermitianTranspose(Q, MAT_INITIAL_MATRIX, &Qc));
      CHKERRQ(MatMatMult(Qc, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
      CHKERRQ(MatDestroy(&Qc));
    }
    CHKERRQ(KSPSetOperators(QRsolver, Q, C));
    CHKERRQ(KSPSetFromOptions(QRsolver));
    CHKERRQ(VecDuplicate(a, &b));
    CHKERRQ(KSPSolve(QRsolver, v, b));
    CHKERRQ(KSPMatSolve(QRsolver, V, B));
    CHKERRQ(KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecAXPY(a, -1.0, b));
    CHKERRQ(VecNorm(a, NORM_2, &norm));
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||a-b|| > PETSC_SMALL (%g)", (double)norm);
    CHKERRQ(MatAXPY(A, -1.0, B, SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(A, NORM_FROBENIUS, &norm));
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||A-B|| > PETSC_SMALL (%g)", (double)norm);
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(MatDestroy(&V));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&B));
  }
  CHKERRQ(KSPDestroy(&QRsolver));
  CHKERRQ(VecDestroy(&a));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&Q));

  ierr = PetscFinalize();
  return ierr;
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
