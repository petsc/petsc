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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);CHKERRMPI(ierr);

  if (rank == 1 || rank == 2) localRows = 4;
  if (size == 1) localRows = 8;
  ierr = MatSetSizes(m, localRows, PETSC_DECIDE, PETSC_DECIDE, 3);CHKERRQ(ierr);
  ierr = VecSetSizes(v, localRows, PETSC_DECIDE);CHKERRQ(ierr);

  ierr = MatSetFromOptions(m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = MatSetUp(m);CHKERRQ(ierr);

  if (size == 1) {
    PetscInt    idxm1[4] = {0, 1, 2, 3};
    PetscScalar values1[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};
    PetscInt    idxm2[4] = {4, 5, 6, 7};
    PetscScalar values2[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    ierr = MatSetValues(m, 4, idxm1, 3, idxn, values1, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES);CHKERRQ(ierr);

    ierr = MatSetValues(m, 4, idxm2, 3, idxn, values2, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES);CHKERRQ(ierr);
  } else if (rank == 1) {
    PetscInt    idxm[4] = {0, 1, 2, 3};
    PetscScalar values[12] = {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1};

    ierr = MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 0, 1.1, INSERT_VALUES); VecSetValue(v, 1, 2.5, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 2, 3, INSERT_VALUES); VecSetValue(v, 3, 4, INSERT_VALUES);CHKERRQ(ierr);
  } else if (rank == 2) {
    PetscInt    idxm[4] = {4, 5, 6, 7};
    PetscScalar values[12] = {1, 2, 0, 1, 2, 1, 1, 3, 0, 1, 3, 1};

    ierr = MatSetValues(m, 4, idxm, 3, idxn, values, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 4, 5, INSERT_VALUES); VecSetValue(v, 5, 6, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(v, 6, 7, INSERT_VALUES); VecSetValue(v, 7, 8, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
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

  ierr = VecCreate(PETSC_COMM_WORLD, &v);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &Q);CHKERRQ(ierr);
  ierr = MatSetType(Q, MATDENSE);CHKERRQ(ierr);
  ierr = fill(Q, v);CHKERRQ(ierr);

  ierr = MatCreateVecs(Q, &a, NULL);CHKERRQ(ierr);
  ierr = MatCreateNormalHermitian(Q, &C);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &QRsolver);CHKERRQ(ierr);
  ierr = KSPGetPC(QRsolver, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetType(QRsolver, KSPLSQR);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(QRsolver);CHKERRQ(ierr);
  ierr = KSPSetOperators(QRsolver, Q, Q);CHKERRQ(ierr);
  ierr = MatViewFromOptions(Q, NULL, "-sys_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(a, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = KSPSolve(QRsolver, v, a);CHKERRQ(ierr);
  ierr = KSPLSQRGetStandardErrorVec(QRsolver, &se);CHKERRQ(ierr);
  if (se) {
    ierr = VecViewFromOptions(se, NULL, "-se_view");CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL, NULL, "-exact", &exact, NULL);CHKERRQ(ierr);
  if (exact) {
    ierr = KSPDestroy(&QRsolver);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatConvert(Q, MATAIJ, MAT_INPLACE_MATRIX, &Q);CHKERRQ(ierr);
    ierr = MatCreateNormalHermitian(Q, &C);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD, &QRsolver);CHKERRQ(ierr);
    ierr = KSPGetPC(QRsolver, &pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCQR);CHKERRQ(ierr);
    ierr = KSPSetType(QRsolver, KSPLSQR);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(QRsolver);CHKERRQ(ierr);
    ierr = KSPSetOperators(QRsolver, Q, C);CHKERRQ(ierr);
    ierr = VecZeroEntries(a);CHKERRQ(ierr);
    ierr = KSPSolve(QRsolver, v, a);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Q, &m, &n);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &V);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, PETSC_DECIDE, 5, NULL, &A);CHKERRQ(ierr);
    ierr = MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &B);CHKERRQ(ierr);
    ierr = MatSetRandom(V, NULL);CHKERRQ(ierr);
    ierr = KSPMatSolve(QRsolver, V, A);CHKERRQ(ierr);
    ierr = KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCCHOLESKY);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    if (!PetscDefined(USE_COMPLEX)) {
      ierr = MatTransposeMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
    } else {
      Mat Qc;
      ierr = MatHermitianTranspose(Q, MAT_INITIAL_MATRIX, &Qc);CHKERRQ(ierr);
      ierr = MatMatMult(Qc, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
      ierr = MatDestroy(&Qc);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(QRsolver, Q, C);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(QRsolver);CHKERRQ(ierr);
    ierr = VecDuplicate(a, &b);CHKERRQ(ierr);
    ierr = KSPSolve(QRsolver, v, b);CHKERRQ(ierr);
    ierr = KSPMatSolve(QRsolver, V, B);CHKERRQ(ierr);
    ierr = KSPView(QRsolver, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecAXPY(a, -1.0, b);CHKERRQ(ierr);
    ierr = VecNorm(a, NORM_2, &norm);CHKERRQ(ierr);
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||a-b|| > PETSC_SMALL (%g)", (double)norm);
    ierr = MatAXPY(A, -1.0, B, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(A, NORM_FROBENIUS, &norm);CHKERRQ(ierr);
    PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||A-B|| > PETSC_SMALL (%g)", (double)norm);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = MatDestroy(&V);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&QRsolver);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

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
