static const char help[] = "Test MatNest solving a linear system\n\n";

#include <petscksp.h>

PetscErrorCode test_solve(void)
{
  Mat            A11, A12,A21,A22, A, tmp[2][2];
  KSP            ksp;
  PC             pc;
  Vec            b,x, f,h, diag, x1,x2;
  Vec            tmp_x[2],*_tmp_x;
  PetscInt       n, np, i,j;
  PetscBool      flg;

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME));

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &diag));
  CHKERRQ(VecSetSizes(diag, PETSC_DECIDE, n));
  CHKERRQ(VecSetFromOptions(diag));

  CHKERRQ(VecSet(diag, (1.0/10.0))); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A11));
  CHKERRQ(MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n));
  CHKERRQ(MatSetType(A11, MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A11, n, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL));
  CHKERRQ(MatDiagonalSet(A11, diag, INSERT_VALUES));

  CHKERRQ(VecDestroy(&diag));

  /* A12 */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A12));
  CHKERRQ(MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np));
  CHKERRQ(MatSetType(A12, MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A12, np, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL));

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      CHKERRQ(MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES));
    }
  }
  CHKERRQ(MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY));

  /* A21 */
  CHKERRQ(MatTranspose(A12, MAT_INITIAL_MATRIX, &A21));

  A22 = NULL;

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = A22;

  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A));
  CHKERRQ(MatNestSetVecType(A,VECNEST));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Tests MatMissingDiagonal_Nest */
  CHKERRQ(MatMissingDiagonal(A,&flg,NULL));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s\n",flg ? "true" : "false"));
  }

  /* Create vectors */
  CHKERRQ(MatCreateVecs(A12, &h, &f));

  CHKERRQ(VecSet(f, 1.0));
  CHKERRQ(VecSet(h, 0.0));

  /* Create block vector */
  tmp_x[0] = f;
  tmp_x[1] = h;

  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_x,&b));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  CHKERRQ(VecDuplicate(b, &x));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPSetType(ksp, KSPGMRES));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc, PCNONE));
  CHKERRQ(KSPSetFromOptions(ksp));

  CHKERRQ(KSPSolve(ksp, b, x));

  CHKERRQ(VecNestGetSubVecs(x,NULL,&_tmp_x));

  x1 = _tmp_x[0];
  x2 = _tmp_x[1];

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "x1 \n"));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "x2 \n"));
  CHKERRQ(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A11));
  CHKERRQ(MatDestroy(&A12));
  CHKERRQ(MatDestroy(&A21));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(VecDestroy(&h));

  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

PetscErrorCode test_solve_matgetvecs(void)
{
  Mat            A11, A12,A21, A;
  KSP            ksp;
  PC             pc;
  Vec            b,x, f,h, diag, x1,x2;
  PetscInt       n, np, i,j;
  Mat            tmp[2][2];
  Vec            *tmp_x;

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME));

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &diag));
  CHKERRQ(VecSetSizes(diag, PETSC_DECIDE, n));
  CHKERRQ(VecSetFromOptions(diag));

  CHKERRQ(VecSet(diag, (1.0/10.0))); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A11));
  CHKERRQ(MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n));
  CHKERRQ(MatSetType(A11, MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A11, n, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL));
  CHKERRQ(MatDiagonalSet(A11, diag, INSERT_VALUES));

  CHKERRQ(VecDestroy(&diag));

  /* A12 */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A12));
  CHKERRQ(MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np));
  CHKERRQ(MatSetType(A12, MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A12, np, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL));

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      CHKERRQ(MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES));
    }
  }
  CHKERRQ(MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY));

  /* A21 */
  CHKERRQ(MatTranspose(A12, MAT_INITIAL_MATRIX, &A21));

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = NULL;

  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A));
  CHKERRQ(MatNestSetVecType(A,VECNEST));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create vectors */
  CHKERRQ(MatCreateVecs(A, &b, &x));
  CHKERRQ(VecNestGetSubVecs(b,NULL,&tmp_x));
  f    = tmp_x[0];
  h    = tmp_x[1];

  CHKERRQ(VecSet(f, 1.0));
  CHKERRQ(VecSet(h, 0.0));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc, PCNONE));
  CHKERRQ(KSPSetFromOptions(ksp));

  CHKERRQ(KSPSolve(ksp, b, x));
  CHKERRQ(VecNestGetSubVecs(x,NULL,&tmp_x));
  x1   = tmp_x[0];
  x2   = tmp_x[1];

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "x1 \n"));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "x2 \n"));
  CHKERRQ(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A11));
  CHKERRQ(MatDestroy(&A12));
  CHKERRQ(MatDestroy(&A21));

  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args,(char*)0, help);if (ierr) return ierr;
  CHKERRQ(test_solve());
  CHKERRQ(test_solve_matgetvecs());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:

    test:
      suffix: 2
      nsize: 2

    test:
      suffix: 3
      nsize: 2
      args: -ksp_monitor_short -ksp_type bicg
      requires: !single

TEST*/
