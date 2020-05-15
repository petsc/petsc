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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  ierr = VecCreate(PETSC_COMM_WORLD, &diag);CHKERRQ(ierr);
  ierr = VecSetSizes(diag, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(diag);CHKERRQ(ierr);

  ierr = VecSet(diag, (1.0/10.0));CHKERRQ(ierr); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  ierr = MatCreate(PETSC_COMM_WORLD, &A11);CHKERRQ(ierr);
  ierr = MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
  ierr = MatSetType(A11, MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A11, n, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A11, diag, INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecDestroy(&diag);CHKERRQ(ierr);

  /* A12 */
  ierr = MatCreate(PETSC_COMM_WORLD, &A12);CHKERRQ(ierr);
  ierr = MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np);CHKERRQ(ierr);
  ierr = MatSetType(A12, MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A12, np, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      ierr = MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* A21 */
  ierr = MatTranspose(A12, MAT_INITIAL_MATRIX, &A21);CHKERRQ(ierr);

  A22 = NULL;

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = A22;

  ierr = MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A);CHKERRQ(ierr);
  ierr = MatNestSetVecType(A,VECNEST);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Tests MatMissingDiagonal_Nest */
  ierr = MatMissingDiagonal(A,&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s\n",flg ? "true" : "false");CHKERRQ(ierr);
  }

  /* Create vectors */
  ierr = MatCreateVecs(A12, &h, &f);CHKERRQ(ierr);

  ierr = VecSet(f, 1.0);CHKERRQ(ierr);
  ierr = VecSet(h, 0.0);CHKERRQ(ierr);

  /* Create block vector */
  tmp_x[0] = f;
  tmp_x[1] = h;

  ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_x,&b);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b, &x);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);

  ierr = VecNestGetSubVecs(x,NULL,&_tmp_x);CHKERRQ(ierr);

  x1 = _tmp_x[0];
  x2 = _tmp_x[1];

  ierr = PetscPrintf(PETSC_COMM_WORLD, "x1 \n");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = VecView(x1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "x2 \n");CHKERRQ(ierr);
  ierr = VecView(x2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A11);CHKERRQ(ierr);
  ierr = MatDestroy(&A12);CHKERRQ(ierr);
  ierr = MatDestroy(&A21);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&h);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  ierr = VecCreate(PETSC_COMM_WORLD, &diag);CHKERRQ(ierr);
  ierr = VecSetSizes(diag, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(diag);CHKERRQ(ierr);

  ierr = VecSet(diag, (1.0/10.0));CHKERRQ(ierr); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  ierr = MatCreate(PETSC_COMM_WORLD, &A11);CHKERRQ(ierr);
  ierr = MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
  ierr = MatSetType(A11, MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A11, n, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A11, diag, INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecDestroy(&diag);CHKERRQ(ierr);

  /* A12 */
  ierr = MatCreate(PETSC_COMM_WORLD, &A12);CHKERRQ(ierr);
  ierr = MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np);CHKERRQ(ierr);
  ierr = MatSetType(A12, MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A12, np, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      ierr = MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* A21 */
  ierr = MatTranspose(A12, MAT_INITIAL_MATRIX, &A21);CHKERRQ(ierr);

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = NULL;

  ierr = MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A);CHKERRQ(ierr);
  ierr = MatNestSetVecType(A,VECNEST);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create vectors */
  ierr = MatCreateVecs(A, &b, &x);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(b,NULL,&tmp_x);CHKERRQ(ierr);
  f    = tmp_x[0];
  h    = tmp_x[1];

  ierr = VecSet(f, 1.0);CHKERRQ(ierr);
  ierr = VecSet(h, 0.0);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(x,NULL,&tmp_x);CHKERRQ(ierr);
  x1   = tmp_x[0];
  x2   = tmp_x[1];

  ierr = PetscPrintf(PETSC_COMM_WORLD, "x1 \n");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = VecView(x1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "x2 \n");CHKERRQ(ierr);
  ierr = VecView(x2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A11);CHKERRQ(ierr);
  ierr = MatDestroy(&A12);CHKERRQ(ierr);
  ierr = MatDestroy(&A21);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args,(char*)0, help);if (ierr) return ierr;
  ierr = test_solve();CHKERRQ(ierr);
  ierr = test_solve_matgetvecs();CHKERRQ(ierr);
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
