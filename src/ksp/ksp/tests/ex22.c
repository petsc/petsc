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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME));

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &diag));
  PetscCall(VecSetSizes(diag, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(diag));

  PetscCall(VecSet(diag, (1.0/10.0))); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A11));
  PetscCall(MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A11, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A11, n, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL));
  PetscCall(MatDiagonalSet(A11, diag, INSERT_VALUES));

  PetscCall(VecDestroy(&diag));

  /* A12 */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A12));
  PetscCall(MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np));
  PetscCall(MatSetType(A12, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A12, np, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL));

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      PetscCall(MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES));
    }
  }
  PetscCall(MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY));

  /* A21 */
  PetscCall(MatTranspose(A12, MAT_INITIAL_MATRIX, &A21));

  A22 = NULL;

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = A22;

  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A));
  PetscCall(MatNestSetVecType(A,VECNEST));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Tests MatMissingDiagonal_Nest */
  PetscCall(MatMissingDiagonal(A,&flg,NULL));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Unexpected %s\n",flg ? "true" : "false"));
  }

  /* Create vectors */
  PetscCall(MatCreateVecs(A12, &h, &f));

  PetscCall(VecSet(f, 1.0));
  PetscCall(VecSet(h, 0.0));

  /* Create block vector */
  tmp_x[0] = f;
  tmp_x[1] = h;

  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_x,&b));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecDuplicate(b, &x));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecNestGetSubVecs(x,NULL,&_tmp_x));

  x1 = _tmp_x[0];
  x2 = _tmp_x[1];

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x1 \n"));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x2 \n"));
  PetscCall(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A11));
  PetscCall(MatDestroy(&A12));
  PetscCall(MatDestroy(&A21));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&h));

  PetscCall(MatDestroy(&A));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s \n", PETSC_FUNCTION_NAME));

  n  = 3;
  np = 2;
  /* Create matrices */
  /* A11 */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &diag));
  PetscCall(VecSetSizes(diag, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(diag));

  PetscCall(VecSet(diag, (1.0/10.0))); /* so inverse = diag(10) */

  /* As a test, create a diagonal matrix for A11 */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A11));
  PetscCall(MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A11, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A11, n, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A11, np, NULL,np, NULL));
  PetscCall(MatDiagonalSet(A11, diag, INSERT_VALUES));

  PetscCall(VecDestroy(&diag));

  /* A12 */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A12));
  PetscCall(MatSetSizes(A12, PETSC_DECIDE, PETSC_DECIDE, n, np));
  PetscCall(MatSetType(A12, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A12, np, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A12, np, NULL,np, NULL));

  for (i=0; i<n; i++) {
    for (j=0; j<np; j++) {
      PetscCall(MatSetValue(A12, i,j, (PetscScalar)(i+j*n), INSERT_VALUES));
    }
  }
  PetscCall(MatSetValue(A12, 2,1, (PetscScalar)(4), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A12, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A12, MAT_FINAL_ASSEMBLY));

  /* A21 */
  PetscCall(MatTranspose(A12, MAT_INITIAL_MATRIX, &A21));

  /* Create block matrix */
  tmp[0][0] = A11;
  tmp[0][1] = A12;
  tmp[1][0] = A21;
  tmp[1][1] = NULL;

  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&tmp[0][0],&A));
  PetscCall(MatNestSetVecType(A,VECNEST));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create vectors */
  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(VecNestGetSubVecs(b,NULL,&tmp_x));
  f    = tmp_x[0];
  h    = tmp_x[1];

  PetscCall(VecSet(f, 1.0));
  PetscCall(VecSet(h, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecNestGetSubVecs(x,NULL,&tmp_x));
  x1   = tmp_x[0];
  x2   = tmp_x[1];

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x1 \n"));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x2 \n"));
  PetscCall(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A11));
  PetscCall(MatDestroy(&A12));
  PetscCall(MatDestroy(&A21));

  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{

  PetscCall(PetscInitialize(&argc, &args,(char*)0, help));
  PetscCall(test_solve());
  PetscCall(test_solve_matgetvecs());
  PetscCall(PetscFinalize());
  return 0;
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
