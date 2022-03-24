
static char help[] = "Tests MatTranspose(), MatNorm(), MatAXPY() and MatAYPX().\n\n";

#include <petscmat.h>

static PetscErrorCode TransposeAXPY(Mat C,PetscScalar alpha,Mat mat,PetscErrorCode (*f)(Mat,Mat*))
{
  Mat            D,E,F,G;
  MatType        mtype;

  PetscFunctionBegin;
  CHKERRQ(MatGetType(mat,&mtype));
  if (f == MatCreateTranspose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMatAXPY:  (C^T)^T = (C^T)^T + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMatAXPY:  (C^H)^H = (C^H)^H + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
  }
  CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  CHKERRQ(f(C,&D));
  CHKERRQ(f(D,&E));
  CHKERRQ(MatAXPY(E,alpha,mat,SAME_NONZERO_PATTERN));
  CHKERRQ(MatConvert(E,mtype,MAT_INPLACE_MATRIX,&E));
  CHKERRQ(MatView(E,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&E));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&C));
  if (f == MatCreateTranspose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n"));
  }
  if (f == MatCreateTranspose) {
    CHKERRQ(MatTranspose(mat,MAT_INITIAL_MATRIX,&D));
  } else {
    CHKERRQ(MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&D));
  }
  CHKERRQ(f(D,&E));
  CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  CHKERRQ(MatAXPY(C,alpha,E,SAME_NONZERO_PATTERN));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&E));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&C));
  if (f == MatCreateTranspose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n"));
  }
  CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  CHKERRQ(f(C,&D));
  CHKERRQ(f(D,&E));
  CHKERRQ(f(mat,&F));
  CHKERRQ(f(F,&G));
  CHKERRQ(MatAXPY(E,alpha,G,SAME_NONZERO_PATTERN));
  CHKERRQ(MatConvert(E,mtype,MAT_INPLACE_MATRIX,&E));
  CHKERRQ(MatView(E,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&G));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&E));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&C));

  /* Call f on a matrix that does not implement the transposition */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  Now without the transposition operation\n"));
  CHKERRQ(MatConvert(mat,MATSHELL,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(f(C,&D));
  CHKERRQ(f(D,&E));
  /* XXX cannot use MAT_INPLACE_MATRIX, it leaks mat */
  CHKERRQ(MatConvert(E,mtype,MAT_INITIAL_MATRIX,&F));
  CHKERRQ(MatAXPY(F,alpha,mat,SAME_NONZERO_PATTERN));
  CHKERRQ(MatView(F,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&E));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&C));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            mat,tmat = 0;
  PetscInt       m = 7,n,i,j,rstart,rend,rect = 0;
  PetscMPIInt    size,rank;
  PetscBool      flg;
  PetscScalar    v, alpha;
  PetscReal      normf,normi,norm1;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = m;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-rectA",&flg));
  if (flg) {n += 2; rect = 1;}
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-rectB",&flg));
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix --------- */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(mat));
  CHKERRQ(MatSetUp(mat));
  CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j+1.0;
      CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* ----------------- Test MatNorm()  ----------------- */
  CHKERRQ(MatNorm(mat,NORM_FROBENIUS,&normf));
  CHKERRQ(MatNorm(mat,NORM_1,&norm1));
  CHKERRQ(MatNorm(mat,NORM_INFINITY,&normi));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"original A: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* --------------- Test MatTranspose()  -------------- */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-in_place",&flg));
  if (!rect && flg) {
    CHKERRQ(MatTranspose(mat,MAT_REUSE_MATRIX,&mat));   /* in-place transpose */
    tmat = mat;
    mat  = NULL;
  } else { /* out-of-place transpose */
    CHKERRQ(MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat));
  }

  /* ----------------- Test MatNorm()  ----------------- */
  /* Print info about transpose matrix */
  CHKERRQ(MatNorm(tmat,NORM_FROBENIUS,&normf));
  CHKERRQ(MatNorm(tmat,NORM_1,&norm1));
  CHKERRQ(MatNorm(tmat,NORM_INFINITY,&normi));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B = A^T: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi));
  CHKERRQ(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));

  /* ----------------- Test MatAXPY(), MatAYPX()  ----------------- */
  if (mat && !rect) {
    alpha = 1.0;
    CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A\n"));
    CHKERRQ(MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAYPX:  B = alpha*B + A\n"));
    CHKERRQ(MatAYPX(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));
  }

  {
    Mat C;
    alpha = 1.0;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
    CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&C));
    CHKERRQ(MatAXPY(C,alpha,mat,SAME_NONZERO_PATTERN));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(TransposeAXPY(C,alpha,mat,MatCreateTranspose));
    CHKERRQ(TransposeAXPY(C,alpha,mat,MatCreateHermitianTranspose));
  }

  {
    Mat matB;
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&matB));
    CHKERRQ(MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n));
    CHKERRQ(MatSetFromOptions(matB));
    CHKERRQ(MatSetUp(matB));
    CHKERRQ(MatGetOwnershipRange(matB,&rstart,&rend));
    if (rstart % 2 != 0) rstart++;
    for (i=rstart; i<rend; i += 2) {
      for (j=0; j<n; j += 2) {
        v    = 10.0*i+j+1.0;
        CHKERRQ(MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," A: original matrix:\n"));
    CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," B(a subset of A):\n"));
    CHKERRQ(MatView(matB,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n"));
    CHKERRQ(MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN));
    CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(MatDestroy(&matB));
  }

  /* Test MatZeroRows */
  j = rstart - 1;
  if (j < 0) j = m-1;
  CHKERRQ(MatZeroRows(mat,1,&j,0.0,NULL,NULL));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  /* Free data structures */
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(MatDestroy(&tmat));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 11_A
      args: -mat_type seqaij -rectA
      filter: grep -v "Mat Object"

   test:
      suffix: 12_A
      args: -mat_type seqdense -rectA
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: cuda
      suffix: 12_A_cuda
      args: -mat_type seqdensecuda -rectA
      output_file: output/ex2_12_A.out
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: kokkos_kernels
      suffix: 12_A_kokkos
      args: -mat_type aijkokkos -rectA
      output_file: output/ex2_12_A.out
      filter: grep -v type | grep -v "Mat Object"

   test:
      suffix: 11_B
      args: -mat_type seqaij -rectB
      filter: grep -v "Mat Object"

   test:
      suffix: 12_B
      args: -mat_type seqdense -rectB
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: cuda
      suffix: 12_B_cuda
      args: -mat_type seqdensecuda -rectB
      output_file: output/ex2_12_B.out
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: kokkos_kernels
      suffix: 12_B_kokkos
      args: -mat_type aijkokkos -rectB
      output_file: output/ex2_12_B.out
      filter: grep -v type | grep -v "Mat Object"

   test:
      suffix: 21
      args: -mat_type mpiaij
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 22
      args: -mat_type mpidense
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: cuda
      suffix: 22_cuda
      output_file: output/ex2_22.out
      args: -mat_type mpidensecuda
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: kokkos_kernels
      suffix: 22_kokkos
      output_file: output/ex2_22.out
      args: -mat_type aijkokkos
      filter: grep -v type | grep -v "Mat Object"

   test:
      suffix: 23
      nsize: 3
      args: -mat_type mpiaij
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 24
      nsize: 3
      args: -mat_type mpidense
      filter: grep -v type | grep -v "Mat Object"

   test:
      requires: cuda
      suffix: 24_cuda
      nsize: 3
      output_file: output/ex2_24.out
      args: -mat_type mpidensecuda
      filter: grep -v type | grep -v "Mat Object"

   test:
      suffix: 2_aijcusparse_1
      args: -mat_type mpiaijcusparse
      output_file: output/ex2_21.out
      requires: cuda
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 2_aijkokkos_1
      args: -mat_type aijkokkos
      output_file: output/ex2_21.out
      requires: kokkos_kernels
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 2_aijcusparse_2
      nsize: 3
      args: -mat_type mpiaijcusparse
      output_file: output/ex2_23.out
      requires: cuda
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 2_aijkokkos_2
      nsize: 3
      args: -mat_type aijkokkos
      output_file: output/ex2_23.out
      # Turn off hip due to intermittent CI failures on hip.txcorp.com. Should re-enable this test when the machine is upgraded.
      requires: !sycl !hip kokkos_kernels
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 3
      nsize: 2
      args: -mat_type mpiaij -rectA

   test:
      suffix: 3_aijcusparse
      nsize: 2
      args: -mat_type mpiaijcusparse -rectA
      requires: cuda

   test:
      suffix: 4
      nsize: 2
      args: -mat_type mpidense -rectA
      filter: grep -v type | grep -v "MPI processes"

   test:
      requires: cuda
      suffix: 4_cuda
      nsize: 2
      output_file: output/ex2_4.out
      args: -mat_type mpidensecuda -rectA
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: aijcusparse_1
      args: -mat_type seqaijcusparse -rectA
      filter: grep -v "Mat Object"
      output_file: output/ex2_11_A_aijcusparse.out
      requires: cuda

   test:
      suffix: aijcusparse_2
      args: -mat_type seqaijcusparse -rectB
      filter: grep -v "Mat Object"
      output_file: output/ex2_11_B_aijcusparse.out
      requires: cuda

TEST*/
