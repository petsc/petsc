
static char help[] = "Tests MatTranspose(), MatNorm(), MatAXPY() and MatAYPX().\n\n";

#include <petscmat.h>

static PetscErrorCode TransposeAXPY(Mat C,PetscScalar alpha,Mat mat,PetscErrorCode (*f)(Mat,Mat*))
{
  Mat            D,E,F,G;
  MatType        mtype;

  PetscFunctionBegin;
  PetscCall(MatGetType(mat,&mtype));
  if (f == MatCreateTranspose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nMatAXPY:  (C^T)^T = (C^T)^T + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nMatAXPY:  (C^H)^H = (C^H)^H + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
  }
  PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  PetscCall(f(C,&D));
  PetscCall(f(D,&E));
  PetscCall(MatAXPY(E,alpha,mat,SAME_NONZERO_PATTERN));
  PetscCall(MatConvert(E,mtype,MAT_INPLACE_MATRIX,&E));
  PetscCall(MatView(E,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  if (f == MatCreateTranspose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n"));
  }
  if (f == MatCreateTranspose) {
    PetscCall(MatTranspose(mat,MAT_INITIAL_MATRIX,&D));
  } else {
    PetscCall(MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&D));
  }
  PetscCall(f(D,&E));
  PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  PetscCall(MatAXPY(C,alpha,E,SAME_NONZERO_PATTERN));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  if (f == MatCreateTranspose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n"));
  }
  PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,&C));
  PetscCall(f(C,&D));
  PetscCall(f(D,&E));
  PetscCall(f(mat,&F));
  PetscCall(f(F,&G));
  PetscCall(MatAXPY(E,alpha,G,SAME_NONZERO_PATTERN));
  PetscCall(MatConvert(E,mtype,MAT_INPLACE_MATRIX,&E));
  PetscCall(MatView(E,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));

  /* Call f on a matrix that does not implement the transposition */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  Now without the transposition operation\n"));
  PetscCall(MatConvert(mat,MATSHELL,MAT_INITIAL_MATRIX,&C));
  PetscCall(f(C,&D));
  PetscCall(f(D,&E));
  /* XXX cannot use MAT_INPLACE_MATRIX, it leaks mat */
  PetscCall(MatConvert(E,mtype,MAT_INITIAL_MATRIX,&F));
  PetscCall(MatAXPY(F,alpha,mat,SAME_NONZERO_PATTERN));
  PetscCall(MatView(F,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = m;
  PetscCall(PetscOptionsHasName(NULL,NULL,"-rectA",&flg));
  if (flg) {n += 2; rect = 1;}
  PetscCall(PetscOptionsHasName(NULL,NULL,"-rectB",&flg));
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&mat));
  PetscCall(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j+1.0;
      PetscCall(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* ----------------- Test MatNorm()  ----------------- */
  PetscCall(MatNorm(mat,NORM_FROBENIUS,&normf));
  PetscCall(MatNorm(mat,NORM_1,&norm1));
  PetscCall(MatNorm(mat,NORM_INFINITY,&normi));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"original A: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi));
  PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* --------------- Test MatTranspose()  -------------- */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-in_place",&flg));
  if (!rect && flg) {
    PetscCall(MatTranspose(mat,MAT_REUSE_MATRIX,&mat));   /* in-place transpose */
    tmat = mat;
    mat  = NULL;
  } else { /* out-of-place transpose */
    PetscCall(MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat));
  }

  /* ----------------- Test MatNorm()  ----------------- */
  /* Print info about transpose matrix */
  PetscCall(MatNorm(tmat,NORM_FROBENIUS,&normf));
  PetscCall(MatNorm(tmat,NORM_1,&norm1));
  PetscCall(MatNorm(tmat,NORM_INFINITY,&normi));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"B = A^T: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi));
  PetscCall(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));

  /* ----------------- Test MatAXPY(), MatAYPX()  ----------------- */
  if (mat && !rect) {
    alpha = 1.0;
    PetscCall(PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A\n"));
    PetscCall(MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAYPX:  B = alpha*B + A\n"));
    PetscCall(MatAYPX(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatView(tmat,PETSC_VIEWER_STDOUT_WORLD));
  }

  {
    Mat C;
    alpha = 1.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * A, C=A, SAME_NONZERO_PATTERN\n"));
    PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,&C));
    PetscCall(MatAXPY(C,alpha,mat,SAME_NONZERO_PATTERN));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatDestroy(&C));
    PetscCall(TransposeAXPY(C,alpha,mat,MatCreateTranspose));
    PetscCall(TransposeAXPY(C,alpha,mat,MatCreateHermitianTranspose));
  }

  {
    Mat matB;
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&matB));
    PetscCall(MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n));
    PetscCall(MatSetFromOptions(matB));
    PetscCall(MatSetUp(matB));
    PetscCall(MatGetOwnershipRange(matB,&rstart,&rend));
    if (rstart % 2 != 0) rstart++;
    for (i=rstart; i<rend; i += 2) {
      for (j=0; j<n; j += 2) {
        v    = 10.0*i+j+1.0;
        PetscCall(MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," A: original matrix:\n"));
    PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," B(a subset of A):\n"));
    PetscCall(MatView(matB,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n"));
    PetscCall(MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN));
    PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatDestroy(&matB));
  }

  /* Test MatZeroRows */
  j = rstart - 1;
  if (j < 0) j = m-1;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatZeroRows:\n"));
  PetscCall(MatZeroRows(mat,1,&j,0.0,NULL,NULL));
  PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatShift */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatShift: B = B - 2*I\n"));
  PetscCall(MatShift(mat,-2.0));
  PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  /* Free data structures */
  PetscCall(MatDestroy(&mat));
  PetscCall(MatDestroy(&tmat));
  PetscCall(PetscFinalize());
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

   testset:
      args: -rectB
      output_file: output/ex2_12_B.out
      filter: grep -v type | grep -v "Mat Object"

      test:
         requires: cuda
         suffix: 12_B_cuda
         args: -mat_type {{seqdensecuda seqaijcusparse}}

      test:
         requires: kokkos_kernels
         suffix: 12_B_kokkos
         args: -mat_type aijkokkos

      test:
         suffix: 12_B_aij
         args: -mat_type aij
   test:
      suffix: 21
      args: -mat_type mpiaij
      filter: grep -v type | grep -v " MPI process"

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
      filter: grep -v type | grep -v " MPI process"

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
      filter: grep -v type | grep -v " MPI process"

   test:
      suffix: 2_aijkokkos_1
      args: -mat_type aijkokkos
      output_file: output/ex2_21.out
      requires: kokkos_kernels
      filter: grep -v type | grep -v " MPI process"

   test:
      suffix: 2_aijcusparse_2
      nsize: 3
      args: -mat_type mpiaijcusparse
      output_file: output/ex2_23.out
      requires: cuda
      filter: grep -v type | grep -v " MPI process"

   test:
      suffix: 2_aijkokkos_2
      nsize: 3
      args: -mat_type aijkokkos
      output_file: output/ex2_23.out
      # Turn off hip due to intermittent CI failures on hip.txcorp.com. Should re-enable this test when the machine is upgraded.
      requires: !sycl !hip kokkos_kernels
      filter: grep -v type | grep -v " MPI process"

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
      filter: grep -v type | grep -v " MPI process"

   test:
      requires: cuda
      suffix: 4_cuda
      nsize: 2
      output_file: output/ex2_4.out
      args: -mat_type mpidensecuda -rectA
      filter: grep -v type | grep -v " MPI process"

   test:
      suffix: aijcusparse_1
      args: -mat_type seqaijcusparse -rectA
      filter: grep -v "Mat Object"
      output_file: output/ex2_11_A_aijcusparse.out
      requires: cuda

TEST*/
