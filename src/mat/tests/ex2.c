
static char help[] = "Tests MatTranspose(), MatNorm(), MatAXPY() and MatAYPX().\n\n";

#include <petscmat.h>

static PetscErrorCode TransposeAXPY(Mat C,PetscScalar alpha,Mat mat,PetscErrorCode (*f)(Mat,Mat*))
{
  Mat            D,E,F,G;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr  = f(C,&D);CHKERRQ(ierr);
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = MatAXPY(E,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  /* MATTRANSPOSE should have a MatTranspose_Transpose or MatTranspose_HT implementation */
  if (f == MatCreateTranspose) {
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  } else {
    ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  }
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = MatAXPY(C,alpha,E,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr  = f(C,&D);CHKERRQ(ierr);
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = f(mat,&F);CHKERRQ(ierr);
  ierr  = f(F,&G);CHKERRQ(ierr);
  ierr  = MatAXPY(E,alpha,G,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&G);CHKERRQ(ierr);
  ierr  = MatDestroy(&F);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            mat,tmat = 0;
  PetscInt       m = 7,n,i,j,rstart,rend,rect = 0;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flg;
  PetscScalar    v, alpha;
  PetscReal      normf,normi,norm1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = m;
  ierr = PetscOptionsHasName(NULL,NULL,"-rectA",&flg);CHKERRQ(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = PetscOptionsHasName(NULL,NULL,"-rectB",&flg);CHKERRQ(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix, test MatValid() --------- */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j;
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* ----------------- Test MatNorm()  ----------------- */
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original A: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* --------------- Test MatTranspose()  -------------- */
  ierr = PetscOptionsHasName(NULL,NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,MAT_REUSE_MATRIX,&mat);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr);
  }

  /* ----------------- Test MatNorm()  ----------------- */
  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"B = A^T: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* ----------------- Test MatAXPY(), MatAYPX()  ----------------- */
  if (mat && !rect) {
    alpha = 1.0;
    ierr  = PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL);CHKERRQ(ierr);
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr  = MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAYPX:  B = alpha*B + A\n");CHKERRQ(ierr);
    ierr = MatAYPX(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  {
    Mat C;
    alpha = 1.0;
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
    ierr  = MatAXPY(C,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr  = MatDestroy(&C);CHKERRQ(ierr);
    ierr  = TransposeAXPY(C,alpha,mat,MatCreateTranspose);CHKERRQ(ierr);
    ierr  = TransposeAXPY(C,alpha,mat,MatCreateHermitianTranspose);CHKERRQ(ierr);
  }

  {
    Mat matB;
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    ierr = MatCreate(PETSC_COMM_WORLD,&matB);CHKERRQ(ierr);
    ierr = MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(matB);CHKERRQ(ierr);
    ierr = MatSetUp(matB);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(matB,&rstart,&rend);CHKERRQ(ierr);
    if (rstart % 2 != 0) rstart++;
    for (i=rstart; i<rend; i += 2) {
      for (j=0; j<n; j += 2) {
        v    = 10.0*i+j;
        ierr = MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," A: original matrix:\n");
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," B(a subset of A):\n");
    ierr = MatView(matB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&matB);CHKERRQ(ierr);
  }

  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free data structures */
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}





/*TEST

   test:
      suffix: 11_A
      args: -mat_type seqaij -rectA
      filter: grep -v "Mat Object"

   test:
      suffix: 12_A
      args: -mat_type seqdense -rectA
      filter: grep -v "Mat Object"

   test:
      suffix: 11_B
      args: -mat_type seqaij -rectB
      filter: grep -v "Mat Object"

   test:
      suffix: 12_B
      args: -mat_type seqdense -rectB
      filter: grep -v "Mat Object"

   test:
      suffix: 21
      args: -mat_type mpiaij

   test:
      suffix: 22
      args: -mat_type mpidense

   test:
      suffix: 23
      nsize: 3
      args: -mat_type mpiaij
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 24
      nsize: 3
      args: -mat_type mpidense

   test:
      suffix: 2_aijcusparse_1
      args: -mat_type mpiaijcusparse
      output_file: output/ex2_23.out
      requires: cuda
      filter: grep -v type | grep -v "MPI processes"

   test:
      suffix: 2_aijcusparse_2
      nsize: 3
      args: -mat_type mpiaijcusparse
      output_file: output/ex2_23.out
      requires: cuda
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
