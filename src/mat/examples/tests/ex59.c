
static char help[] = "Tests MatCreateSubmatrix() in parallel.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 3,n = 2,rstart,rend;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow,iscol;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = 2*size;

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  /* Create parallel IS with the rows we want on THIS processor */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow);CHKERRQ(ierr);
  /* Create parallel IS with the rows we want on THIS processor (same as rows for now) */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&iscol);CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(C,isrow,iscol,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(C,isrow,iscol,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

   test:
      suffix: 2_baij
      nsize: 3
      args: -mat_type baij

   test:
      suffix: 2_sbaij
      nsize: 3
      args: -mat_type sbaij

   test:
      suffix: baij
      args: -mat_type baij
      output_file: output/ex59_1_baij.out

   test:
      suffix: sbaij
      args: -mat_type sbaij
      output_file: output/ex59_1_sbaij.out

TEST*/
