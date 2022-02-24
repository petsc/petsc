
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  /* Create parallel IS with the rows we want on THIS processor */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow));
  /* Create parallel IS with the rows we want on THIS processor (same as rows for now) */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&iscol));

  CHKERRQ(MatCreateSubMatrix(C,isrow,iscol,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatCreateSubMatrix(C,isrow,iscol,MAT_REUSE_MATRIX,&A));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
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
