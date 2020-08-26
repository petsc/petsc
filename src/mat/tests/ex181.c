
static char help[] = "Tests MatCreateSubmatrix() with entire matrix, modified from ex59.c.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A,Adup;
  PetscInt       i,j,m = 3,n = 2,rstart,rend;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow;
  PetscBool      detect_bug = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsHasName(NULL,NULL,"-detect_bug",&detect_bug);CHKERRQ(ierr);
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
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Generate a new matrix consisting every row and column of the original matrix
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* Create parallel IS with the rows we want on THIS processor */
  if (detect_bug && !rank) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,rstart,1,&isrow);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isrow);CHKERRQ(ierr);
  }
  ierr = MatCreateSubMatrix(C,isrow,NULL,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);

  /* Change C to test the case MAT_REUSE_MATRIX */
  if (!rank) {
    i = 0; j = 0; v = 100;
    ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(C,isrow,NULL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test MatDuplicate */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&Adup);CHKERRQ(ierr);
  ierr = MatDestroy(&Adup);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 2
      filter: grep -v "Mat Object"
      requires: !complex

   test:
      suffix: 2
      nsize: 3
      args: -detect_bug
      filter: grep -v "Mat Object"
      requires: !complex

TEST*/
