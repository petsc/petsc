static char help[] = "Test MatMatMult() for Dense matrices.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscInt       m,i,k,M=10,N=5;
  PetscScalar    *array;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal;
  PetscReal      fill = 1.0;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DETERMINE,PETSC_DETERMINE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  k = 0;
  for (i=0; i<m*N; i++){
    ierr = PetscRandomGetValue(r,&array[k++]);CHKERRQ(ierr);
  }
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatMatMult() */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr); /* B = A^T */
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB);
  ierr = MatView(B,0);CHKERRQ(ierr);
  ierr = MatView(A,0);CHKERRQ(ierr);
  ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = B*A = A^T*A */
  ierr = MatView(C,0);CHKERRQ(ierr);

  ierr = MatMatMultSymbolic(B,A,fill,&D);CHKERRQ(ierr); /* D = B*A = A^T*A */
  for (i=0; i<2; i++){
    /* Repeat the numeric product to test reuse of the previous symbolic product */
    ierr = MatMatMultNumeric(B,A,D);CHKERRQ(ierr);
  }
  ierr = MatEqual(C,D,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  ierr = MatDestroy(&D);CHKERRQ(ierr);


  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);
  ierr = MatDestroy(&A);
  
  PetscFinalize();
  return(0);
}
