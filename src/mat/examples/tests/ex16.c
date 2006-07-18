
static char help[] = "Tests MatGetArray().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat              A,B; 
  PetscInt         i,j,m = 6,n = 9,rstart,rend;
  PetscErrorCode   ierr;
  PetscScalar      v,*array;

  PetscInitialize(&argc,&args,(char *)0,help);

  /*
      Create a parallel dense matrix shared by all processors 
  */
  ierr = MatCreateMPIDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL,&A);
        CHKERRQ(ierr);
  /*
     Set values into the matrix 
  */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = 9.0/(i+j+1); ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  /*
       Print the matrix to the screen 
  */
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  /*
      Print the local portion of the matrix to the screen
  */
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%6.4e ",PetscRealPart(array[j*(rend-rstart)+i-rstart]));
    }
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n");
  }
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);

  /*
      Free the space used by the matrix
  */
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
