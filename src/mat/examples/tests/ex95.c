static char help[] = "Testing Matrix-Matrix multiplication for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat         A,B;
  MatScalar   a[1];
  int         ierr,size,rank,m,n,i;
  int         prid=0;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-prid",&prid,PETSC_NULL);CHKERRQ(ierr);

  m = n = 10*size;
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DETERMINE,PETSC_DETERMINE,m,n,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);

  a[0] = rank;
  for (i=0; i<m; i++){
    ierr = MatSetValues(A,1,&i,1,&i,a,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (rank == prid){
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] A: \n",rank);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /* Test MatMerge_SeqsToMPI */
  ierr = MatMerge_SeqsToMPI(PETSC_COMM_WORLD,A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  ierr = MatDestroy(B); 
 
  PetscFinalize();
  return(0);
}
