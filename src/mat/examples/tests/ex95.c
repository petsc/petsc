static char help[] = "Testing MatMerge_SeqsToMPI().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) 
{
  Mat            A,B;
  MatScalar      a[1],alpha;
  PetscMPIInt    size,rank;
  PetscInt       m,n,i,col, prid;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  prid = size;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-prid",&prid,PETSC_NULL);CHKERRQ(ierr);

  m = n = 10*size;
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DETERMINE,PETSC_DETERMINE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);

  a[0] = rank+1;
  for (i=0; i<m-rank; i++){
    col = i+rank;
    ierr = MatSetValues(A,1,&i,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (rank == prid){
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] A: \n",rank);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /* Test MatMerge_SeqsToMPI */
  ierr = MatMerge_SeqsToMPI(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX */
  alpha = 0.1;
  for (i=0; i<3; i++){
    ierr = MatScale(A,alpha);CHKERRQ(ierr);
    ierr = MatMerge_SeqsToMPI(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  }
  ierr = MatView(B, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
 
  PetscFinalize();
  return(0);
}
