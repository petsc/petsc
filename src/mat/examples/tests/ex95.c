static char help[] = "Testing Matrix-Matrix multiplication for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat         A,B;
  MatScalar   a[1],alpha;
  int         ierr,size,rank,m,n,i,col;
  int         prid;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  prid = size;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-prid",&prid,PETSC_NULL);CHKERRQ(ierr);

  m = n = 10*size;
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DETERMINE,PETSC_DETERMINE,m,n,&A);CHKERRQ(ierr);
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
  ierr = MatMerge_SeqsToMPI(PETSC_COMM_WORLD,A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX */
  alpha = 0.1;
  for (i=1; i<4; i++){
    ierr = MatScale(&alpha,A);CHKERRQ(ierr);
    ierr = MatMerge_SeqsToMPI(PETSC_COMM_WORLD,A,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  }
  ierr = MatView(B, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  ierr = MatDestroy(B); 
 
  PetscFinalize();
  return(0);
}
