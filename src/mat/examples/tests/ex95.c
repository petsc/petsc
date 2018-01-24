static char help[] = "Testing MatCreateMPIAIJSumSeqAIJ().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  MatScalar      a[1],alpha;
  PetscMPIInt    size,rank;
  PetscInt       m,n,i,col, prid;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  prid = size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prid",&prid,NULL);CHKERRQ(ierr);

  m    = n = 10*size;
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DETERMINE,PETSC_DETERMINE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  a[0] = rank+1;
  for (i=0; i<m-rank; i++) {
    col  = i+rank;
    ierr = MatSetValues(A,1,&i,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (rank == prid) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] A: \n",rank);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /* Test MatCreateMPIAIJSumSeqAIJ */
  ierr = MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX */
  alpha = 0.1;
  for (i=0; i<3; i++) {
    ierr = MatScale(A,alpha);CHKERRQ(ierr);
    ierr = MatCreateMPIAIJSumSeqAIJ(PETSC_COMM_WORLD,A,PETSC_DECIDE,PETSC_DECIDE,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  }
  ierr = MatView(B, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 3
      filter: grep -v "MPI processes"

   test:
      suffix: 2
      filter: grep -v "MPI processes"

TEST*/
