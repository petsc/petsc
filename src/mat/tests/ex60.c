
static char help[] = "Tests MatGetColumnVector().";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 3,n = 2,Ii,J,col = 0;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  Vec            yy;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-col",&col,NULL));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSeqAIJSetPreallocation(C,5,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(C,5,NULL,5,NULL));
  CHKERRQ(MatSetUp(C));

  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatCreateVecs(C,NULL,&yy));
  CHKERRQ(VecSetFromOptions(yy));

  CHKERRQ(MatGetColumnVector(C,yy,col));

  CHKERRQ(VecView(yy,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&yy));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      args: -col 7

   test:
      suffix: dense
      nsize: 3
      args: -col 7 -mat_type dense -vec_type {{mpi standard}}
      filter: grep -v type

   test:
      requires: cuda
      suffix: dense_cuda
      nsize: 3
      output_file: output/ex60_dense.out
      args: -col 7 -mat_type {{mpidense mpidensecuda}} -vec_type {{mpi standard cuda mpicuda}}
      filter: grep -v type

TEST*/
