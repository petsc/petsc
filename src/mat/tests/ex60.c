
static char help[] = "Tests MatGetColumnVector().";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 3,n = 2,Ii,J,col = 0;
  PetscMPIInt    size,rank;
  PetscScalar    v;
  Vec            yy;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-col",&col,NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSeqAIJSetPreallocation(C,5,NULL));
  PetscCall(MatMPIAIJSetPreallocation(C,5,NULL,5,NULL));
  PetscCall(MatSetUp(C));

  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateVecs(C,NULL,&yy));
  PetscCall(VecSetFromOptions(yy));

  PetscCall(MatGetColumnVector(C,yy,col));

  PetscCall(VecView(yy,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&yy));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
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
