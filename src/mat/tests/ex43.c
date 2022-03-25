
static char help[] = "Saves a dense matrix in a dense format (binary).\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    v;
  PetscInt       i,j,m = 4,n = 4;
  PetscMPIInt    rank,size;
  PetscViewer    viewer;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* PART 1:  Generate matrix, then write it in binary format */

  /* Generate matrix */
  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,m,n,NULL,&C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v    = i*m+j;
      PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer));
  PetscCall(MatView(C,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
