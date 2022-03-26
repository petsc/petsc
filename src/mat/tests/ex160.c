
static char help[] = "Tests MatMPIBAIJ format in sequential run \n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A,B;
  PetscInt       i,rstart,rend;
  PetscMPIInt    rank,size;
  PetscScalar    v;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a MPIBAIJ matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,32,32));
  PetscCall(MatSetType(A,MATMPIBAIJ));
  PetscCall(MatSeqBAIJSetPreallocation(A,2,2,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(A,2,2,NULL,2,NULL));

  v    = 1.0;
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    PetscCall(MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Convert A to AIJ format */
  PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex160.out

TEST*/
