static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

PetscErrorCode Assemble(MPI_Comm comm,PetscInt n,MatType mtype)
{
  Mat            A;
  PetscInt       first,last,i;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetType(A,MATMPISBAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank < size-1) {
    PetscCall(MatMPISBAIJSetPreallocation(A,1,1,NULL,1,NULL));
  } else {
    PetscCall(MatMPISBAIJSetPreallocation(A,1,2,NULL,0,NULL));
  }
  PetscCall(MatGetOwnershipRange(A,&first,&last));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  last--;
  for (i=first; i<=last; i++) {
    PetscCall(MatSetValue(A,i,i,2.,INSERT_VALUES));
    if (i != n-1) PetscCall(MatSetValue(A,i,n-1,-1.,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  MPI_Comm       comm;
  PetscInt       n = 6;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(Assemble(comm,n,MATMPISBAIJ));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args: -n 1000 -mat_view ascii::ascii_info_detail
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
