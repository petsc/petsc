static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

PetscErrorCode Assemble(MPI_Comm comm,PetscInt n,MatType mtype)
{
  Mat            A;
  PetscInt       first,last,i;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetType(A,MATMPISBAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank < size-1) {
    CHKERRQ(MatMPISBAIJSetPreallocation(A,1,1,NULL,1,NULL));
  } else {
    CHKERRQ(MatMPISBAIJSetPreallocation(A,1,2,NULL,0,NULL));
  }
  CHKERRQ(MatGetOwnershipRange(A,&first,&last));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  last--;
  for (i=first; i<=last; i++) {
    CHKERRQ(MatSetValue(A,i,i,2.,INSERT_VALUES));
    if (i != n-1) CHKERRQ(MatSetValue(A,i,n-1,-1.,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  MPI_Comm       comm;
  PetscInt       n = 6;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(Assemble(comm,n,MATMPISBAIJ));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args: -n 1000 -mat_view ascii::ascii_info_detail
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
