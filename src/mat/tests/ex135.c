static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

PetscErrorCode Assemble(MPI_Comm comm,PetscInt n,MatType mtype)
{
  Mat            A;
  PetscInt       first,last,i;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank < size-1) {
    ierr = MatMPISBAIJSetPreallocation(A,1,1,NULL,1,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatMPISBAIJSetPreallocation(A,1,2,NULL,0,NULL);CHKERRQ(ierr);
  }
  ierr = MatGetOwnershipRange(A,&first,&last);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  last--;
  for (i=first; i<=last; i++) {
    ierr = MatSetValue(A,i,i,2.,INSERT_VALUES);CHKERRQ(ierr);
    if (i != n-1) {ierr = MatSetValue(A,i,n-1,-1.,INSERT_VALUES);CHKERRQ(ierr);}
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       n = 6;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = Assemble(comm,n,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 4
      args: -n 1000 -mat_view ascii::ascii_info_detail
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)

TEST*/
