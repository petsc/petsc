static char help[] = "Test MatSetValuesCOO for MPIAIJKOKKOS mat \n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  Mat             A,B;
  PetscInt        k;
  const PetscInt  M = 18,N = 18;
  PetscMPIInt     rank,size;
  PetscBool       equal;
  PetscScalar     *vals;

  /* Construct 18 x 18 matrices, which are big enough to have complex communication patterns but still small enough for debugging */
  PetscInt i0[] = {7, 7, 8, 8,  9, 16, 17,  9, 10, 1, 1, -2, 2, 3, 3, 14, 4, 5, 10, 13,  9,  9, 10, 1, 0, 0, 5,   5, 6, 6, 13, 13, 14, -14, 4, 4, 5, 11, 11, 12, 15, 15, 16};
  PetscInt j0[] = {1, 6, 2, 4, 10, 15, 13, 16, 11, 2, 7,  3, 8, 4, 9, 13, 5, 2, 15, 14, 10, 16, 11, 2, 0, 1, 6, -11, 0, 7, 15, 17, 11,  13, 5, 8, 2, 12, 17, 13,  3, 16,  9};

  PetscInt i1[] = {8, 5, 15, 16, 6, 13, 4, 17, 8,  9,  9, 10, -6, 12, 7, 3, -4, 1, 1, 2, 5, 5,  6, 14, 17, 8,  9,  9, 10, 4,  5, 10, 11, 1, 2};
  PetscInt j1[] = {2, 3, 16,  9, 5, 17, 1, 13, 4, 10, 16, 11, -5, 12, 1, 7, -1, 2, 7, 3, 6, 11, 0, 11, 13, 4, 10, 16, 11, 8, -2, 15, 12, 7, 3};

  PetscInt i2[] = {3, 4, 1, 10, 0, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 5, 5,  6, 4, 17, 0, 1, 1, 8, 5, 5,  6, 4, 7, 8, 5};
  PetscInt j2[] = {7, 1, 2, 11, 5, 2, 7, 3, 2, 7, 3, 8, 4, 9, 3, 5, 7, 3, 6, 11, 0, 1, 13, 5, 2, 7, 4, 6, 11, 0, 1, 3, 4, 2};

  struct {
    PetscInt *i,*j,n;
  } coo[3] = {{i0,j0,sizeof(i0)/sizeof(PetscInt)}, {i1,j1,sizeof(i1)/sizeof(PetscInt)}, {i2,j2,sizeof(i2)/sizeof(PetscInt)}};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  if (size > 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"This test requires at most 3 processes");

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,2,NULL);
  ierr = MatMPIAIJSetPreallocation(A,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  for (k=0; k<coo[rank].n; k++) {
    PetscScalar val = coo[rank].j[k];
    ierr = MatSetValue(A,coo[rank].i[k],coo[rank].j[k],val,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetPreallocationCOO(B,coo[rank].n,coo[rank].i,coo[rank].j);CHKERRQ(ierr);

  ierr = PetscMalloc1(coo[rank].n,&vals);CHKERRQ(ierr);
  for (k=0; k<coo[rank].n; k++) vals[k] = coo[rank].j[k];
  ierr = MatSetValuesCOO(B,vals,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatEqual(A,B,&equal);CHKERRQ(ierr);

  if (!equal) {
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"MatSetValuesCOO() failed");
  }

  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    nsize: {{1 2 3}}
    requires: kokkos_kernels
    args: -mat_type aijkokkos
    output_file: output/ex254_1.out

TEST*/

