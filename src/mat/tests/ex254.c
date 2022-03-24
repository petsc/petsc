static char help[] = "Test MatSetValuesCOO for MPIAIJ and its subclasses \n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat             A,B;
  PetscInt        k;
  const PetscInt  M   = 18,N = 18;
  PetscMPIInt     rank,size;
  PetscBool       equal;
  PetscScalar    *vals;
  PetscBool       flg = PETSC_FALSE;

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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ignore_remote",&flg,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheckFalse(size > 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"This test requires at most 3 processes");

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A,2,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,2,NULL,2,NULL));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,flg));

  for (k=0; k<coo[rank].n; k++) {
    PetscScalar val = coo[rank].j[k];
    CHKERRQ(MatSetValue(A,coo[rank].i[k],coo[rank].j[k],val,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetOption(B,MAT_IGNORE_OFF_PROC_ENTRIES,flg));
  CHKERRQ(MatSetPreallocationCOO(B,coo[rank].n,coo[rank].i,coo[rank].j));

  CHKERRQ(PetscMalloc1(coo[rank].n,&vals));
  for (k=0; k<coo[rank].n; k++) vals[k] = coo[rank].j[k];
  CHKERRQ(MatSetValuesCOO(B,vals,ADD_VALUES));

  CHKERRQ(MatEqual(A,B,&equal));

  if (!equal) {
    CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"MatSetValuesCOO() failed");
  }

  CHKERRQ(PetscFree(vals));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/ex254_1.out
    nsize: {{1 2 3}}
    args: -ignore_remote {{0 1}}

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -mat_type aijkokkos

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: aij
      args: -mat_type aij

TEST*/
