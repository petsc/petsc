static char help[] = "Test VecSetValuesCOO\n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Vec             x,y;
  PetscInt        k;
  const PetscInt  M = 18;
  PetscMPIInt     rank,size;
  PetscBool       equal;
  PetscScalar     *vals;
  PetscBool       ignoreRemote = PETSC_FALSE;

  PetscInt i0[] = {3, 4, 1, 10, 0, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 5, 5,  6, 4, 17, 0, 1, 1, 8, 5, 5,  6, 4, 7, 8, 5};
  PetscInt i1[] = {8, 5, 15, 16, 6, 13, 4, 17, 8,  9,  9, 10, 6, 12, 7, 3, 4, 1, 1, 2, 5, 5,  6, 14, 17, 8,  9,  9, 10, 4,  5, 10, 11, 1, 2};
  PetscInt i2[] = {7, 7, 8, 8,  9, 16, 17,  9, 10, 1, 1, -2, 2, 3, 3, 14, 4, 5, 10, 13,  9,  9, 10, 1, 0, 0, 5, 5, 6, 6, 13, 13, 14, -14, 4, 4, 5, 11, 11, 12, 15, 15, 16};

  struct {
    PetscInt   *i;
    PetscCount n;
  } coo[3] = {{i0,sizeof(i0)/sizeof(PetscInt)}, {i1,sizeof(i1)/sizeof(PetscInt)}, {i2,sizeof(i2)/sizeof(PetscInt)}};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-ignore_remote",&ignoreRemote,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheck(size <= 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test requires at most 3 processes");

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,M));
  PetscCall(VecSetType(x,VECSTANDARD));
  PetscCall(VecSetOption(x,VEC_IGNORE_OFF_PROC_ENTRIES,ignoreRemote));
  PetscCall(VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));

  for (k=0; k<coo[rank].n; k++) {
    PetscScalar val = (PetscScalar)coo[rank].i[k];
    PetscCall(VecSetValues(x,1,&coo[rank].i[k],&val,ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,M));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecSetOption(y,VEC_IGNORE_OFF_PROC_ENTRIES,ignoreRemote));
  PetscCall(VecSetOption(y,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  PetscCall(VecSetPreallocationCOO(y,coo[rank].n,coo[rank].i));

  PetscCall(PetscMalloc1(coo[rank].n,&vals));
  for (k=0; k<coo[rank].n; k++) vals[k] = (PetscScalar)coo[rank].i[k];
  PetscCall(VecSetValuesCOO(y,vals,ADD_VALUES));

  PetscCall(VecEqual(x,y,&equal));

  if (!equal) {
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"VecSetValuesCOO() failed");
  }

  PetscCall(PetscFree(vals));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2 3}}
    args: -ignore_remote {{0 1}}

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -vec_type kokkos

    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda

    test:
      suffix: std
      args: -vec_type standard

TEST*/

