static char help[] = "Test MatCreateNest with block sizes.\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                    A, B, C, mats[2];
  ISLocalToGlobalMapping cmap, rmap;
  const PetscInt         indices[1] = {0};
  PetscMPIInt            size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only coded for one process");
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 2));
  PetscCall(MatSetBlockSizes(A, 1, 2));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 2, 1, indices, PETSC_COPY_VALUES, &cmap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 1, indices, PETSC_COPY_VALUES, &rmap));
  PetscCall(MatSetLocalToGlobalMapping(A, rmap, cmap));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetBlockSizes(A, 1, 1));
  PetscCall(MatSetType(B, MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(B, rmap, rmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetUp(B));
  mats[0] = A;
  mats[1] = B;
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 1, NULL, 2, NULL, mats, &C));
  PetscCall(MatSetUp(C));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C, NULL));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:

TEST*/
