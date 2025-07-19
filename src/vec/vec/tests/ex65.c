const char help[] = "Test VecGetLocalVector()";

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec                global, global_copy, local;
  PetscMPIInt        rank;
  PetscMemType       memtype;
  PetscScalar       *array;
  PetscInt           N = 10;
  const PetscScalar *copy_array;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &global));
  PetscCall(PetscObjectSetName((PetscObject)global, "global"));
  PetscCall(VecSetType(global, VECMPICUDA));
  PetscCall(VecSetSizes(global, rank == 0 ? N : 0, N));
  PetscCall(VecSetRandom(global, NULL));
  PetscCall(VecDuplicate(global, &global_copy));
  PetscCall(VecCopy(global, global_copy));

  PetscCall(VecGetArrayRead(global_copy, &copy_array));
  PetscCall(VecGetArrayAndMemType(global, &array, &memtype));
  PetscCall(VecRestoreArrayAndMemType(global, &array));
  if (rank == 0) {
    PetscOffloadMask mask;

    PetscCall(VecGetOffloadMask(global, &mask));
    PetscCheck(mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected offload state");
  }

  PetscCall(VecCreateLocalVector(global, &local));
  PetscCall(PetscObjectSetName((PetscObject)local, "local"));
  PetscCall(VecGetLocalVector(global, local));
  if (rank == 0) {
    const PetscScalar *local_array;
    PetscOffloadMask   mask;

    PetscCall(VecGetOffloadMask(local, &mask));
    PetscCheck(mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local vector has synced with host");
    PetscCall(VecGetOffloadMask(global, &mask));
    PetscCheck(mask == PETSC_OFFLOAD_GPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global vector has synced with host");

    PetscCall(VecGetArrayRead(local, &local_array));
    for (PetscInt i = 0; i < N; i++) {
      PetscCheck(copy_array[i] == local_array[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecGetLocalVector() value mismatch: local[%" PetscInt_FMT "] = %g, global[%" PetscInt_FMT "] = %g", i, (double)PetscRealPart(local_array[i]), i, PetscRealPart(copy_array[i]));
    }
    PetscCall(VecRestoreArrayRead(local, &local_array));
  }
  PetscCall(VecRestoreLocalVector(global, local));
  PetscCall(VecDestroy(&local));
  PetscCall(VecRestoreArrayRead(global_copy, &copy_array));
  PetscCall(VecDestroy(&global_copy));
  PetscCall(VecDestroy(&global));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: 0
    output_file: output/empty.out

TEST*/
