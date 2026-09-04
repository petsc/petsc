static char help[] = "Tests VecKokkosPlaceArray() and array-less vectors from VecCreate{Seq,MPI}KokkosWithArray().\n\n";

#include <petscvec.h>
#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
  PetscInt     n = 10;
  PetscMPIInt  size;
  Vec          x, y, z;
  PetscReal    norm;
  PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  {
    // Create a VecKokkos x and init it
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
    PetscCall(VecSetType(x, VECKOKKOS));
    PetscCall(VecSet(x, 4.0));

    // Allocate a Kokkos View kv and init it with a different value
    auto kv = Kokkos::View<PetscScalar *>("kv", n);
    PetscCallCXX(Kokkos::deep_copy(kv, 2.0));

    // Use kv's array to replace the device array in x
    PetscCall(VecKokkosPlaceArray(x, kv.data())); // x = {2.0, 2.0, ...}
    PetscCall(VecScale(x, 0.5));                  // x = {1.0, 1.0, ...}
    PetscCall(VecGetArray(x, &array));            // must see the placed array, not the original one
    PetscCheck(array[0] == 1.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "VecGetArray() did not return the array placed with VecKokkosPlaceArray()");
    PetscCall(VecRestoreArray(x, &array));
    PetscCall(VecKokkosResetArray(x)); // x = {4.0, 4.0, ...}, kv = {1,0, 1.0, ...}

    // Create a vector y with kv
    PetscCall(VecCreateMPIKokkosWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, kv.data(), &y));

    // Check both x and y have correct values
    PetscCall(VecAXPY(x, -4.0, y)); // x -= 4 * y
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecKokkosPlaceArray");

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
  }
  {
    // Create an array-less VecKokkos z: no host or device memory is allocated until an array is placed
    if (size == 1) PetscCall(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, n, NULL, &z));
    else PetscCall(VecCreateMPIKokkosWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, NULL, &z));

    // Place a device array, use z on device and on host, and take the array back
    auto kz = Kokkos::View<PetscScalar *>("kz", n);
    PetscCallCXX(Kokkos::deep_copy(kz, 2.0));
    PetscCall(VecKokkosPlaceArray(z, kz.data())); // z = {2.0, 2.0, ...}
    PetscCall(VecScale(z, 0.5));                  // z = {1.0, 1.0, ...}
    PetscCall(VecNorm(z, NORM_1, &norm));
    PetscCheck(PetscAbsReal(norm - (PetscReal)(n * size)) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecKokkosPlaceArray() on an array-less vector");
    PetscCall(VecGetArray(z, &array));
    array[0] = 3.0; // z = {3.0, 1.0, ...}
    PetscCall(VecRestoreArray(z, &array));
    PetscCall(VecKokkosResetArray(z)); // kz = {3.0, 1.0, ...}
    PetscCallCXX(Kokkos::fence());
    {
      auto kz_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kz);

      PetscCheck(kz_h(0) == 3.0 && kz_h(n - 1) == 1.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecKokkosResetArray() on an array-less vector");
    }

    // Place a host array, as MatDenseGetColumnVec() does, use z, and take the array back
    PetscCall(PetscMalloc1(n, &array));
    for (PetscInt i = 0; i < n; i++) array[i] = 1.0;
    PetscCall(VecPlaceArray(z, array));
    PetscCall(VecScale(z, 4.0)); // z = {4.0, 4.0, ...}
    PetscCall(VecNorm(z, NORM_INFINITY, &norm));
    PetscCheck(PetscAbsReal(norm - 4.0) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecPlaceArray() on an array-less vector");
    PetscCall(VecResetArray(z)); // array = {4.0, 4.0, ...}
    PetscCheck(array[0] == 4.0 && array[n - 1] == 4.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecResetArray() on an array-less vector");
    PetscCall(PetscFree(array));

    // Place the device array once more; z must still work after the arrays were reset
    PetscCall(VecKokkosPlaceArray(z, kz.data())); // z = {3.0, 1.0, ...}
    PetscCall(VecNorm(z, NORM_INFINITY, &norm));
    PetscCheck(PetscAbsReal(norm - 3.0) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with a second VecKokkosPlaceArray() on an array-less vector");
    PetscCall(VecKokkosResetArray(z));
    PetscCall(VecDestroy(&z));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: kokkos_kernels
      nsize: {{1 2}}
      output_file: output/empty.out

TEST*/
