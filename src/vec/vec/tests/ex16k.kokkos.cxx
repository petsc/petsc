static char help[] = "Tests VecKokkosPlaceArray().\n\n";

#include <petscvec.h>
#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
  PetscInt  n = 10;
  Vec       x, y;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
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
    PetscCall(VecKokkosResetArray(x));            // x = {4.0, 4.0, ...}, kv = {1,0, 1.0, ...}

    // Create a vector y with kv
    PetscCall(VecCreateMPIKokkosWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, kv.data(), &y));

    // Check both x and y have correct values
    PetscCall(VecAXPY(x, -4.0, y)); // x -= 4 * y
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test failed with VecKokkosPlaceArray");

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
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
