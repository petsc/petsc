#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscLogDouble x, y;
  PetscScalar   *A, *B;
  PetscInt       n      = 8000000;
  size_t         nbytes = 2 * (size_t)n * sizeof(PetscScalar); // memory traffic of the copy: read B and write A

  PetscCall(PetscInitialize(&argc, &argv, 0, 0));
  PetscCall(PetscCalloc1(n, &A));
  PetscCall(PetscMalloc1(n, &B));

  for (PetscInt i = 0; i < n; i++) {
    A[i] = i % 61897;
    B[i] = i % 61897;
  }
  /* To take care of paging effects */
  PetscCall(PetscArraycpy(A, B, n));
  PetscCall(PetscTime(&x));
  PetscCall(PetscArraycpy(A, B, n));
  PetscCall(PetscTime(&y));

  fprintf(stdout, "%s : \n", "PetscMemcpy");
  fprintf(stdout, "    %-15s : %e MB/s\n", "Bandwidth", (PetscLogDouble)nbytes / (1.0e6 * (y - x)));

  PetscCall(PetscFree(A));
  PetscCall(PetscFree(B));
  PetscCall(PetscFinalize());
  return 0;
}
