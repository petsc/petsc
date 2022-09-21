static char help[] = "Tests VecMDot(),VecDot(),VecMTDot(), and VecTDot()\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec         *V, t;
  PetscInt     i, j, reps, n = 15, k = 6;
  PetscRandom  rctx;
  PetscScalar *val_dot, *val_mdot, *tval_dot, *tval_mdot;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-k", &k, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test with %" PetscInt_FMT " random vectors of length %" PetscInt_FMT "\n", k, n));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscRandomSetInterval(rctx, -1. + 4. * PETSC_i, 1. + 5. * PETSC_i));
#else
  PetscCall(PetscRandomSetInterval(rctx, -1., 1.));
#endif
  PetscCall(VecCreate(PETSC_COMM_WORLD, &t));
  PetscCall(VecSetSizes(t, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(t));
  PetscCall(VecDuplicateVecs(t, k, &V));
  PetscCall(VecSetRandom(t, rctx));
  PetscCall(VecViewFromOptions(t, NULL, "-t_view"));
  PetscCall(PetscMalloc1(k, &val_dot));
  PetscCall(PetscMalloc1(k, &val_mdot));
  PetscCall(PetscMalloc1(k, &tval_dot));
  PetscCall(PetscMalloc1(k, &tval_mdot));
  for (i = 0; i < k; i++) PetscCall(VecSetRandom(V[i], rctx));
  for (reps = 0; reps < 20; reps++) {
    for (i = 1; i < k; i++) {
      PetscCall(VecMDot(t, i, V, val_mdot));
      PetscCall(VecMTDot(t, i, V, tval_mdot));
      for (j = 0; j < i; j++) {
        PetscCall(VecDot(t, V[j], &val_dot[j]));
        PetscCall(VecTDot(t, V[j], &tval_dot[j]));
      }
      /* Check result */
      for (j = 0; j < i; j++) {
        if (PetscAbsScalar(val_mdot[j] - val_dot[j]) / PetscAbsScalar(val_dot[j]) > 1e-5) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", val_mdot[j]=%g, val_dot[j]=%g\n", i, j, (double)PetscAbsScalar(val_mdot[j]), (double)PetscAbsScalar(val_dot[j])));
          break;
        }
        if (PetscAbsScalar(tval_mdot[j] - tval_dot[j]) / PetscAbsScalar(tval_dot[j]) > 1e-5) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", tval_mdot[j]=%g, tval_dot[j]=%g\n", i, j, (double)PetscAbsScalar(tval_mdot[j]), (double)PetscAbsScalar(tval_dot[j])));
          break;
        }
      }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test completed successfully!\n"));
  PetscCall(PetscFree(val_dot));
  PetscCall(PetscFree(val_mdot));
  PetscCall(PetscFree(tval_dot));
  PetscCall(PetscFree(tval_mdot));
  PetscCall(VecDestroyVecs(k, &V));
  PetscCall(VecDestroy(&t));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   testset:
      output_file: output/ex43_1.out

      test:
         suffix: cuda
         args: -vec_type cuda -random_type curand
         requires: cuda

      test:
         suffix: kokkos
         args: -vec_type kokkos
         requires: kokkos_kernels

      test:
         suffix: hip
         args: -vec_type hip
         requires: hip
TEST*/
