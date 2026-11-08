static char help[] = "Tests VecMDot(),VecDot(),VecMTDot(), and VecTDot()\n";

#include <petscvec.h>
#include <petscdevice.h>

static PetscErrorCode TestRepeatedMDot(void)
{
  const PetscStreamType streams[] = {PETSC_STREAM_DEFAULT, PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  const PetscInt        counts[]  = {2, 8, 9};
  PetscDeviceContext    saved, dctx;
  PetscInt              n = 8 * 1024 * 1024, N;
  PetscMPIInt           rank;
  PetscBool             empty = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mdot_size", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mdot_empty_rank", &empty, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (empty && rank) n = 0;
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  for (size_t s = 0; s < PETSC_STATIC_ARRAY_LENGTH(streams); ++s) {
    Vec                x, y, ys[9];
    PetscScalar       *a;
    PetscScalar        dots[9];
    const PetscScalar *ar;

    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[s]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecGetSize(x, &N));
    PetscCall(VecDuplicate(x, &y));
    PetscCall(VecSetPinnedMemoryMin(y, 0));
    PetscCall(VecSet(x, 1));
    PetscCall(VecGetArrayWrite(y, &a));
    for (PetscInt i = 0; i < n; ++i) a[i] = 1;
    PetscCall(VecRestoreArrayWrite(y, &a));
    PetscCall(VecGetArrayReadAndMemType(y, &ar, NULL));
    PetscCall(VecRestoreArrayReadAndMemType(y, &ar));
    for (PetscInt j = 0; j < 9; ++j) ys[j] = y;
    for (size_t c = 0; c < PETSC_STATIC_ARRAY_LENGTH(counts); ++c) {
      PetscInt nv = counts[c];

      // Warm allocations and kernels before making the host copy newer.
      PetscCall(VecMDot(x, nv, ys, dots));
      for (PetscInt pass = 0; pass < 3; ++pass) {
#if PetscDefined(USE_COMPLEX)
        PetscScalar value = PetscCMPLX(pass + 2, 1);
#else
        PetscScalar value = pass + 2;
#endif
        PetscScalar expected = N * PetscConj(value);

        PetscCall(VecGetArrayWrite(y, &a));
        for (PetscInt i = 0; i < n; ++i) a[i] = value;
        PetscCall(VecRestoreArrayWrite(y, &a));
        // The first call needs an upload; the second uses the current device copy.
        for (PetscInt repeat = 0; repeat < 2; ++repeat) {
          PetscCall(VecMDot(x, nv, ys, dots));
          for (PetscInt j = 0; j < nv; ++j)
            PetscCheck(PetscAbsScalar(dots[j] - expected) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(PetscAbsScalar(expected), 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incorrect repeated-input dot product %" PetscInt_FMT " (nv %" PetscInt_FMT ", stream type %d, repeat %" PetscInt_FMT "): error %g", j, nv, (int)streams[s], repeat, (double)PetscAbsScalar(dots[j] - expected));
        }
      }
    }
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&x));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec         *V, t;
  PetscInt     i, j, reps, n = 15, k = 6;
  PetscRandom  rctx;
  PetscScalar *val_dot, *val_mdot, *tval_dot, *tval_mdot;
  PetscBool    test_repeated = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-k", &k, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_repeated_mdot", &test_repeated, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test with %" PetscInt_FMT " random vectors of length %" PetscInt_FMT "\n", k, n));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
#if PetscDefined(USE_COMPLEX)
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
  if (test_repeated) PetscCall(TestRepeatedMDot());
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

      test:
         suffix: repeated_cuda
         nsize: {{1 2}}
         args: -vec_type cuda -test_repeated_mdot
         requires: cuda

      test:
         suffix: repeated_cuda_empty
         nsize: 2
         args: -vec_type cuda -test_repeated_mdot -mdot_empty_rank
         requires: cuda

      test:
         suffix: repeated_hip
         nsize: {{1 2}}
         args: -vec_type hip -test_repeated_mdot
         requires: hip

      test:
         suffix: repeated_hip_empty
         nsize: 2
         args: -vec_type hip -test_repeated_mdot -mdot_empty_rank
         requires: hip
TEST*/
