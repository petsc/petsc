static const char help[] = "Tests PetscDeviceContextFork/Join.\n\n";

#include "petscdevicetestcommon.h"

static PetscErrorCode DoFork(PetscDeviceContext parent, PetscInt n, PetscDeviceContext **sub)
{
  PetscDeviceType dtype;
  PetscStreamType stype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetDeviceType(parent, &dtype));
  PetscCall(PetscDeviceContextGetStreamType(parent, &stype));
  PetscCall(PetscDeviceContextFork(parent, n, sub));
  if (n) PetscCheck(*sub, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContextFork() return NULL pointer for %" PetscInt_FMT " children", n);
  for (PetscInt i = 0; i < n; ++i) {
    PetscDeviceType sub_dtype;
    PetscStreamType sub_stype;

    PetscCall(AssertDeviceContextExists((*sub)[i]));
    PetscCall(PetscDeviceContextGetStreamType((*sub)[i], &sub_stype));
    PetscCall(AssertPetscStreamTypesValidAndEqual(sub_stype, stype, "Child stream type %s != parent stream type %s"));
    PetscCall(PetscDeviceContextGetDeviceType((*sub)[i], &sub_dtype));
    PetscCall(AssertPetscDeviceTypesValidAndEqual(sub_dtype, dtype, "Child device type %s != parent device type %s"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestNestedPetscDeviceContextForkJoin(PetscDeviceContext parCtx, PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;

  PetscFunctionBegin;
  PetscValidPointer(sub, 2);
  PetscCall(AssertPetscDeviceContextsValidAndEqual(parCtx, sub[0], "Current global context does not match expected global context"));
  /* create some children from an active child */
  PetscCall(DoFork(sub[1], nsub, &subsub));
  /* join on a sibling to the parent */
  PetscCall(PetscDeviceContextJoin(sub[2], nsub - 2, PETSC_DEVICE_CONTEXT_JOIN_SYNC, &subsub));
  /* join on the grandparent */
  PetscCall(PetscDeviceContextJoin(parCtx, nsub - 2, PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC, &subsub));
  PetscCall(PetscDeviceContextJoin(sub[1], nsub, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &subsub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* test fork-join */
static PetscErrorCode TestPetscDeviceContextForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;

  PetscFunctionBegin;
  /* mostly for valgrind to catch errors */
  PetscCall(DoFork(dctx, n, &sub));
  PetscCall(PetscDeviceContextJoin(dctx, n, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  /* do it twice */
  PetscCall(DoFork(dctx, n, &sub));
  PetscCall(PetscDeviceContextJoin(dctx, n, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));

  /* create some children */
  PetscCall(DoFork(dctx, n + 1, &sub));
  /* test forking within nested function */
  PetscCall(TestNestedPetscDeviceContextForkJoin(sub[0], sub));
  /* join a subset */
  PetscCall(PetscDeviceContextJoin(dctx, n - 1, PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC, &sub));
  /* back to the ether from whence they came */
  PetscCall(PetscDeviceContextJoin(dctx, n + 1, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  MPI_Comm           comm;
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dctx, "local_"));
  PetscCall(PetscDeviceContextSetFromOptions(comm, dctx));
  PetscCall(TestPetscDeviceContextForkJoin(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestPetscDeviceContextForkJoin(dctx));

  PetscCall(TestPetscDeviceContextForkJoin(NULL));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    output_file: ./output/ExitSuccess.out
    nsize: {{1 3}}
    args: -device_enable {{lazy eager}}
    args: -local_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
    test:
      requires: !device
      suffix: host_no_device
    test:
      requires: device
      args: -root_device_context_device_type host
      suffix: host_with_device
    test:
      requires: cuda
      args: -root_device_context_device_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -root_device_context_device_type hip
      suffix: hip

TEST*/
