static const char help[] = "Tests PetscDeviceContextFork/Join.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

static PetscErrorCode TestNestedPetscDeviceContextForkJoin(PetscDeviceContext parCtx, PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;

  PetscFunctionBegin;
  PetscValidDeviceContext(parCtx,1);
  PetscValidPointer(sub,2);
  CHKERRQ(AssertPetscDeviceContextsValidAndEqual(parCtx,sub[0],"Current global context does not match expected global context"));
  /* create some children from an active child */
  CHKERRQ(PetscDeviceContextFork(sub[1],nsub,&subsub));
  /* join on a sibling to the parent */
  CHKERRQ(PetscDeviceContextJoin(sub[2],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub));
  /* join on the grandparent */
  CHKERRQ(PetscDeviceContextJoin(parCtx,nsub-2,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&subsub));
  CHKERRQ(PetscDeviceContextJoin(sub[1],nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub));
  PetscFunctionReturn(0);
}

/* test fork-join */
static PetscErrorCode TestPetscDeviceContextForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* mostly for valgrind to catch errors */
  CHKERRQ(PetscDeviceContextFork(dctx,n,&sub));
  CHKERRQ(PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));
  /* do it twice */
  CHKERRQ(PetscDeviceContextFork(dctx,n,&sub));
  CHKERRQ(PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));

  /* create some children */
  CHKERRQ(PetscDeviceContextFork(dctx,n+1,&sub));
  /* test forking within nested function */
  CHKERRQ(TestNestedPetscDeviceContextForkJoin(sub[0],sub));
  /* join a subset */
  CHKERRQ(PetscDeviceContextJoin(dctx,n-1,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&sub));
  /* back to the ether from whence they came */
  CHKERRQ(PetscDeviceContextJoin(dctx,n+1,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRQ(PetscDeviceContextCreate(&dctx));
  CHKERRQ(PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD,"local_",dctx));
  CHKERRQ(PetscDeviceContextSetUp(dctx));
  CHKERRQ(TestPetscDeviceContextForkJoin(dctx));
  CHKERRQ(PetscDeviceContextDestroy(&dctx));

  CHKERRQ(PetscDeviceContextGetCurrentContext(&dctx));
  CHKERRQ(TestPetscDeviceContextForkJoin(dctx));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 test:
   TODO: broken in ci
   requires: !device
   suffix: no_device
   filter: Error: grep -E -o -e ".*No support for this operation for this object type" -e ".*PETSc is not configured with device support.*" -e "^\[0\]PETSC ERROR:.*[0-9]{1} [A-z]+\(\)"

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 3}}
   args: -local_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip

TEST*/
