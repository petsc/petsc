static const char help[] = "Tests PetscDeviceContextFork/Join.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

static PetscErrorCode TestNestedPetscDeviceContextForkJoin(PetscDeviceContext parCtx, PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(parCtx,1);
  PetscValidPointer(sub,2);
  ierr = AssertPetscDeviceContextsValidAndEqual(parCtx,sub[0],"Current global context does not match expected global context");CHKERRQ(ierr);
  /* create some children from an active child */
  ierr = PetscDeviceContextFork(sub[1],nsub,&subsub);CHKERRQ(ierr);
  /* join on a sibling to the parent */
  ierr = PetscDeviceContextJoin(sub[2],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub);CHKERRQ(ierr);
  /* join on the grandparent */
  ierr = PetscDeviceContextJoin(parCtx,nsub-2,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&subsub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(sub[1],nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* test fork-join */
static PetscErrorCode TestPetscDeviceContextForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* mostly for valgrind to catch errors */
  ierr = PetscDeviceContextFork(dctx,n,&sub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);
  /* do it twice */
  ierr = PetscDeviceContextFork(dctx,n,&sub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);

  /* create some children */
  ierr = PetscDeviceContextFork(dctx,n+1,&sub);CHKERRQ(ierr);
  /* test forking within nested function */
  ierr = TestNestedPetscDeviceContextForkJoin(sub[0],sub);CHKERRQ(ierr);
  /* join a subset */
  ierr = PetscDeviceContextJoin(dctx,n-1,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&sub);CHKERRQ(ierr);
  /* back to the ether from whence they came */
  ierr = PetscDeviceContextJoin(dctx,n+1,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDeviceContextCreate(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD,"local_",dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(dctx);CHKERRQ(ierr);
  ierr = TestPetscDeviceContextForkJoin(dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRQ(ierr);

  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = TestPetscDeviceContextForkJoin(dctx);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX_DIALECT_CXX11)

 test:
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
