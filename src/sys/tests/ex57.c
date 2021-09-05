static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include <petscdevice.h>

/* test duplication creates the same object type */
static PetscErrorCode testDuplicate(PetscDeviceContext dctx)
{
  PetscStreamType    stype,dupSType;
  PetscDeviceContext dtmp,ddup;
  PetscDevice        device,dupDevice;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetStreamType(dctx,&stype);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetDevice(dctx,&device);CHKERRQ(ierr);

  /* create manually first */
  ierr = PetscDeviceContextCreate(&dtmp);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetDevice(dtmp,device);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetStreamType(dtmp,stype);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(dtmp);CHKERRQ(ierr);

  /* duplicate */
  ierr = PetscDeviceContextDuplicate(dctx,&ddup);CHKERRQ(ierr);

  ierr = PetscDeviceContextGetDevice(ddup,&dupDevice);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetDevice(dtmp,&device);CHKERRQ(ierr);
  if (device != dupDevice) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"PetscDeviceContextDevices do not match");

  ierr = PetscDeviceContextGetStreamType(ddup,&dupSType);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dtmp,&stype);CHKERRQ(ierr);
  if (dupSType != stype) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamTypes %d and %d do not match",dupSType,stype);

  ierr = PetscDeviceContextDestroy(&dtmp);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&ddup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode testNestedForkJoin(PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;
  PetscDeviceContext  parCtx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetCurrentContext(&parCtx);CHKERRQ(ierr);
  if (parCtx != sub[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Current global context does not match expected global context");
  ierr = PetscDeviceContextFork(parCtx,nsub,&subsub);CHKERRQ(ierr);
  /* join on a different sub */
  ierr = PetscDeviceContextJoin(sub[1],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(parCtx,nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* test fork-join */
static PetscErrorCode testForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* mostly for valgrind to catch errors */
  ierr = PetscDeviceContextFork(dctx,n,&sub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);

  /* create some children */
  ierr = PetscDeviceContextFork(dctx,n+1,&sub);CHKERRQ(ierr);

  /* make the first child the new current context, and test forking within nested function */
  ierr = PetscDeviceContextSetCurrentContext(sub[0]);CHKERRQ(ierr);
  ierr = testNestedForkJoin(sub);CHKERRQ(ierr);
  /* should always reset global context when finished */
  ierr = PetscDeviceContextSetCurrentContext(dctx);CHKERRQ(ierr);

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

  /* Initialize the root */
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);

  /* tests */
  ierr = testDuplicate(dctx);CHKERRQ(ierr);
  ierr = testForkJoin(dctx);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: defined(PETSC_HAVE_CXX_DIALECT_CXX11) && defined(PETSC_EXPERIMENTAL)

  test:
    requires: cuda
    suffix: cuda

  test:
    requires: hip
    suffix: hip
TEST*/
