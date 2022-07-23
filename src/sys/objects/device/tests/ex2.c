static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL,ddup = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  /* basic creation and destruction */
  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(AssertDeviceContextDoesNotExist(dctx));
  /* double free is no-op */
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(AssertDeviceContextDoesNotExist(dctx));

  /* test global context returns a valid context */
  dctx = NULL;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));
  /* test locally setting to null doesn't clobber the global */
  dctx = NULL;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(AssertDeviceContextExists(dctx));

  /* test duplicate */
  PetscCall(PetscDeviceContextDuplicate(dctx,&ddup));
  /* both device contexts should exist */
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(AssertDeviceContextExists(ddup));

  /* destroying the dup should leave the original untouched */
  PetscCall(PetscDeviceContextDestroy(&ddup));
  PetscCall(AssertDeviceContextDoesNotExist(ddup));
  PetscCall(AssertDeviceContextExists(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
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
   nsize: {{1 2 4}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip

TEST*/
