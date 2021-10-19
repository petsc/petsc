static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL,ddup = NULL;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* basic creation and destruction */
  ierr = PetscDeviceContextCreate(&dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextDoesNotExist(dctx);CHKERRQ(ierr);
  /* double free is no-op */
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextDoesNotExist(dctx);CHKERRQ(ierr);

  /* test global context returns a valid context */
  dctx = NULL;
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRQ(ierr);
  /* test locally setting to null doesn't clobber the global */
  dctx = NULL;
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRQ(ierr);

  /* test duplicate */
  ierr = PetscDeviceContextDuplicate(dctx,&ddup);CHKERRQ(ierr);
  /* both device contexts should exist */
  ierr = AssertDeviceContextExists(dctx);CHKERRQ(ierr);
  ierr = AssertDeviceContextExists(ddup);CHKERRQ(ierr);

  /* destroying the dup should leave the original untouched */
  ierr = PetscDeviceContextDestroy(&ddup);CHKERRQ(ierr);
  ierr = AssertDeviceContextDoesNotExist(ddup);CHKERRQ(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRQ(ierr);

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
   nsize: {{1 2 4}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip

TEST*/
