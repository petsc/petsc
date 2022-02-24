static const char help[] = "Tests creation and destruction of PetscDevice.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  const PetscInt n = 10;
  PetscDevice    device = NULL;
  PetscDevice    devices[n];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* normal create and destroy */
  CHKERRQ(PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device));
  CHKERRQ(AssertDeviceExists(device));
  CHKERRQ(PetscDeviceDestroy(&device));
  CHKERRQ(AssertDeviceDoesNotExist(device));
  /* should not destroy twice */
  CHKERRQ(PetscDeviceDestroy(&device));
  CHKERRQ(AssertDeviceDoesNotExist(device));

  /* test reference counting */
  device = NULL;
  CHKERRQ(PetscArrayzero(devices,n));
  CHKERRQ(PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device));
  CHKERRQ(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    CHKERRQ(PetscDeviceReference_Internal(device));
    devices[i] = device;
  }
  CHKERRQ(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    CHKERRQ(PetscDeviceDestroy(&devices[i]));
    CHKERRQ(AssertDeviceExists(device));
    CHKERRQ(AssertDeviceDoesNotExist(devices[i]));
  }
  CHKERRQ(PetscDeviceDestroy(&device));
  CHKERRQ(AssertDeviceDoesNotExist(device));

  /* test the default devices exist */
  device = NULL;
  CHKERRQ(PetscArrayzero(devices,n));
  {
    PetscDeviceContext dctx;
    /* global context will have the default device */
    CHKERRQ(PetscDeviceContextGetCurrentContext(&dctx));
    CHKERRQ(PetscDeviceContextGetDevice(dctx,&device));
  }
  CHKERRQ(AssertDeviceExists(device));
  /* test reference counting for default device */
  for (int i = 0; i < n; ++i) {
    CHKERRQ(PetscDeviceReference_Internal(device));
    devices[i] = device;
  }
  CHKERRQ(AssertDeviceExists(device));
  for (int i = 0; i < n; ++i) {
    CHKERRQ(PetscDeviceDestroy(&devices[i]));
    CHKERRQ(AssertDeviceExists(device));
    CHKERRQ(AssertDeviceDoesNotExist(devices[i]));
  }

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   TODO: broken in ci
   requires: !device
   suffix: no_device
   filter: Error: grep -E -o -e ".*No support for this operation for this object type" -e ".*PETSc is not configured with device support.*" -e "^\[0\]PETSC ERROR:.*[0-9]{1} [A-z]+\(\)"
   test:
     requires: debug
     suffix:   debug
   test:
     requires: !debug
     suffix:   opt

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 2 5}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip
   test:
     requires: sycl
     suffix: sycl

TEST*/
