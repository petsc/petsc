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
  ierr = PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device);CHKERRQ(ierr);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);
  /* should not destroy twice */
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);

  /* test reference counting */
  device = NULL;
  ierr = PetscArrayzero(devices,n);CHKERRQ(ierr);
  ierr = PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device);CHKERRQ(ierr);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < n; ++i) {
    ierr = PetscDeviceReference_Internal(device);CHKERRQ(ierr);
    devices[i] = device;
  }
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < n; ++i) {
    ierr = PetscDeviceDestroy(&devices[i]);CHKERRQ(ierr);
    ierr = AssertDeviceExists(device);CHKERRQ(ierr);
    ierr = AssertDeviceDoesNotExist(devices[i]);CHKERRQ(ierr);
  }
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);

  /* test the default devices exist */
  device = NULL;
  ierr = PetscArrayzero(devices,n);CHKERRQ(ierr);
  {
    PetscDeviceContext dctx;
    /* global context will have the default device */
    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextGetDevice(dctx,&device);CHKERRQ(ierr);
  }
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  /* test reference counting for default device */
  for (int i = 0; i < n; ++i) {
    ierr = PetscDeviceReference_Internal(device);CHKERRQ(ierr);
    devices[i] = device;
  }
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < n; ++i) {
    ierr = PetscDeviceDestroy(&devices[i]);CHKERRQ(ierr);
    ierr = AssertDeviceExists(device);CHKERRQ(ierr);
    ierr = AssertDeviceDoesNotExist(devices[i]);CHKERRQ(ierr);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRQ(ierr);
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
