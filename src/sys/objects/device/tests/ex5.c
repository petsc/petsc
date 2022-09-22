static const char help[] = "Tests PetscDeviceGetAttribute().\n\n";

#include "petscdevicetestcommon.h"
#include <petscviewer.h>

int main(int argc, char *argv[])
{
  PetscDevice device = NULL;
  size_t      shmem  = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT(), PETSC_DECIDE, &device));
  PetscCall(PetscDeviceConfigure(device));
  PetscCall(PetscDeviceGetAttribute(device, PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK, &shmem));
  if (PetscDefined(HAVE_CXX) && ((shmem == 0) || (shmem == (size_t)-1))) {
    // if no C++ then PetscDeviceGetAttribute defaults to 0
    PetscCall(PetscDeviceView(device, PETSC_VIEWER_STDOUT_SELF));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Maximum shared memory of %zu seems fishy", shmem);
  }
  PetscCall(PetscDeviceDestroy(&device));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
   requires: defined(PETSC_HAVE_CXX)

  testset:
   output_file: ./output/ExitSuccess.out
   args: -device_enable {{lazy eager}}
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -default_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -default_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -default_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -default_device_type sycl
     suffix: sycl

TEST*/
