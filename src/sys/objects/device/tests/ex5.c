static const char help[] = "Tests PetscDeviceGetAttribute().\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"
#include <petscviewer.h>

int main(int argc, char *argv[]) {
  PetscDevice device = NULL;
  size_t      shmem  = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT, PETSC_DECIDE, &device));
  PetscCall(PetscDeviceConfigure(device));
  PetscCall(PetscDeviceGetAttribute(device, PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK, &shmem));
  if (shmem == 0 || shmem == (size_t)-1) {
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
