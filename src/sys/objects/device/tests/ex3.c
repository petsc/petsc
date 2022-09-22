static const char help[] = "Tests PetscDeviceContextDuplicate.\n\n";

#include "petscdevicetestcommon.h"

/* test duplication creates the same object type */
static PetscErrorCode TestPetscDeviceContextDuplicate(PetscDeviceContext dctx)
{
  PetscDevice        origDevice;
  PetscStreamType    origStype;
  PetscDeviceContext ddup;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  /* get everything we want first before any duplication */
  PetscCall(PetscDeviceContextGetStreamType(dctx, &origStype));
  PetscCall(PetscDeviceContextGetDevice(dctx, &origDevice));

  /* duplicate */
  PetscCall(PetscDeviceContextDuplicate(dctx, &ddup));
  PetscValidDeviceContext(ddup, 2);
  PetscCheckCompatibleDeviceContexts(dctx, 1, ddup, 2);

  {
    PetscDevice parDevice, dupDevice;

    PetscCall(PetscDeviceContextGetDevice(dctx, &parDevice));
    PetscCall(AssertPetscDevicesValidAndEqual(parDevice, origDevice, "Parent PetscDevice after duplication does not match parent original PetscDevice"));
    PetscCall(PetscDeviceContextGetDevice(ddup, &dupDevice));
    PetscCall(AssertPetscDevicesValidAndEqual(dupDevice, origDevice, "Duplicated PetscDevice does not match parent original PetscDevice"));
  }

  {
    PetscStreamType parStype, dupStype;

    PetscCall(PetscDeviceContextGetStreamType(dctx, &parStype));
    PetscCall(AssertPetscStreamTypesValidAndEqual(parStype, origStype, "Parent PetscStreamType after duplication does not match parent original PetscStreamType"));
    PetscCall(PetscDeviceContextGetStreamType(ddup, &dupStype));
    PetscCall(AssertPetscStreamTypesValidAndEqual(dupStype, origStype, "Duplicated PetscStreamType '%s' does not match parent original PetscStreamType '%s'"));
  }

  PetscCall(PetscDeviceContextDestroy(&ddup));
  /* duplicate should not take the original down with it */
  PetscValidDeviceContext(dctx, 1);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm           comm;
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* basic creation and destruction */
  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dctx, "local_"));
  PetscCall(PetscDeviceContextSetFromOptions(comm, dctx));
  PetscCall(TestPetscDeviceContextDuplicate(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestPetscDeviceContextDuplicate(dctx));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 4}}
   args: -device_enable {{lazy eager}}
   args: -local_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -default_device_type host -root_device_context_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -default_device_type cuda -root_device_context_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -default_device_type hip -root_device_context_device_type hip
     suffix: hip

TEST*/
