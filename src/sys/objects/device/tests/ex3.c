static const char help[] = "Tests PetscDeviceContextDuplicate.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

/* test duplication creates the same object type */
static PetscErrorCode TestPetscDeviceContextDuplicate(PetscDeviceContext dctx)
{
  PetscDevice        origDevice;
  PetscStreamType    origStype;
  PetscDeviceContext ddup;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* get everything we want first before any duplication */
  ierr = PetscDeviceContextGetStreamType(dctx,&origStype);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetDevice(dctx,&origDevice);CHKERRQ(ierr);

  /* duplicate */
  ierr = PetscDeviceContextDuplicate(dctx,&ddup);CHKERRQ(ierr);
  PetscValidDeviceContext(ddup,2);
  PetscCheckCompatibleDeviceContexts(dctx,1,ddup,2);

  {
    PetscDevice parDevice,dupDevice;

    ierr = PetscDeviceContextGetDevice(dctx,&parDevice);CHKERRQ(ierr);
    ierr = AssertPetscDevicesValidAndEqual(parDevice,origDevice,"Parent PetscDevice after duplication does not match parent original PetscDevice");CHKERRQ(ierr);
    ierr = PetscDeviceContextGetDevice(ddup,&dupDevice);CHKERRQ(ierr);
    ierr = AssertPetscDevicesValidAndEqual(dupDevice,origDevice,"Duplicated PetscDevice does not match parent original PetscDevice");CHKERRQ(ierr);
  }

  {
    PetscStreamType parStype,dupStype;

    ierr = PetscDeviceContextGetStreamType(dctx,&parStype);CHKERRQ(ierr);
    ierr = AssertPetscStreamTypesValidAndEqual(parStype,origStype,"Parent PetscStreamType after duplication does not match parent original PetscStreamType");CHKERRQ(ierr);
    ierr = PetscDeviceContextGetStreamType(ddup,&dupStype);CHKERRQ(ierr);
    ierr = AssertPetscStreamTypesValidAndEqual(dupStype,origStype,"Duplicated PetscStreamType '%s' does not match parent original PetscStreamType '%s'");CHKERRQ(ierr);
  }

  ierr = PetscDeviceContextDestroy(&ddup);CHKERRQ(ierr);
  /* duplicate should not take the original down with it */
  PetscValidDeviceContext(dctx,1);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* basic creation and destruction */
  ierr = PetscDeviceContextCreate(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD,"local_",dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(dctx);CHKERRQ(ierr);
  ierr = TestPetscDeviceContextDuplicate(dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRQ(ierr);

  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = TestPetscDeviceContextDuplicate(dctx);CHKERRQ(ierr);

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

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 4}}
   args: -local_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip

TEST*/
