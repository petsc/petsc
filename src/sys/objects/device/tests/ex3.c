static const char help[] = "Tests PetscDeviceContextDuplicate.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

/* test duplication creates the same object type */
static PetscErrorCode TestPetscDeviceContextDuplicate(PetscDeviceContext dctx)
{
  PetscDevice        origDevice;
  PetscStreamType    origStype;
  PetscDeviceContext ddup;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* get everything we want first before any duplication */
  CHKERRQ(PetscDeviceContextGetStreamType(dctx,&origStype));
  CHKERRQ(PetscDeviceContextGetDevice(dctx,&origDevice));

  /* duplicate */
  CHKERRQ(PetscDeviceContextDuplicate(dctx,&ddup));
  PetscValidDeviceContext(ddup,2);
  PetscCheckCompatibleDeviceContexts(dctx,1,ddup,2);

  {
    PetscDevice parDevice,dupDevice;

    CHKERRQ(PetscDeviceContextGetDevice(dctx,&parDevice));
    CHKERRQ(AssertPetscDevicesValidAndEqual(parDevice,origDevice,"Parent PetscDevice after duplication does not match parent original PetscDevice"));
    CHKERRQ(PetscDeviceContextGetDevice(ddup,&dupDevice));
    CHKERRQ(AssertPetscDevicesValidAndEqual(dupDevice,origDevice,"Duplicated PetscDevice does not match parent original PetscDevice"));
  }

  {
    PetscStreamType parStype,dupStype;

    CHKERRQ(PetscDeviceContextGetStreamType(dctx,&parStype));
    CHKERRQ(AssertPetscStreamTypesValidAndEqual(parStype,origStype,"Parent PetscStreamType after duplication does not match parent original PetscStreamType"));
    CHKERRQ(PetscDeviceContextGetStreamType(ddup,&dupStype));
    CHKERRQ(AssertPetscStreamTypesValidAndEqual(dupStype,origStype,"Duplicated PetscStreamType '%s' does not match parent original PetscStreamType '%s'"));
  }

  CHKERRQ(PetscDeviceContextDestroy(&ddup));
  /* duplicate should not take the original down with it */
  PetscValidDeviceContext(dctx,1);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  /* basic creation and destruction */
  CHKERRQ(PetscDeviceContextCreate(&dctx));
  CHKERRQ(PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD,"local_",dctx));
  CHKERRQ(PetscDeviceContextSetUp(dctx));
  CHKERRQ(TestPetscDeviceContextDuplicate(dctx));
  CHKERRQ(PetscDeviceContextDestroy(&dctx));

  CHKERRQ(PetscDeviceContextGetCurrentContext(&dctx));
  CHKERRQ(TestPetscDeviceContextDuplicate(dctx));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  CHKERRQ(PetscFinalize());
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
