static const char help[] = "Tests PetscDeviceContextSetStreamType().\n\n";

#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  const PetscStreamType stypes[] = {
#if PetscDefined(HAVE_CXX)
    PETSC_STREAM_GLOBAL_BLOCKING,
    PETSC_STREAM_DEFAULT_BLOCKING,
    PETSC_STREAM_GLOBAL_NONBLOCKING
#else
    PETSC_STREAM_GLOBAL_BLOCKING,
#endif
  };
  const PetscInt ntypes = PETSC_STATIC_ARRAY_LENGTH(stypes);

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // test that get-set trivially work
  for (PetscInt i = 0; i < ntypes; ++i) {
    PetscDeviceContext tmp;
    PetscStreamType    tmp_type;

    PetscCall(PetscDeviceContextCreate(&tmp));
    PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[i]));
    PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
    PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[i], "Set PetscDeviceStreamType %s does not match expected %s"));
    // test that any combination of get-set trivially works
    for (PetscInt j = 0; j < ntypes; ++j) {
      PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[j]));
      PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
      PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[j], "Set PetscDeviceStreamType %s does not match expected %s"));
      // reset it back to original
      PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[i]));
    }
    PetscCall(PetscDeviceContextDestroy(&tmp));
  }

  // test that any combination of get-set works when set up
  for (PetscInt i = 0; i < ntypes; ++i) {
    for (PetscInt j = 0; j < ntypes; ++j) {
      PetscDeviceContext tmp;
      PetscStreamType    tmp_type;

      PetscCall(PetscDeviceContextCreate(&tmp));
      // check this works through setup
      PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[i]));
      PetscCall(PetscDeviceContextSetUp(tmp));
      PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
      PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[i], "Set PetscDeviceStreamType %s does not match expected %s after PetscDeviceContextSetUp"));
      // now change the stream type
      PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[j]));
      PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
      PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[j], "Set PetscDeviceStreamType %s does not match expected %s when changing after PetscDeviceContextSetUp"));
      // reset it back to original
      PetscCall(PetscDeviceContextSetStreamType(tmp, stypes[i]));
      // and ensure this works
      PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
      PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[i], "Set PetscDeviceStreamType %s does not match expected %s after setting back to original"));
      // finally set up again
      PetscCall(PetscDeviceContextSetUp(tmp));
      // and ensure it has not changed
      PetscCall(PetscDeviceContextGetStreamType(tmp, &tmp_type));
      PetscCall(AssertPetscStreamTypesValidAndEqual(tmp_type, stypes[i], "Set PetscDeviceStreamType %s does not match expected %s after setting back to original and PetscDeviceContextSetUp"));
      PetscCall(PetscDeviceContextDestroy(&tmp));
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
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

  test:
    requires: !cxx
    output_file: ./output/ExitSuccess.out
    suffix: no_cxx

TEST*/
