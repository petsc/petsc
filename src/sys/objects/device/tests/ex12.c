static const char help[] = "Tests PetscDevice and PetscDeviceContext init sequence control from command line.\n\n";

#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  // check that things are properly caught at init-time, i.e. allow failures for "lazy" during
  // initialize
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // and check that things are properly handled if explicitly requested
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: defined(PETSC_HAVE_DEVICE), defined(PETSC_USE_INFO)
    filter: grep -e PetscDevice -e "PETSC ERROR" -e "PETSc successfully started"
    args: -info -device_enable {{none lazy eager}separate output}
    args: -petsc_ci_portable_error_output -error_output_stdout
    test:
      requires: cuda
      args: -device_enable_cuda {{none lazy eager}separate output}
      suffix: cuda_no_env
    test:
      requires: cuda
      env: CUDA_VISIBLE_DEVICES=0
      args: -device_enable_cuda {{none lazy eager}separate output}
      suffix: cuda_env_set
    test:
      requires: cuda
      env: CUDA_VISIBLE_DEVICES=
      args: -device_enable_cuda {{none lazy eager}separate output}
      suffix: cuda_env_set_empty
    test:
      requires: hip
      args: -device_enable_hip {{none lazy eager}separate output}
      suffix: hip_no_env
    test:
      requires: hip
      env: HIP_VISIBLE_DEVICES=0
      args: -device_enable_hip {{none lazy eager}separate output}
      suffix: hip_env_set
    test:
      requires: hip
      env: HIP_VISIBLE_DEVICES=
      args: -device_enable_hip {{none lazy eager}separate output}
      suffix: hip_env_set_empty

TEST*/
