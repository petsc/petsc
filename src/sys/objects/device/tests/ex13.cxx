static const char help[] = "Test CUPM two-dimensional memory operations on a delayed device context.\n";

#include <petsc/private/cupminterface.hpp>

using Petsc::device::cupm::DeviceType;

template <DeviceType T>
struct TestMemory2D : Petsc::device::cupm::impl::Interface<T> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  static PetscErrorCode Run(PetscDeviceContext dctx)
  {
    const PetscInt width = 3, height = 2, src_pitch = 7, dst_pitch = 5;
    const auto     dtod = cupmMemcpyDeviceToDevice;
    unsigned char *src, *dst, *result;
    void          *handle;
    cupmStream_t   stream;

    PetscFunctionBeginUser;
    PetscCall(PetscDeviceContextGetStreamHandle(dctx, &handle));
    stream = *static_cast<cupmStream_t *>(handle);
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, src_pitch * height, PETSC_DECIDE, &src));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, dst_pitch * height, PETSC_DECIDE, &dst));
    PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, dst_pitch * height, PETSC_DECIDE, &result));
    for (PetscInt op = 0; op < 2; ++op) {
      PetscCall(PetscDeviceMemset(dctx, src, 1, src_pitch * height));
      PetscCall(PetscDeviceMemset(dctx, dst, 7, dst_pitch * height));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscDeviceContextDelay(dctx, 0.1));
      if (op) {
        PetscCall(PetscDeviceMemset(dctx, dst, 2, dst_pitch * height));
        PetscCallCUPM(cupmMemset2DAsync(dst, dst_pitch, 0, width, height, stream));
      } else {
        PetscCall(PetscDeviceMemset(dctx, src, 2, src_pitch * height));
        PetscCallCUPM(cupmMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, dtod, stream));
      }
      // Drain misplaced default-stream work before the delayed producer can overwrite its result.
      PetscCall(PetscDeviceContextSynchronize(PetscDeviceContextDefault));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscDeviceArrayCopy(dctx, result, dst, dst_pitch * height));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      for (PetscInt j = 0; j < height; ++j) {
        for (PetscInt i = 0; i < dst_pitch; ++i) {
          const unsigned char expected = i < width ? (op ? 0 : 2) : (op ? 2 : 7);

          PetscCheck(result[j * dst_pitch + i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect %s result at row %" PetscInt_FMT ", column %" PetscInt_FMT ": got %u, expected %u", op ? "memset" : "copy", j, i, (unsigned int)result[j * dst_pitch + i], (unsigned int)expected);
        }
      }
    }
    PetscCall(PetscDeviceFree(dctx, src));
    PetscCall(PetscDeviceFree(dctx, dst));
    PetscCall(PetscDeviceFree(dctx, result));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

int main(int argc, char **argv)
{
  PetscDeviceContext current, dctx;
  PetscDeviceType    dtype;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&current));
  PetscCall(PetscDeviceContextDuplicate(current, &dctx));
  PetscCall(PetscDeviceContextSetStreamType(dctx, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCall(PetscDeviceContextSynchronize(PetscDeviceContextDefault));
  switch (dtype) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(TestMemory2D<DeviceType::CUDA>::Run(dctx));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(TestMemory2D<DeviceType::HIP>::Run(dctx));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This test requires CUDA or HIP");
  }
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cxx

  testset:
    output_file: output/empty.out
    test:
      suffix: cuda
      requires: cuda
      args: -default_device_type cuda
    test:
      suffix: hip
      requires: hip
      args: -default_device_type hip

TEST*/
