#define HPDDM_MIXED_PRECISION 1
#include <petsc/private/petschpddm.h>
#include <petsc/private/deviceimpl.h>
#include <petscdevice_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

PetscErrorCode KSPSolve_HPDDM_CUDA_Private(KSP_HPDDM *data, const PetscScalar *b, PetscScalar *x, PetscInt n, MPI_Comm comm)
{
  const PetscInt     N = data->op->getDof() * n;
  PetscDeviceContext dctx;
  cudaStream_t      *stream;
  using K = std::conditional_t<PetscDefined(USE_REAL_DOUBLE), HPDDM::downscaled_type<PetscScalar>, HPDDM::upscaled_type<PetscScalar>>;

  PetscFunctionBegin; // TODO: remove host staging once HPDDM::IterativeMethod::solve() handles device pointers
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, (void **)&stream));
  if (data->precision != PETSC_SCALAR_PRECISION) {
    const thrust::device_ptr<const PetscScalar> db = thrust::device_pointer_cast(b);
    const thrust::device_ptr<PetscScalar>       dx = thrust::device_pointer_cast(x);
    K                                          *ptr, *host_ptr;
    thrust::device_ptr<K>                       dptr[2];

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMalloc((void **)&ptr, 2 * N * sizeof(K)));
    dptr[0] = thrust::device_pointer_cast(ptr);
    dptr[1] = thrust::device_pointer_cast(ptr + N);
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), db, N, dptr[0]));
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), dx, N, dptr[1]));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr, ptr, 2 * N * sizeof(K), cudaMemcpyDeviceToHost, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
#if PetscDefined(USE_COMPLEX)
    /* reinterpret thrust::complex<> as std::complex<> so that HPDDM deduces a type with BLAS/LAPACK and MPI support */
    std::complex<HPDDM::underlying_type<K>> *const hb = reinterpret_cast<std::complex<HPDDM::underlying_type<K>> *>(host_ptr);
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, hb, hb + N, n, comm));
#else
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
#endif
    PetscCallCUDA(cudaMemcpyAsync(ptr + N, host_ptr + N, N * sizeof(K), cudaMemcpyHostToDevice, *stream));
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), dptr[1], N, dx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCallCUDA(cudaFree(ptr));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(K)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(K)));
  } else {
    PetscScalar *host_ptr;

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr, b, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost, *stream));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr + N, x, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
#if PetscDefined(USE_COMPLEX)
    std::complex<PetscReal> *const hb = reinterpret_cast<std::complex<PetscReal> *>(host_ptr);
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, hb, hb + N, n, comm));
#else
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
#endif
    PetscCallCUDA(cudaMemcpyAsync(x, host_ptr + N, N * sizeof(PetscScalar), cudaMemcpyHostToDevice, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(PetscScalar)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(PetscScalar)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
