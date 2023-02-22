#define HPDDM_MIXED_PRECISION 1
#include <petsc/private/petschpddm.h>
#include <petscdevice_cuda.h>
#include <thrust/device_ptr.h>

PetscErrorCode KSPSolve_HPDDM_CUDA_Private(KSP_HPDDM *data, const PetscScalar *b, PetscScalar *x, PetscInt n, MPI_Comm comm)
{
  const PetscInt N = data->op->getDof() * n;
#if PetscDefined(USE_REAL_DOUBLE)
  typedef HPDDM::downscaled_type<PetscScalar> K;
#endif
#if PetscDefined(USE_REAL_SINGLE)
  typedef HPDDM::upscaled_type<PetscScalar> K;
#endif

  PetscFunctionBegin; // TODO: remove all cudaMemcpy() once HPDDM::IterativeMethod::solve() handles device pointers
  if (data->precision != PETSC_KSPHPDDM_DEFAULT_PRECISION) {
    const thrust::device_ptr<const PetscScalar> db = thrust::device_pointer_cast(b);
    const thrust::device_ptr<PetscScalar>       dx = thrust::device_pointer_cast(x);
    K                                          *ptr, *host_ptr;
    thrust::device_ptr<K>                       dptr[2];

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMalloc((void **)&ptr, 2 * N * sizeof(K)));
    dptr[0] = thrust::device_pointer_cast(ptr);
    dptr[1] = thrust::device_pointer_cast(ptr + N);
    thrust::copy_n(thrust::cuda::par.on(PetscDefaultCudaStream), db, N, dptr[0]);
    thrust::copy_n(thrust::cuda::par.on(PetscDefaultCudaStream), dx, N, dptr[1]);
    PetscCallCUDA(cudaMemcpy(host_ptr, ptr, 2 * N * sizeof(K), cudaMemcpyDeviceToHost));
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
    PetscCallCUDA(cudaMemcpy(ptr + N, host_ptr + N, N * sizeof(K), cudaMemcpyHostToDevice));
    thrust::copy_n(thrust::cuda::par.on(PetscDefaultCudaStream), dptr[1], N, dx);
    PetscCallCUDA(cudaFree(ptr));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(K)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(K)));
  } else {
    PetscScalar *host_ptr;

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMemcpy(host_ptr, b, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaMemcpy(host_ptr + N, x, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
    PetscCallCUDA(cudaMemcpy(x, host_ptr + N, N * sizeof(PetscScalar), cudaMemcpyHostToDevice));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(PetscScalar)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(PetscScalar)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
