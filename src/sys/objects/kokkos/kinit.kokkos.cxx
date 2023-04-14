#include <petsc/private/deviceimpl.h>
#include <petscpkg_version.h>
#include <petsc_kokkos.hpp>

PetscBool PetscKokkosInitialized = PETSC_FALSE;

Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr = nullptr;

PetscErrorCode PetscKokkosFinalize_Private(void)
{
  PetscFunctionBegin;
  PetscCallCXX(delete PetscKokkosExecutionSpacePtr);
  Kokkos::finalize();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscKokkosIsInitialized_Private(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  *isInitialized = Kokkos::is_initialized() ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
#if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    auto args = Kokkos::InitializationSettings();
#else
    auto args             = Kokkos::InitArguments{}; /* use default constructor */
#endif

#if (defined(KOKKOS_ENABLE_CUDA) && PetscDefined(HAVE_CUDA)) || (defined(KOKKOS_ENABLE_HIP) && PetscDefined(HAVE_HIP)) || (defined(KOKKOS_ENABLE_SYCL) && PetscDefined(HAVE_SYCL))
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  #if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    args.set_device_id(static_cast<int>(dctx->device->deviceId));
  #else
    PetscCall(PetscMPIIntCast(dctx->device->deviceId, &args.device_id));
  #endif
#endif

#if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    args.set_disable_warnings(!PetscDefined(HAVE_KOKKOS_INIT_WARNINGS));
#else
    args.disable_warnings = !PetscDefined(HAVE_KOKKOS_INIT_WARNINGS);
#endif

    /* To use PetscNumOMPThreads, one has to configure petsc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
#if defined(KOKKOS_ENABLE_OPENMP) && PetscDefined(HAVE_OPENMP)
  #if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    args.set_num_threads(PetscNumOMPThreads);
  #else
    args.num_threads = PetscNumOMPThreads;
  #endif
#endif

    Kokkos::initialize(args);
#if defined(PETSC_HAVE_CUDA)
    extern cudaStream_t PetscDefaultCudaStream;
    PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace(PetscDefaultCudaStream));
#elif defined(PETS_HAVE_HIP)
    extern hipStream_t PetscDefaultHipStream;
    PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace(PetscDefaultHipStream));
#else
    PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace());
#endif
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
