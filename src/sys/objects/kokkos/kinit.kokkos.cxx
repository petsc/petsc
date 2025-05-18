#include <petsc/private/deviceimpl.h>
#include <petsc/private/kokkosimpl.hpp>
#include <petscpkg_version.h>
#include <petsc_kokkos.hpp>
#include <petscdevice_cupm.h>

PetscBool    PetscKokkosInitialized = PETSC_FALSE; // Has Kokkos been initialized (either by PETSc or by users)?
PetscScalar *PetscScalarPool        = nullptr;
PetscInt     PetscScalarPoolSize    = 0;

Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr = nullptr;

PetscErrorCode PetscKokkosFinalize_Private(void)
{
  PetscFunctionBegin;
  PetscCallCXX(delete PetscKokkosExecutionSpacePtr);
  PetscKokkosExecutionSpacePtr = nullptr;
  PetscCallCXX(Kokkos::kokkos_free(PetscScalarPool));
  PetscScalarPoolSize = 0;
  if (PetscBeganKokkos) {
    PetscCallCXX(Kokkos::finalize());
    PetscBeganKokkos = PETSC_FALSE;
  }
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
    auto args = Kokkos::InitArguments{}; /* use default constructor */
#endif

#if (defined(KOKKOS_ENABLE_CUDA) && defined(PETSC_HAVE_CUDA)) || (defined(KOKKOS_ENABLE_HIP) && defined(PETSC_HAVE_HIP)) || (defined(KOKKOS_ENABLE_SYCL) && defined(PETSC_HAVE_SYCL))
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDevice device;
    PetscInt    deviceId;
    PetscCall(PetscDeviceCreate(PETSC_DEVICE_DEFAULT(), PETSC_DECIDE, &device));
    PetscCall(PetscDeviceGetDeviceId(device, &deviceId));
    PetscCall(PetscDeviceDestroy(&device));
  #if PETSC_PKG_KOKKOS_VERSION_GE(4, 0, 0)
    // if device_id is not set, and no gpus have been found, kokkos will use CPU
    if (deviceId >= 0) args.set_device_id(static_cast<int>(deviceId));
  #elif PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    args.set_device_id(static_cast<int>(deviceId));
  #else
    PetscCall(PetscMPIIntCast(deviceId, &args.device_id));
  #endif
#endif

    /* To use PetscNumOMPThreads, one has to configure PETSc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
#if defined(KOKKOS_ENABLE_OPENMP) && PetscDefined(HAVE_OPENMP)
  #if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
    args.set_num_threads(PetscNumOMPThreads);
  #else
    args.num_threads = PetscNumOMPThreads;
  #endif
#endif
    PetscCallCXX(Kokkos::initialize(args));
    PetscBeganKokkos = PETSC_TRUE;
  }

  if (!PetscKokkosExecutionSpacePtr) { // No matter Kokkos is init'ed by PETSc or by user, we need to init PetscKokkosExecutionSpacePtr
#if (defined(KOKKOS_ENABLE_CUDA) && defined(PETSC_HAVE_CUDA)) || (defined(KOKKOS_ENABLE_HIP) && defined(PETSC_HAVE_HIP))
    PetscDeviceContext dctx;
    PetscDeviceType    dtype;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx)); // it internally sets PetscDefaultCuda/HipStream
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));

  #if defined(PETSC_HAVE_CUDA)
    if (dtype == PETSC_DEVICE_CUDA) PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace(PetscDefaultCudaStream));
  #elif defined(PETSC_HAVE_HIP)
    if (dtype == PETSC_DEVICE_HIP) PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace(PetscDefaultHipStream));
  #endif
#else
    // In all other cases, we use Kokkos default
    PetscCallCXX(PetscKokkosExecutionSpacePtr = new Kokkos::DefaultExecutionSpace());
#endif
  }

  if (!PetscScalarPoolSize) { // A pool for a small count of PetscScalars
    PetscScalarPoolSize = 1024;
    PetscCallCXX(PetscScalarPool = static_cast<PetscScalar *>(Kokkos::kokkos_malloc(sizeof(PetscScalar) * PetscScalarPoolSize)));
  }

  PetscKokkosInitialized = PETSC_TRUE; // PetscKokkosInitializeCheck() was called
  PetscFunctionReturn(PETSC_SUCCESS);
}
