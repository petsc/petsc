#include <petsc/private/deviceimpl.h>
#include <Kokkos_Core.hpp>

PetscBool PetscKokkosInitialized = PETSC_FALSE;

PetscErrorCode PetscKokkosFinalize_Private(void)
{
  PetscFunctionBegin;
  Kokkos::finalize();
  PetscFunctionReturn(0);
}

PetscErrorCode PetscKokkosIsInitialized_Private(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  *isInitialized = Kokkos::is_initialized() ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
    auto args = Kokkos::InitArguments{}; /* use default constructor */

#if (defined(KOKKOS_ENABLE_CUDA) && PetscDefined(HAVE_CUDA)) || (defined(KOKKOS_ENABLE_HIP) && PetscDefined(HAVE_HIP)) || (defined(KOKKOS_ENABLE_SYCL) && PetscDefined(HAVE_SYCL))
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscMPIIntCast(dctx->device->deviceId,&args.device_id));
#endif

    args.disable_warnings = !PetscDefined(HAVE_KOKKOS_INIT_WARNINGS);

    /* To use PetscNumOMPThreads, one has to configure petsc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
#if defined(KOKKOS_ENABLE_OPENMP) && PetscDefined(HAVE_OPENMP)
    args.num_threads = PetscNumOMPThreads;
#endif

    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
