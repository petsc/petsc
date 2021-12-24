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

#define PETSC_AND_KOKKOS_HAVE(CUPM) (defined(KOKKOS_ENABLE_##CUPM) && PetscDefined(HAVE_##CUPM))

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  Kokkos::InitArguments args;

  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
    args.num_threads = -1; /* Kokkos default value of each parameter is -1 */
    args.num_numa    = -1;
    args.device_id   = -1;
    args.ndevices    = -1;
    args.skip_device = -1;

#if defined(PETSC_HAVE_KOKKOS_INIT_WARNINGS)
    args.disable_warnings = false;
#else
    args.disable_warnings = true;
#endif

#if PETSC_AND_KOKKOS_HAVE(CUDA) || PETSC_AND_KOKKOS_HAVE(HIP) || PETSC_AND_KOKKOS_HAVE(SYCL)
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(dctx->device->deviceId,&args.device_id);CHKERRQ(ierr);
#endif

    /* To use PetscNumOMPThreads, one has to configure petsc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
   #if defined(KOKKOS_ENABLE_OPENMP) && defined(PETSC_HAVE_OPENMP)
    args.num_threads = PetscNumOMPThreads;
   #endif

    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
