#include <petscdevice.h>
#include <petsc/private/petscimpl.h>
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
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  PetscErrorCode        ierr;
#endif
  Kokkos::InitArguments args;
  int                   devId = -1;

  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
    args.num_threads = -1; /* Kokkos default value of each parameter is -1 */
    args.num_numa    = -1;
    args.device_id   = -1;
    args.ndevices    = -1;
    args.skip_device = -1;
    args.disable_warnings = false;
   #if defined(KOKKOS_ENABLE_CUDA)
    cudaError_t cerr;

    ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
    cerr = cudaGetDevice(&devId);CHKERRCUDA(cerr);
   #elif defined(KOKKOS_ENABLE_HIP) /* Kokkos does not support CUDA and HIP at the same time */
    hipError_t herr;

    ierr = PetscHIPInitializeCheck();CHKERRQ(ierr);
    herr = hipGetDevice(&devId);CHKERRHIP(herr);
   #endif

    /* To use PetscNumOMPThreads, one has to configure petsc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
   #if defined(KOKKOS_ENABLE_OPENMP) && defined(PETSC_HAVE_OPENMP)
    args.num_threads = PetscNumOMPThreads;
   #endif

    args.device_id   = devId;
    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
