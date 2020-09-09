#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <Kokkos_Core.hpp>

/* These wrappers are used as C bindings for the Kokkos routines */
PetscErrorCode PetscKokkosInitialize_Private(void)
{
  Kokkos::InitArguments args;
  int                   devId = -1;

  PetscFunctionBegin;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaGetDevice(&devId);
#elif defined(KOKKOS_ENABLE_HIP) /* Kokkos does not support CUDA and HIP at the same time */
  hipGetDevice(&devId);
#endif
  args.device_id = devId;
  Kokkos::initialize(args);
  PetscFunctionReturn(0);
}

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