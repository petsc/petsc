#if !defined(PETSCVEC_KOKKOS_HPP)
#define PETSCVEC_KOKKOS_HPP

#include <petscvec.h>

#if defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>

  #if defined(PETSC_HAVE_CUDA)
    #define WaitForKokkos() PetscCUDASynchronize ? (Kokkos::fence(),0) : 0
  #elif defined(PETSC_HAVE_HIP)
    #define WaitForKokkos() PetscHIPSynchronize ? (Kokkos::fence(),0) : 0
  #else
    #define WaitForKokkos() 0
  #endif

  /* Routines to get/restore Kokkos Views from PETSc vectors */

  /* Like VecGetArrayRead() */
  template<class MemorySpace> PetscErrorCode VecGetKokkosView    (Vec,Kokkos::View<const PetscScalar*,MemorySpace>*);
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,Kokkos::View<const PetscScalar*,MemorySpace>*){return 0;}

  /* Like VecGetArray() */
  template<class MemorySpace> PetscErrorCode VecGetKokkosView    (Vec,Kokkos::View<PetscScalar*,MemorySpace>*);
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,Kokkos::View<PetscScalar*,MemorySpace>*);

  /* Like VecGetArrayWrite() */
  template<class MemorySpace> PetscErrorCode VecGetKokkosViewWrite    (Vec,Kokkos::View<PetscScalar*,MemorySpace>*);
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosViewWrite(Vec,Kokkos::View<PetscScalar*,MemorySpace>*);

#endif

#endif
