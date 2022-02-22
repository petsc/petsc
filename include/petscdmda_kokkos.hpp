#if !defined(PETSCDMDA_KOKKOS_HPP)
#define PETSCDMDA_KOKKOS_HPP

#include <petscvec_kokkos.hpp>
#include <petscdmda.h>

#if defined(PETSC_HAVE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****,MemorySpace>*);
#endif

#endif
