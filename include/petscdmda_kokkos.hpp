#if !defined(PETSCDMDA_KOKKOS_HPP)
#define PETSCDMDA_KOKKOS_HPP

#include <petscdmda.h>

#if defined(PETSC_HAVE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosViewWrite    (DM,Vec,Kokkos::View<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosViewWrite(DM,Vec,Kokkos::View<      PetscScalar*,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<const PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<const PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosViewWrite    (DM,Vec,Kokkos::View<      PetscScalar**,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosViewWrite(DM,Vec,Kokkos::View<      PetscScalar**,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<const PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<const PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosViewWrite    (DM,Vec,Kokkos::View<      PetscScalar***,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosViewWrite(DM,Vec,Kokkos::View<      PetscScalar***,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<const PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<const PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView         (DM,Vec,Kokkos::View<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView     (DM,Vec,Kokkos::View<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosViewWrite    (DM,Vec,Kokkos::View<      PetscScalar****,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosViewWrite(DM,Vec,Kokkos::View<      PetscScalar****,MemorySpace>*);

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
