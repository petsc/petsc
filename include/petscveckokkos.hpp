#ifndef __PETSCVEC_HPP
#define __PETSCVEC_HPP

#include <petscvec.h>

#if defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #include <Kokkos_DualView.hpp>

 #if defined(PETSC_HAVE_CUDA)
  #define WaitForKokkos() PetscCUDASynchronize ? (Kokkos::fence(),0) : 0
 #elif defined(PETSC_HAVE_HIP)
  #define WaitForKokkos() PetscHIPSynchronize ? (Kokkos::fence(),0) : 0
 #else
  #define WaitForKokkos() 0
 #endif

  using DeviceExecutionSpace               = Kokkos::DefaultExecutionSpace;
  using DeviceMemorySpace                  = typename DeviceExecutionSpace::memory_space;
  using HostMemorySpace                    = Kokkos::HostSpace;
  using PetscScalarKokkosDualView_t        = Kokkos::DualView<PetscScalar*>;

  using PetscScalarViewDevice_t            = Kokkos::View<PetscScalar*,DeviceMemorySpace>;
  using PetscScalarViewHost_t              = PetscScalarViewDevice_t::HostMirror;
  using ConstPetscScalarViewDevice_t       = Kokkos::View<const PetscScalar*,DeviceMemorySpace>;
  using ConstPetscScalarViewHost_t         = ConstPetscScalarViewDevice_t::HostMirror;

  PETSC_EXTERN PetscErrorCode VecKokkosGetDeviceView(Vec,PetscScalarViewDevice_t*);
  PETSC_EXTERN PetscErrorCode VecKokkosRestoreDeviceView(Vec,PetscScalarViewDevice_t*);

  PETSC_EXTERN PetscErrorCode VecKokkosGetDeviceViewRead(Vec,ConstPetscScalarViewDevice_t*);
  PETSC_STATIC_INLINE PetscErrorCode VecKokkosRestoreDeviceViewRead(Vec v,ConstPetscScalarViewDevice_t* dv) {return 0;}

  PETSC_EXTERN PetscErrorCode VecKokkosGetDeviceViewWrite(Vec,PetscScalarViewDevice_t*);
  PETSC_EXTERN PetscErrorCode VecKokkosRestoreDeviceViewWrite(Vec,PetscScalarViewDevice_t*);
#endif

#endif
