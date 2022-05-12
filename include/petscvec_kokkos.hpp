#if !defined(PETSCVEC_KOKKOS_HPP)
#define PETSCVEC_KOKKOS_HPP

#include <petscconf.h>

/* SUBMANSEC = Vec */

#if defined(PETSC_HAVE_KOKKOS)
  #if defined(petsccomplexlib)
    #error "Error: You must include petscvec_kokkos.hpp before other petsc headers in this C++ file to use petsc complex with Kokkos"
  #endif

  #define PETSC_DESIRE_KOKKOS_COMPLEX   1   /* To control the definition of petsccomplexlib in petscsystypes.h */
#endif

#include <petscvec.h>

#if defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>

/*@C
     VecGetKokkosView - Returns a constant Kokkos View that contains up-to-date data of a vector in the specified memory space.

   Synopsis:
   #include <petscvec_kokkos.hpp>
   PetscErrorCode VecGetKokkosView  (Vec v,Kokkos::View<const PetscScalar*,MemorySpace>* kv);
   PetscErrorCode VecGetKokkosView  (Vec v,Kokkos::View<PetscScalar*,MemorySpace>* kv);

   Logically Collective on Vec

   Input Parameter:
.  v - the vector in type of VECKOKKOS

   Output Parameter:
.  kv - the Kokkos View with a user-specified template parameter MemorySpace

   Notes:
   If the vector is not of type VECKOKKOS, an error will be raised.
   The functions are similar to VecGetArrayRead() and VecGetArray() respectively. One can read-only or read/write the returned Kokkos View.
   Note that passing in a const View enables read-only access.
   One must return the View by a matching VecRestoreKokkosView() after finishing using the View. Currently, only two memory
   spaces are supported: Kokkos::HostSpace and Kokkos::DefaultExecutionSpace::memory_space.
   If needed, a memory copy will be internally called to copy the latest vector data to the specified memory space.

   Level: beginner

.seealso: `VecRestoreKokkosView()`, `VecRestoreArray()`, `VecGetKokkosViewWrite()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
  template<class MemorySpace> PetscErrorCode VecGetKokkosView (Vec,Kokkos::View<const PetscScalar*,MemorySpace>*);
  template<class MemorySpace> PetscErrorCode VecGetKokkosView (Vec,Kokkos::View<PetscScalar*,MemorySpace>*);

/*@C
   VecRestoreKokkosView - Returns a Kokkos View gotten by VecGetKokkosView().

   Synopsis:
   #include <petscvec_kokkos.hpp>
   PetscErrorCode VecRestoreKokkosView  (Vec v,Kokkos::View<const PetscScalar*,MemorySpace>* kv);
   PetscErrorCode VecRestoreKokkosView  (Vec v,Kokkos::View<PetscScalar*,MemorySpace>* kv);

   Logically Collective on Vec

   Input Parameters:
+  v  - the vector in type of VECKOKKOS
-  kv - the Kokkos View with a user-specified template parameter MemorySpace

   Notes:
   If the vector is not of type VECKOKKOS, an error will be raised.
   The functions are similar to VecRestoreArrayRead() and VecRestoreArray() respectively. They are the counterpart of VecGetKokkosView().

   Level: beginner

.seealso: `VecGetKokkosView()`, `VecRestoreKokkosViewWrite()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,Kokkos::View<const PetscScalar*,MemorySpace>*){return 0;}
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,Kokkos::View<PetscScalar*,MemorySpace>*);


/*@C
   VecGetKokkosViewWrite - Returns a Kokkos View that contains the array of a vector in the specified memory space.

   Synopsis:
   #include <petscvec_kokkos.hpp>
   PetscErrorCode VecGetKokkosViewWrite  (Vec v,Kokkos::View<PetscScalar*,MemorySpace>* kv);

   Logically Collective on Vec

   Input Parameter:
.  v - the vector in type of VECKOKKOS

   Output Parameter:
.  kv - the Kokkos View with a user-specified template parameter MemorySpace

   Notes:
   If the vector is not of type VECKOKKOS, an error will be raised.
   The functions is similar to VecGetArrayWrite(). The returned view might contain garbage data or stale data and one is not
   expected to read data from the View. Instead, one is expected to overwrite all data in the View.
   One must return the View by a matching VecRestoreKokkosViewWrite() after finishing using the View.
  Currently, only two memory spaces are supported: Kokkos::HostSpace and Kokkos::DefaultExecutionSpace::memory_space.

   Level: beginner

.seealso: `VecRestoreKokkosViewWrite()`, `VecRestoreKokkosView()`, `VecGetKokkosView()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
  template<class MemorySpace> PetscErrorCode VecGetKokkosViewWrite    (Vec,Kokkos::View<PetscScalar*,MemorySpace>*);

/*@C
   VecRestoreKokkosViewWrite - Returns a Kokkos View gotten by VecGetKokkosViewWrite().

   Synopsis:
   #include <petscvec_kokkos.hpp>
   PetscErrorCode VecRestoreKokkosViewWrite  (Vec v,Kokkos::View<PetscScalar*,MemorySpace>* kv);

   Logically Collective on Vec

   Input Parameters:
+  v  - the vector in type of VECKOKKOS
-  kv - the Kokkos View with a user-specified template parameter MemorySpace

   Notes:
   If the vector is not of type VECKOKKOS, an error will be raised.
   The function is similar to VecRestoreArrayWrite(). It is the counterpart of VecGetKokkosViewWrite().

   Level: beginner

.seealso: `VecGetKokkosViewWrite()`, `VecGetKokkosView()`, `VecGetKokkosView()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosViewWrite(Vec,Kokkos::View<PetscScalar*,MemorySpace>*);

 #if defined(PETSC_HAVE_COMPLEX) && defined(PETSC_USE_COMPLEX)
  static_assert(std::alignment_of<Kokkos::complex<PetscReal>>::value == std::alignment_of<std::complex<PetscReal>>::value,
   "Alignment of Kokkos::complex<PetscReal> and std::complex<PetscReal> mismatch. Reconfigure your Kokkos with -DKOKKOS_ENABLE_COMPLEX_ALIGN=OFF, or let PETSc install Kokkos for you with --download-kokkos --download-kokkos-kernels");
 #endif

#endif

#endif
