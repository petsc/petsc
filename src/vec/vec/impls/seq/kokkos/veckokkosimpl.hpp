#if !defined(VECKOKKOSIMPL_HPP)
#define VECKOKKOSIMPL_HPP

#include <petscvec.hpp>
#include <Kokkos_Core.hpp>

struct Vec_Kokkos {
  PetscScalar  *d_array;           /* this always holds the device data */
  PetscScalar  *d_array_allocated; /* if the array was allocated by PETSc this is its pointer */

  PetscScalarViewHost_t            h_v;
  PetscScalarViewDevice_t          d_v;
  PetscScalarKokkosDualView_t      dual_v;

  Vec_Kokkos(PetscInt n,PetscScalar *h_array_,PetscScalar *d_array_,PetscScalar *d_array_allocated_ = NULL)
    : d_array(d_array_),
      d_array_allocated(d_array_allocated_),
      h_v(h_array_,n),
      d_v(d_array_,n),
      dual_v(d_v,h_v){}

  ~Vec_Kokkos()
  {
    if (!std::is_same<DeviceMemorySpace,HostMemorySpace>::value) {
      Kokkos::kokkos_free<DeviceMemorySpace>(d_array_allocated);
    }
  }
};

#if defined(PETSC_HAVE_CUDA)
  #define WaitForKokkos() PetscCUDASynchronize ? (Kokkos::fence(),0) : 0
#elif defined(PETSC_HAVE_HIP)
  #define WaitForKokkos() PetscHIPSynchronize ? (Kokkos::fence(),0) : 0;
#else
  #define WaitForKokkos() 0;
#endif

PETSC_INTERN PetscErrorCode VecAbs_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqKokkos(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqKokkos(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode VecSet_SeqKokkos(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqKokkos(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqKokkos(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqKokkos(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqKokkos(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqKokkos(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqKokkos(Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqKokkos(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqKokkos(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqKokkos(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqKokkos(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqKokkos(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecConjugate_SeqKokkos(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqKokkos(Vec,NormType,PetscReal*);
PETSC_EXTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos_Private(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos_Private(Vec,PetscBool,PetscInt,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_Kokkos(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqKokkos(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqKokkos(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec,Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos_Private(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos_Private(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecDestroy_SeqKokkos_Private(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos_Private(Vec);
PETSC_INTERN PetscErrorCode VecMin_SeqKokkos(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMax_SeqKokkos(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecShift_SeqKokkos(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecGetArray_SeqKokkos(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqKokkos(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec,PetscScalar**,PetscMemType*);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec,PetscScalar**);
#endif
