#ifndef __VECKOKKOSIMPL_HPP
#define __VECKOKKOSIMPL_HPP

#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petsc/private/vecimpl_kokkos.hpp>

#if defined(PETSC_USE_DEBUG)
  #define VecErrorIfNotKokkos(v) \
    do { \
      PetscBool isKokkos = PETSC_FALSE; \
      PetscCall(PetscObjectTypeCompareAny((PetscObject)(v), &isKokkos, VECSEQKOKKOS, VECMPIKOKKOS, VECKOKKOS, "")); \
      PetscCheck(isKokkos, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Calling VECKOKKOS methods on a non-VECKOKKOS object"); \
    } while (0)
#else
  #define VecErrorIfNotKokkos(v) \
    do { \
      (void)(v); \
    } while (0)
#endif

/* Stuff related to Vec_Kokkos */

struct Vec_Kokkos {
  PetscScalarKokkosDualView v_dual;

  /* COO stuff */
  PetscCountKokkosView jmap1_d; /* [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats in COO arrays */
  PetscCountKokkosView perm1_d; /* [tot1]: permutation array for local entries */

  PetscCountKokkosView  imap2_d;              /* [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in the vector */
  PetscCountKokkosView  jmap2_d;              /* [nnz2+1] */
  PetscCountKokkosView  perm2_d;              /* [recvlen] */
  PetscCountKokkosView  Cperm_d;              /* [sendlen]: permutation array to fill sendbuf[]. 'C' for communication */
  PetscScalarKokkosView sendbuf_d, recvbuf_d; /* Buffers for remote values in VecSetValuesCOO() */

  /* Construct Vec_Kokkos with the given array(s). n is the length of the array.
    If n != 0, host array (array_h) must not be NULL.
    If device array (array_d) is NULL, then a proper device mirror will be allocated.
    Otherwise, the mirror will be created using the given array_d.
  */
  Vec_Kokkos(PetscInt n, PetscScalar *array_h, PetscScalar *array_d = NULL)
  {
    PetscScalarKokkosViewHost v_h(array_h, n);
    PetscScalarKokkosView     v_d;

    if (array_d) {
      v_d = PetscScalarKokkosView(array_d, n); /* Use the given device array */
    } else {
      v_d = Kokkos::create_mirror_view(DefaultMemorySpace(), v_h); /* Create a mirror in DefaultMemorySpace but do not copy values */
    }
    v_dual = PetscScalarKokkosDualView(v_d, v_h);
    if (!array_d) v_dual.modify_host();
  }

  /* SFINAE: Update the object with an array in the given memory space,
     assuming the given array contains the latest value for this vector.
   */
  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  void UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));
    /* Kokkos said they would add error-checking so that users won't accidentally pass two different Views in this case */
    v_dual = PetscScalarKokkosDualView(v_h, v_h);
  }

  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<!std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  void UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));
    v_dual = PetscScalarKokkosDualView(v_dual.view<DefaultMemorySpace>(), v_h);
    v_dual.modify_host();
  }

  template <typename MemorySpace, std::enable_if_t<!std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  void UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosView v_d(array, v_dual.extent(0));
    v_dual = PetscScalarKokkosDualView(v_d, v_dual.view<Kokkos::HostSpace>());
    v_dual.modify_device();
  }

  void SetUpCOO(const Vec_Seq *vecseq, PetscInt m)
  {
    jmap1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecseq->jmap1, m + 1));
    perm1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecseq->perm1, vecseq->tot1));
  }

  void SetUpCOO(const Vec_MPI *vecmpi, PetscInt m)
  {
    jmap1_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->jmap1, m + 1));
    perm1_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->perm1, vecmpi->tot1));
    imap2_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->imap2, vecmpi->nnz2));
    jmap2_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->jmap2, vecmpi->nnz2 + 1));
    perm2_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->perm2, vecmpi->recvlen));
    Cperm_d   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->Cperm, vecmpi->sendlen));
    sendbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->sendbuf, vecmpi->sendlen));
    recvbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->recvbuf, vecmpi->recvlen));
  }
};

PETSC_INTERN PetscErrorCode VecAbs_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqKokkos(Vec, Vec, PetscScalar *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqKokkos(Vec, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecMTDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_EXTERN PetscErrorCode VecSet_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos(Vec, PetscInt, const PetscScalar *, Vec *);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecTDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecScale_SeqKokkos(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqKokkos(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqKokkos(Vec, PetscScalar, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqKokkos(Vec, Vec *);
PETSC_INTERN PetscErrorCode VecConjugate_SeqKokkos(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqKokkos(Vec, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos_Private(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos_Private(Vec, PetscBool, PetscInt, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_Kokkos(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos_Private(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos_Private(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecDestroy_SeqKokkos_Private(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos_Private(Vec);
PETSC_INTERN PetscErrorCode VecMin_SeqKokkos(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecMax_SeqKokkos(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecSum_SeqKokkos(Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecShift_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecGetArray_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqKokkos(Vec, PetscScalar **);

PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecGetArrayWriteAndMemType_SeqKokkos(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecGetSubVector_Kokkos_Private(Vec, PetscBool, IS, Vec *);
PETSC_INTERN PetscErrorCode VecRestoreSubVector_SeqKokkos(Vec, IS, Vec *);
#endif
