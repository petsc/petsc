#pragma once

#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petsc/private/kokkosimpl.hpp>

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
  PetscScalarKokkosView     unplaced_d; /* Unplaced device array in VecKokkosPlaceArray() */

  /* COO stuff */
  PetscCountKokkosView jmap1_d; /* [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats in COO arrays */
  PetscCountKokkosView perm1_d; /* [tot1]: permutation array for local entries */

  PetscCountKokkosView  imap2_d;              /* [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in the vector */
  PetscCountKokkosView  jmap2_d;              /* [nnz2+1] */
  PetscCountKokkosView  perm2_d;              /* [recvlen] */
  PetscCountKokkosView  Cperm_d;              /* [sendlen]: permutation array to fill sendbuf[]. 'C' for communication */
  PetscScalarKokkosView sendbuf_d, recvbuf_d; /* Buffers for remote values in VecSetValuesCOO() */

  // (internal use only) sometimes we need to allocate multiple vectors from a contiguous memory block.
  // We stash the memory in w_dual, which has the same lifespan as this vector. See VecDuplicateVecs_SeqKokkos_GEMV.
  PetscScalarKokkosDualView w_dual;

  /* Construct Vec_Kokkos with the given array(s). n is the length of the array.
    If n != 0, host array (array_h) must not be NULL.
    If device array (array_d) is NULL, then a proper device mirror will be allocated.
    Otherwise, the mirror will be created using the given array_d.
    If both arrays are given, we assume they contain the same value (i.e., sync'ed)
  */
  Vec_Kokkos(PetscInt n, PetscScalar *array_h, PetscScalar *array_d = NULL)
  {
    PetscScalarKokkosViewHost v_h(array_h, n);
    PetscScalarKokkosView     v_d;

    if (array_d) {
      v_d = PetscScalarKokkosView(array_d, n); /* Use the given device array */
    } else {
      v_d = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, DefaultMemorySpace(), v_h); /* Create a mirror in DefaultMemorySpace but do not copy values */
    }
    v_dual = PetscScalarKokkosDualView(v_d, v_h);
    if (!array_d) v_dual.modify_host();
  }

  // Construct Vec_Kokkos with the given DualView. Use the sync state as is. With reference counting, Kokkos manages its lifespan.
  Vec_Kokkos(PetscScalarKokkosDualView dual) : v_dual(dual) { }

  /* SFINAE: Update the object with an array in the given memory space,
     assuming the given array contains the latest value for this vector.
   */
  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, HostMirrorMemorySpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosView     v_d(array, v_dual.extent(0));
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));

    PetscFunctionBegin;
    /* Kokkos said they would add error-checking so that users won't accidentally pass two different Views in this case */
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_d, v_h));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, HostMirrorMemorySpace>::value, bool> = true, std::enable_if_t<!std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));

    PetscFunctionBegin;
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_dual.view<DefaultMemorySpace>(), v_h));
    PetscCallCXX(v_dual.modify_host());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename MemorySpace, std::enable_if_t<!std::is_same<MemorySpace, HostMirrorMemorySpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosView v_d(array, v_dual.extent(0));

    PetscFunctionBegin;
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_d, v_dual.view_host()));
    PetscCallCXX(v_dual.modify_device());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode SetUpCOO(const Vec_Seq *vecseq, PetscInt m)
  {
    PetscFunctionBegin;
    PetscCallCXX(jmap1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecseq->jmap1, m + 1)));
    PetscCallCXX(perm1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecseq->perm1, vecseq->tot1)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode SetUpCOO(const Vec_MPI *vecmpi, PetscInt m)
  {
    PetscFunctionBegin;
    PetscCallCXX(jmap1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->jmap1, m + 1)));
    PetscCallCXX(perm1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->perm1, vecmpi->tot1)));
    PetscCallCXX(imap2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->imap2, vecmpi->nnz2)));
    PetscCallCXX(jmap2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->jmap2, vecmpi->nnz2 + 1)));
    PetscCallCXX(perm2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->perm2, vecmpi->recvlen)));
    PetscCallCXX(Cperm_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(vecmpi->Cperm, vecmpi->sendlen)));
    PetscCallCXX(sendbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->sendbuf, vecmpi->sendlen)));
    PetscCallCXX(recvbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->recvbuf, vecmpi->recvlen)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

PETSC_INTERN PetscErrorCode VecAbs_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqKokkos(Vec, Vec, PetscScalar *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqKokkos(Vec, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecMTDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecSet_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos(Vec, PetscInt, const PetscScalar *, Vec *);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecTDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecScale_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqKokkos(Vec, PetscScalar, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecConjugate_SeqKokkos(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqKokkos(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecErrorWeightedNorms_SeqKokkos(Vec, Vec, Vec, NormType, PetscReal, Vec, PetscReal, Vec, PetscReal, PetscReal *, PetscInt *, PetscReal *, PetscInt *, PetscReal *, PetscInt *);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos_Private(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos_Private(Vec, PetscBool, PetscInt, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_Kokkos(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos_Private(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos_Private(Vec, PetscRandom);
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

PETSC_INTERN PetscErrorCode VecDuplicateVecs_Kokkos_GEMV(Vec, PetscInt, Vec **);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos_GEMV(Vec, PetscInt, const Vec *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecMTDot_SeqKokkos_GEMV(Vec, PetscInt, const Vec *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos_GEMV(Vec, PetscInt, const PetscScalar *, Vec *);

PETSC_INTERN PetscErrorCode VecCreateMPIKokkosWithLayoutAndArrays_Private(PetscLayout map, const PetscScalar *, const PetscScalar *, Vec *);
