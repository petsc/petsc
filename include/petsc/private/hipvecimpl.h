#ifndef PETSC_HIPVECIMPL_H
#define PETSC_HIPVECIMPL_H

#include <petscvec.h>
#include <petscdevice_hip.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscScalar *GPUarray;           /* this always holds the GPU data */
  PetscScalar *GPUarray_allocated; /* if the array was allocated by PETSc this is its pointer */
  hipStream_t  stream;             /* A stream for doing asynchronous data transfers */
} Vec_HIP;

PETSC_INTERN PetscErrorCode VecHIPGetArrays_Private(Vec, const PetscScalar **, const PetscScalar **, PetscOffloadMask *);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqHIP(Vec, Vec, PetscScalar *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqHIP(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqHIP(Vec, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqHIP(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_EXTERN PetscErrorCode VecSet_SeqHIP(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqHIP(Vec, PetscInt, const PetscScalar *, Vec *);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqHIP(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqHIP(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqHIP(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqHIP(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecDot_SeqHIP(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecTDot_SeqHIP(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecScale_SeqHIP(Vec, PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqHIP(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqHIP(Vec, Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqHIP(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqHIP(Vec, PetscScalar, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqHIP(Vec, Vec *);
PETSC_INTERN PetscErrorCode VecConjugate_SeqHIP(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqHIP(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecHIPCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecHIPAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqHIP_Private(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP_Private(Vec, PetscBool, PetscInt, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_HIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqHIP(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqHIP(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqHIP(Vec, Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqHIP(Vec, Vec);
PETSC_INTERN PetscErrorCode VecGetLocalVectorRead_SeqHIP(Vec, Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVectorRead_SeqHIP(Vec, Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqHIP(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecGetArray_SeqHIP(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqHIP(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqHIP(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqHIP(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecGetArrayWriteAndMemType_SeqHIP(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecCopy_SeqHIP_Private(Vec, Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecMax_SeqHIP(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecMin_SeqHIP(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecSum_SeqHIP(Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecShift_SeqHIP(Vec, PetscScalar);

/* complex single */
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipblasXaxpy(a, b, c, d, e, f, g)                      hipblasCaxpy((a), (b), (hipblasComplex *)(c), (hipblasComplex *)(d), (e), (hipblasComplex *)(f), (g))
    #define hipblasXscal(a, b, c, d, e)                            hipblasCscal((a), (b), (hipblasComplex *)(c), (hipblasComplex *)(d), (e))
    #define hipblasXdotu(a, b, c, d, e, f, g)                      hipblasCdotu((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f), (hipblasComplex *)(g))
    #define hipblasXdot(a, b, c, d, e, f, g)                       hipblasCdotc((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f), (hipblasComplex *)(g))
    #define hipblasXswap(a, b, c, d, e, f)                         hipblasCswap((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f))
    #define hipblasXnrm2(a, b, c, d, e)                            hipblasScnrm2((a), (b), (hipblasComplex *)(c), (d), (e))
    #define hipblasIXamax(a, b, c, d, e)                           hipblasIcamax((a), (b), (hipblasComplex *)(c), (d), (e))
    #define hipblasXasum(a, b, c, d, e)                            hipblasScasum((a), (b), (hipblasComplex *)(c), (d), (e))
    #define hipblasXgemv(a, b, c, d, e, f, g, h, i, j, k, l)       hipblasCgemv((a), (b), (c), (d), (hipblasComplex *)(e), (hipblasComplex *)(f), (g), (hipblasComplex *)(h), (i), (hipblasComplex *)(j), (hipblasComplex *)(k), (l))
    #define hipblasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) hipblasCgemm((a), (b), (c), (d), (e), (f), (hipblasComplex *)(g), (hipblasComplex *)(h), (i), (hipblasComplex *)(j), (k), (hipblasComplex *)(l), (hipblasComplex *)(m), (n))
    #define hipblasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m)    hipblasCgeam((a), (b), (c), (d), (e), (hipblasComplex *)(f), (hipblasComplex *)(g), (h), (hipblasComplex *)(i), (hipblasComplex *)(j), (k), (hipblasComplex *)(l), (m))
  #else /* complex double */
    #define hipblasXaxpy(a, b, c, d, e, f, g) hipblasZaxpy((a), (b), (hipblasDoubleComplex *)(c), (hipblasDoubleComplex *)(d), (e), (hipblasDoubleComplex *)(f), (g))
    #define hipblasXscal(a, b, c, d, e)       hipblasZscal((a), (b), (hipblasDoubleComplex *)(c), (hipblasDoubleComplex *)(d), (e))
    #define hipblasXdotu(a, b, c, d, e, f, g) hipblasZdotu((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f), (hipblasDoubleComplex *)(g))
    #define hipblasXdot(a, b, c, d, e, f, g)  hipblasZdotc((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f), (hipblasDoubleComplex *)(g))
    #define hipblasXswap(a, b, c, d, e, f)    hipblasZswap((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f))
    #define hipblasXnrm2(a, b, c, d, e)       hipblasDznrm2((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
    #define hipblasIXamax(a, b, c, d, e)      hipblasIzamax((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
    #define hipblasXasum(a, b, c, d, e)       hipblasDzasum((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
    #define hipblasXgemv(a, b, c, d, e, f, g, h, i, j, k, l) \
      hipblasZgemv((a), (b), (c), (d), (hipblasDoubleComplex *)(e), (hipblasDoubleComplex *)(f), (g), (hipblasDoubleComplex *)(h), (i), (hipblasDoubleComplex *)(j), (hipblasDoubleComplex *)(k), (l))
    #define hipblasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
      hipblasZgemm((a), (b), (c), (d), (e), (f), (hipblasDoubleComplex *)(g), (hipblasDoubleComplex *)(h), (i), (hipblasDoubleComplex *)(j), (k), (hipblasDoubleComplex *)(l), (hipblasDoubleComplex *)(m), (n))
    #define hipblasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m) \
      hipblasZgeam((a), (b), (c), (d), (e), (hipblasDoubleComplex *)(f), (hipblasDoubleComplex *)(g), (h), (hipblasDoubleComplex *)(i), (hipblasDoubleComplex *)(j), (k), (hipblasDoubleComplex *)(l), (m))
  #endif
#else /* real single */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipblasXaxpy  hipblasSaxpy
    #define hipblasXscal  hipblasSscal
    #define hipblasXdotu  hipblasSdot
    #define hipblasXdot   hipblasSdot
    #define hipblasXswap  hipblasSswap
    #define hipblasXnrm2  hipblasSnrm2
    #define hipblasIXamax hipblasIsamax
    #define hipblasXasum  hipblasSasum
    #define hipblasXgemv  hipblasSgemv
    #define hipblasXgemm  hipblasSgemm
    #define hipblasXgeam  hipblasSgeam
  #else /* real double */
    #define hipblasXaxpy  hipblasDaxpy
    #define hipblasXscal  hipblasDscal
    #define hipblasXdotu  hipblasDdot
    #define hipblasXdot   hipblasDdot
    #define hipblasXswap  hipblasDswap
    #define hipblasXnrm2  hipblasDnrm2
    #define hipblasIXamax hipblasIdamax
    #define hipblasXasum  hipblasDasum
    #define hipblasXgemv  hipblasDgemv
    #define hipblasXgemm  hipblasDgemm
    #define hipblasXgeam  hipblasDgeam
  #endif
#endif

#endif // PETSC_HIPVECIMPL_H
