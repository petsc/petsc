#if !defined(__HIPVECIMPL)
#define __HIPVECIMPL

#include <petscvec.h>
#include <petschipblas.h>
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscScalar  *GPUarray;           /* this always holds the GPU data */
  PetscScalar  *GPUarray_allocated; /* if the array was allocated by PETSc this is its pointer */
  hipStream_t stream;              /* A stream for doing asynchronous data transfers */
} Vec_HIP;

PETSC_INTERN PetscErrorCode VecHIPGetArrays_Private(Vec,const PetscScalar**,const PetscScalar**,PetscOffloadMask*);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqHIP(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqHIP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqHIP(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqHIP(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode VecSet_SeqHIP(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqHIP(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqHIP(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqHIP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqHIP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqHIP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqHIP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqHIP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqHIP(Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqHIP(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqHIP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqHIP(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqHIP(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecConjugate_SeqHIP(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqHIP(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecHIPCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecHIPAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqHIP_Private(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP_Private(Vec,PetscBool,PetscInt,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_HIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqHIP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqHIP(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArray_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqHIP(Vec,PetscScalar**,PetscMemType*);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecCopy_SeqHIP_Private(Vec,Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecMax_SeqHIP(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMin_SeqHIP(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqHIP(Vec);

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define hipblasXaxpy(a,b,c,d,e,f,g) hipblasCaxpy((a),(b),(hipComplex*)(c),(hipComplex*)(d),(e),(hipComplex*)(f),(g))
#define hipblasXscal(a,b,c,d,e)     hipblasCscal((a),(b),(hipComplex*)(c),(hipComplex*)(d),(e))
#define hipblasXdotu(a,b,c,d,e,f,g) hipblasCdotu((a),(b),(hipComplex*)(c),(d),(hipComplex*)(e),(f),(hipComplex*)(g))
#define hipblasXdot(a,b,c,d,e,f,g)  hipblasCdotc((a),(b),(hipComplex*)(c),(d),(hipComplex*)(e),(f),(hipComplex*)(g))
#define hipblasXswap(a,b,c,d,e,f)   hipblasCswap((a),(b),(hipComplex*)(c),(d),(hipComplex*)(e),(f))
#define hipblasXnrm2(a,b,c,d,e)     hipblasScnrm2((a),(b),(hipComplex*)(c),(d),(e))
#define hipblasIXamax(a,b,c,d,e)    hipblasIcamax((a),(b),(hipComplex*)(c),(d),(e))
#define hipblasXasum(a,b,c,d,e)     hipblasScasum((a),(b),(hipComplex*)(c),(d),(e))
#define hipblasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) hipblasCgemv((a),(b),(c),(d),(hipComplex*)(e),(hipComplex*)(f),(g),(hipComplex*)(h),(i),(hipComplex*)(j),(hipComplex*)(k),(l))
#define hipblasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) hipblasCgemm((a),(b),(c),(d),(e),(f),(hipComplex*)(g),(hipComplex*)(h),(i),(hipComplex*)(j),(k),(hipComplex*)(l),(hipComplex*)(m),(n))
#define hipblasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   hipblasCgeam((a),(b),(c),(d),(e),(hipComplex*)(f),(hipComplex*)(g),(h),(hipComplex*)(i),(hipComplex*)(j),(k),(hipComplex*)(l),(m))
#else /* complex double */
#define hipblasXaxpy(a,b,c,d,e,f,g) hipblasZaxpy((a),(b),(hipDoubleComplex*)(c),(hipDoubleComplex*)(d),(e),(hipDoubleComplex*)(f),(g))
#define hipblasXscal(a,b,c,d,e)     hipblasZscal((a),(b),(hipDoubleComplex*)(c),(hipDoubleComplex*)(d),(e))
#define hipblasXdotu(a,b,c,d,e,f,g) hipblasZdotu((a),(b),(hipDoubleComplex*)(c),(d),(hipDoubleComplex*)(e),(f),(hipDoubleComplex*)(g))
#define hipblasXdot(a,b,c,d,e,f,g)  hipblasZdotc((a),(b),(hipDoubleComplex*)(c),(d),(hipDoubleComplex*)(e),(f),(hipDoubleComplex*)(g))
#define hipblasXswap(a,b,c,d,e,f)   hipblasZswap((a),(b),(hipDoubleComplex*)(c),(d),(hipDoubleComplex*)(e),(f))
#define hipblasXnrm2(a,b,c,d,e)     hipblasDznrm2((a),(b),(hipDoubleComplex*)(c),(d),(e))
#define hipblasIXamax(a,b,c,d,e)    hipblasIzamax((a),(b),(hipDoubleComplex*)(c),(d),(e))
#define hipblasXasum(a,b,c,d,e)     hipblasDzasum((a),(b),(hipDoubleComplex*)(c),(d),(e))
#define hipblasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) hipblasZgemv((a),(b),(c),(d),(hipDoubleComplex*)(e),(hipDoubleComplex*)(f),(g),(hipDoubleComplex*)(h),(i),(hipDoubleComplex*)(j),(hipDoubleComplex*)(k),(l))
#define hipblasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) hipblasZgemm((a),(b),(c),(d),(e),(f),(hipDoubleComplex*)(g),(hipDoubleComplex*)(h),(i),(hipDoubleComplex*)(j),(k),(hipDoubleComplex*)(l),(hipDoubleComplex*)(m),(n))
#define hipblasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   hipblasZgeam((a),(b),(c),(d),(e),(hipDoubleComplex*)(f),(hipDoubleComplex*)(g),(h),(hipDoubleComplex*)(i),(hipDoubleComplex*)(j),(k),(hipDoubleComplex*)(l),(m))
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

#endif
