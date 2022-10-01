/*
     Defines the matrix operations for sequential dense with HIP
     Portions of this code are under:
     Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <petscpkg_version.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsc/private/hipvecimpl.h>         /* hipblas definitions are here */
#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  #include <hipsolver/hipsolver.h>
#else
  #include <hipsolver.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverCpotrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h))
    #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverCpotrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
    #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverCpotrs((a), (b), (c), (d), (hipComplex *)(e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (j), (k))
    #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverCpotri((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h))
    #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverCpotri_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
    #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverCsytrf((a), (b), (c), (hipComplex *)(d), (e), (f), (hipComplex *)(g), (h), (i))
    #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverCsytrf_bufferSize((a), (b), (hipComplex *)(c), (d), (e))
    #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverCgetrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h), (i))
    #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverCgetrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
    #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverCgetrs((a), (b), (c), (d), (hipComplex *)(e), (f), (g), (hipComplex *)(h), (i), (hipComplex *)(j), (k), (l))
    #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverCgeqrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
    #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverCgeqrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (hipComplex *)(g), (h), (i))
    #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverCunmqr_bufferSize((a), (b), (c), (d), (e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (hipComplex *)(j), (k), (l))
    #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverCunmqr((a), (b), (c), (d), (e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (hipComplex *)(j), (k), (hipComplex *)(l), (m), (n))
    #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasCtrsm((a), (b), (c), (d), (e), (f), (g), (hipComplex *)(h), (hipComplex *)(i), (j), (hipComplex *)(k), (l))
  #else /* complex double */
    #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverZpotrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h))
    #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverZpotrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
    #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverZpotrs((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (j), (k))
    #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverZpotri((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h))
    #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverZpotri_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
    #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverZsytrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (f), (hipDoubleComplex *)(g), (h), (i))
    #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverZsytrf_bufferSize((a), (b), (hipDoubleComplex *)(c), (d), (e))
    #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverZgetrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h), (i))
    #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverZgetrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
    #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverZgetrs((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (g), (hipDoubleComplex *)(h), (i), (hipDoubleComplex *)(j), (k), (l))
    #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverZgeqrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
    #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverZgeqrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (hipDoubleComplex *)(g), (h), (i))
    #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverZunmqr_bufferSize((a), (b), (c), (d), (e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (hipDoubleComplex *)(j), (k), (l))
    #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverZunmqr((a), (b), (c), (d), (e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (hipDoubleComplex *)(j), (k), (hipDoubleComplex *)(l), (m), (n))
    #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasZtrsm((a), (b), (c), (d), (e), (f), (g), (hipDoubleComplex *)(h), (hipDoubleComplex *)(i), (j), (hipDoubleComplex *)(k), (l))
  #endif
#else /* real single */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)          hipsolverSpotrf((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h))
    #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)     hipsolverSpotrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
    #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k) hipsolverSpotrs((a), (b), (c), (d), (float *)(e), (f), (float *)(g), (h), (float *)(i), (j), (k))
    #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)          hipsolverSpotri((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h))
    #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)     hipsolverSpotri_bufferSize((a), (b), (c), (float *)(d), (e), (f))
    #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)       hipsolverSsytrf((a), (b), (c), (float *)(d), (e), (f), (float *)(g), (h), (i))
    #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)        hipsolverSsytrf_bufferSize((a), (b), (float *)(c), (d), (e))
    #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)       hipsolverSgetrf((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h), (i))
    #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)     hipsolverSgetrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
    #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverSgetrs((a),(b),(c),(d),(float*)(e),(f),(g),(float*)(h),(i),(float*)(j),(k),(l)))
    #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverSgeqrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
    #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverSgeqrf((a), (b), (c), (float *)(d), (e), (float *)(f), (float *)(g), (h), (i))
    #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverSormqr_bufferSize((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (l))
    #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverSormqr((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (float *)(l), (m), (n))
    #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasStrsm((a), (b), (c), (d), (e), (f), (g), (float *)(h), (float *)(i), (j), (float *)(k), (l))
  #else /* real double */
    #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverDpotrf((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h))
    #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverDpotrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
    #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverDpotrs((a), (b), (c), (d), (double *)(e), (f), (double *)(g), (h), (double *)(i), (j), (k))
    #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverDpotri((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h))
    #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverDpotri_bufferSize((a), (b), (c), (double *)(d), (e), (f))
    #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverDsytrf((a), (b), (c), (double *)(d), (e), (f), (double *)(g), (h), (i))
    #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverDsytrf_bufferSize((a), (b), (double *)(c), (d), (e))
    #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverDgetrf((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h), (i))
    #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverDgetrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
    #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverDgetrs((a), (b), (c), (d), (double *)(e), (f), (g), (double *)(h), (i), (double *)(j), (k), (l))
    #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverDgeqrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
    #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverDgeqrf((a), (b), (c), (double *)(d), (e), (double *)(f), (double *)(g), (h), (i))
    #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverDormqr_bufferSize((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (l))
    #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverDormqr((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (double *)(l), (m), (n))
    #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasDtrsm((a), (b), (c), (d), (e), (f), (g), (double *)(h), (double *)(i), (j), (double *)(k), (l))
  #endif
#endif

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool    user_alloc;
  PetscScalar *unplacedarray; /* if one called MatHIPDensePlaceArray(), this is where it stashed the original */
  PetscBool    unplaced_user_alloc;
  /* factorization support */
  PetscHipBLASInt *d_fact_ipiv; /* device pivots */
  PetscScalar     *d_fact_tau;  /* device QR tau vector */
  PetscScalar     *d_fact_work; /* device workspace */
  PetscHipBLASInt  fact_lwork;
  PetscHipBLASInt *d_fact_info; /* device info */
  /* workspace */
  Vec workvec;
} Mat_SeqDenseHIP;

PetscErrorCode MatSeqDenseHIPSetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_SeqDense    *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscBool        iship;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSEHIP, &iship));
  if (!iship) PetscFunctionReturn(0);
  /* it may happen CPU preallocation has not been performed */
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (cA->lda <= 0) cA->lda = A->rmap->n;
  if (!dA->user_alloc) PetscCallHIP(hipFree(dA->d_v));
  if (!d_data) { /* petsc-allocated storage */
    size_t sz;
    PetscCall(PetscIntMultError(cA->lda, A->cmap->n, NULL));
    sz = cA->lda * A->cmap->n * sizeof(PetscScalar);
    PetscCallHIP(hipMalloc((void **)&dA->d_v, sz));
    PetscCallHIP(hipMemset(dA->d_v, 0, sz));
    dA->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    dA->d_v        = d_data;
    dA->user_alloc = PETSC_TRUE;
  }
  A->offloadmask  = PETSC_OFFLOAD_GPU;
  A->preallocated = PETSC_TRUE;
  A->assembled    = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseHIPCopyFromGPU(Mat A)
{
  Mat_SeqDense    *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQDENSEHIP);
  PetscCall(PetscInfo(A, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing", A->rmap->n, A->cmap->n));
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!cA->v) { /* MatCreateSeqDenseHIP may not allocate CPU memory. Allocate if needed */
      PetscCall(MatSeqDenseSetPreallocation(A, NULL));
    }
    PetscCall(PetscLogEventBegin(MAT_DenseCopyFromGPU, A, 0, 0, 0));
    if (cA->lda > A->rmap->n) {
      PetscCallHIP(hipMemcpy2D(cA->v, cA->lda * sizeof(PetscScalar), dA->d_v, cA->lda * sizeof(PetscScalar), A->rmap->n * sizeof(PetscScalar), A->cmap->n, hipMemcpyDeviceToHost));
    } else {
      PetscCallHIP(hipMemcpy(cA->v, dA->d_v, cA->lda * sizeof(PetscScalar) * A->cmap->n, hipMemcpyDeviceToHost));
    }
    PetscCall(PetscLogGpuToCpu(cA->lda * sizeof(PetscScalar) * A->cmap->n));
    PetscCall(PetscLogEventEnd(MAT_DenseCopyFromGPU, A, 0, 0, 0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseHIPCopyToGPU(Mat A)
{
  Mat_SeqDense    *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscBool        copy;

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQDENSEHIP);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  PetscCall(PetscInfo(A, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", A->rmap->n, A->cmap->n));
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      PetscCall(MatSeqDenseHIPSetPreallocation(A, NULL));
    }
    PetscCall(PetscLogEventBegin(MAT_DenseCopyToGPU, A, 0, 0, 0));
    if (cA->lda > A->rmap->n) {
      PetscCallHIP(hipMemcpy2D(dA->d_v, cA->lda * sizeof(PetscScalar), cA->v, cA->lda * sizeof(PetscScalar), A->rmap->n * sizeof(PetscScalar), A->cmap->n, hipMemcpyHostToDevice));
    } else {
      PetscCallHIP(hipMemcpy(dA->d_v, cA->v, cA->lda * sizeof(PetscScalar) * A->cmap->n, hipMemcpyHostToDevice));
    }
    PetscCall(PetscLogCpuToGpu(cA->lda * sizeof(PetscScalar) * A->cmap->n));
    PetscCall(PetscLogEventEnd(MAT_DenseCopyToGPU, A, 0, 0, 0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_SeqDenseHIP(Mat A, Mat B, MatStructure str)
{
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt           lda1, lda2, m = A->rmap->n, n = A->cmap->n;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    PetscCall(MatCopy_Basic(A, B, str));
    PetscFunctionReturn(0);
  }
  PetscCheck(m == B->rmap->n && n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
  PetscCall(MatDenseHIPGetArrayRead(A, &va));
  PetscCall(MatDenseHIPGetArrayWrite(B, &vb));
  PetscCall(MatDenseGetLDA(A, &lda1));
  PetscCall(MatDenseGetLDA(B, &lda2));
  PetscCall(PetscLogGpuTimeBegin());
  if (lda1 > m || lda2 > m) {
    PetscCallHIP(hipMemcpy2D(vb, lda2 * sizeof(PetscScalar), va, lda1 * sizeof(PetscScalar), m * sizeof(PetscScalar), n, hipMemcpyDeviceToDevice));
  } else {
    PetscCallHIP(hipMemcpy(vb, va, m * (n * sizeof(PetscScalar)), hipMemcpyDeviceToDevice));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArrayWrite(B, &vb));
  PetscCall(MatDenseHIPRestoreArrayRead(A, &va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqDenseHIP(Mat A)
{
  PetscScalar *va;
  PetscInt     lda, m = A->rmap->n, n = A->cmap->n;

  PetscFunctionBegin;
  PetscCall(MatDenseHIPGetArrayWrite(A, &va));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(PetscLogGpuTimeBegin());
  if (lda > m) {
    PetscCallHIP(hipMemset2D(va, lda * sizeof(PetscScalar), 0, m * sizeof(PetscScalar), n));
  } else {
    PetscCallHIP(hipMemset(va, 0, m * (n * sizeof(PetscScalar))));
  }
  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArrayWrite(A, &va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPPlaceArray_SeqDenseHIP(Mat A, const PetscScalar *a)
{
  Mat_SeqDense    *aa = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!aa->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!aa->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseHIPResetArray() must be called first");
  if (aa->v) PetscCall(MatSeqDenseHIPCopyToGPU(A));
  dA->unplacedarray       = dA->d_v;
  dA->unplaced_user_alloc = dA->user_alloc;
  dA->d_v                 = (PetscScalar *)a;
  dA->user_alloc          = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPResetArray_SeqDenseHIP(Mat A)
{
  Mat_SeqDense    *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (a->v) PetscCall(MatSeqDenseHIPCopyToGPU(A));
  dA->d_v           = dA->unplacedarray;
  dA->user_alloc    = dA->unplaced_user_alloc;
  dA->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPReplaceArray_SeqDenseHIP(Mat A, const PetscScalar *a)
{
  Mat_SeqDense    *aa = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!aa->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!aa->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseHIPResetArray() must be called first");
  if (!dA->user_alloc) PetscCallHIP(hipFree(dA->d_v));
  dA->d_v        = (PetscScalar *)a;
  dA->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArrayWrite_SeqDenseHIP(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  if (!dA->d_v) PetscCall(MatSeqDenseHIPSetPreallocation(A, NULL));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArrayWrite_SeqDenseHIP(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArrayRead_SeqDenseHIP(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseHIPCopyToGPU(A));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArrayRead_SeqDenseHIP(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArray_SeqDenseHIP(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseHIPCopyToGPU(A));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArray_SeqDenseHIP(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPInvertFactors_Private(Mat A)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar      *da;
  hipsolverHandle_t handle;
  PetscHipBLASInt   n, lda;
#if defined(PETSC_USE_DEBUG)
  PetscHipBLASInt info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscHipBLASIntCast(a->lda, &lda));
  PetscCheck(A->factortype != MAT_FACTOR_LU, PETSC_COMM_SELF, PETSC_ERR_LIB, "hipsolverDngetri not implemented");
  if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      PetscHipBLASInt il;

      PetscCall(MatDenseHIPGetArray(A, &da));
      PetscCallHIPSOLVER(hipsolverDnXpotri_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, &il));
      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        PetscCallHIP(hipFree(dA->d_fact_work));
        PetscCallHIP(hipMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
      }
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallHIPSOLVER(hipsolverDnXpotri(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(MatDenseHIPRestoreArray(A, &da));
      /* TODO (write hip kernel) */
      PetscCall(MatSeqDenseSymmetrize_Private(A, PETSC_TRUE));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "hipsolverDnsytri not implemented");
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Not implemented");
#if defined(PETSC_USE_DEBUG)
  PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
  PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: leading minor of order %d is zero", info);
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
#endif
  PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  PetscCall(PetscFree(A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Internal(Mat A, Vec xx, Vec yy, PetscBool transpose, PetscErrorCode (*matsolve)(Mat, PetscScalar *, PetscHipBLASInt, PetscHipBLASInt, PetscHipBLASInt, PetscHipBLASInt, PetscBool))
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar     *y;
  PetscHipBLASInt  m = 0, k = 0;
  PetscBool        xiship, yiship, aiship;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &k));
  PetscCall(PetscObjectTypeCompare((PetscObject)xx, VECSEQHIP, &xiship));
  PetscCall(PetscObjectTypeCompare((PetscObject)yy, VECSEQHIP, &yiship));
  {
    const PetscScalar *x;
    PetscBool          xishost = PETSC_TRUE;

    /* The logic here is to try to minimize the amount of memory copying:
       if we call VecHIPGetArrayRead(X,&x) every time xiship and the
       data is not offloaded to the GPU yet, then the data is copied to the
       GPU.  But we are only trying to get the data in order to copy it into the y
       array.  So the array x will be wherever the data already is so that
       only one memcpy is performed */
    if (xiship && xx->offloadmask & PETSC_OFFLOAD_GPU) {
      PetscCall(VecHIPGetArrayRead(xx, &x));
      xishost = PETSC_FALSE;
    } else PetscCall(VecGetArrayRead(xx, &x));
    if (k < m || !yiship) {
      if (!dA->workvec) PetscCall(VecCreateSeqHIP(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
      PetscCall(VecHIPGetArrayWrite(dA->workvec, &y));
    } else PetscCall(VecHIPGetArrayWrite(yy, &y));
    PetscCallHIP(hipMemcpy(y, x, m * sizeof(PetscScalar), xishost ? hipMemcpyHostToDevice : hipMemcpyDeviceToDevice));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSEHIP, &aiship));
  if (!aiship) PetscCall(MatConvert(A, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &A));
  PetscCall((*matsolve)(A, y, m, m, 1, k, transpose));
  if (!aiship) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (k < m || !yiship) {
    PetscScalar *yv;

    /* The logic here is that the data is not yet in either yy's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (yiship) PetscCall(VecHIPGetArrayWrite(yy, &yv));
    else PetscCall(VecGetArray(yy, &yv));
    PetscCallHIP(hipMemcpy(yv, y, k * sizeof(PetscScalar), yiship ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost));
    if (yiship) PetscCall(VecHIPRestoreArrayWrite(yy, &yv));
    else PetscCall(VecRestoreArray(yy, &yv));
    PetscCall(VecHIPRestoreArrayWrite(dA->workvec, &y));
  } else PetscCall(VecHIPRestoreArrayWrite(yy, &y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseHIP_Internal(Mat A, Mat B, Mat X, PetscBool transpose, PetscErrorCode (*matsolve)(Mat, PetscScalar *, PetscHipBLASInt, PetscHipBLASInt, PetscHipBLASInt, PetscHipBLASInt, PetscBool))
{
  PetscScalar    *y;
  PetscInt        n, _ldb, _ldx;
  PetscBool       biship, xiship, aiship;
  PetscHipBLASInt nrhs = 0, m = 0, k = 0, ldb = 0, ldx = 0, ldy = 0;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &k));
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(PetscHipBLASIntCast(n, &nrhs));
  PetscCall(MatDenseGetLDA(B, &_ldb));
  PetscCall(PetscHipBLASIntCast(_ldb, &ldb));
  PetscCall(MatDenseGetLDA(X, &_ldx));
  PetscCall(PetscHipBLASIntCast(_ldx, &ldx));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEHIP, &biship));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSEHIP, &xiship));
  /* The logic here is to try to minimize the amount of memory copying:
     if we call MatDenseHIPGetArrayRead(B,&b) every time biship and the
     data is not offloaded to the GPU yet, then the data is copied to the
     GPU.  But we are only trying to get the data in order to copy it into the y
     array.  So the array b will be wherever the data already is so that
     only one memcpy is performed */
  const PetscScalar *b;
  /* some copying from B will be involved */
  PetscBool bishost = PETSC_TRUE;
  if (biship && B->offloadmask & PETSC_OFFLOAD_GPU) {
    PetscCall(MatDenseHIPGetArrayRead(B, &b));
    bishost = PETSC_FALSE;
  } else PetscCall(MatDenseGetArrayRead(B, &b));
  if (ldx < m || !xiship) {
    /* X's array cannot serve as the array (too small or not on device), B's
     * array cannot serve as the array (const), so allocate a new array  */
    ldy = m;
    PetscCallHIP(hipMalloc((void **)&y, nrhs * m * sizeof(PetscScalar)));
  } else {
    /* X's array should serve as the array */
    ldy = ldx;
    PetscCall(MatDenseHIPGetArrayWrite(X, &y));
  }
  PetscCallHIP(hipMemcpy2D(y, ldy * sizeof(PetscScalar), b, ldb * sizeof(PetscScalar), m * sizeof(PetscScalar), nrhs, bishost ? hipMemcpyHostToDevice : hipMemcpyDeviceToDevice));
  if (bishost) PetscCall(MatDenseRestoreArrayRead(B, &b));
  else PetscCall(MatDenseHIPRestoreArrayRead(B, &b));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSEHIP, &aiship));
  if (!aiship) PetscCall(MatConvert(A, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &A));
  PetscCall((*matsolve)(A, y, ldy, m, nrhs, k, transpose));
  if (!aiship) PetscCall(MatConvert(A, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &A));
  if (ldx < m || !xiship) {
    PetscScalar *x;

    /* The logic here is that the data is not yet in either X's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (xiship) PetscCall(MatDenseHIPGetArrayWrite(X, &x));
    else PetscCall(MatDenseGetArray(X, &x));
    PetscCallHIP(hipMemcpy2D(x, ldx * sizeof(PetscScalar), y, ldy * sizeof(PetscScalar), k * sizeof(PetscScalar), nrhs, xiship ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost));
    if (xiship) PetscCall(MatDenseHIPRestoreArrayWrite(X, &x));
    else PetscCall(MatDenseRestoreArray(X, &x));
    PetscCallHIP(hipFree(y));
  } else PetscCall(MatDenseHIPRestoreArrayWrite(X, &y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Internal_LU(Mat A, PetscScalar *x, PetscHipBLASInt ldx, PetscHipBLASInt m, PetscHipBLASInt nrhs, PetscHipBLASInt k, PetscBool T)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP   *dA  = (Mat_SeqDenseHIP *)A->spptr;
  const PetscScalar *da;
  PetscHipBLASInt    lda;
  hipsolverHandle_t  handle;
  int                info;

  PetscFunctionBegin;
  PetscCall(MatDenseHIPGetArrayRead(A, &da));
  PetscCall(PetscHipBLASIntCast(mat->lda, &lda));
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "LU solve %d x %d on backend\n", m, k));
  PetscCallHIPSOLVER(hipsolverDnXgetrs(handle, T ? HIPSOLVER_OP_T : HIPSOLVER_OP_N, m, nrhs, da, lda, dA->d_fact_ipiv, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArrayRead(A, &da));
  if (PetscDefined(USE_DEBUG)) {
    PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
  }
  PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Internal_Cholesky(Mat A, PetscScalar *x, PetscHipBLASInt ldx, PetscHipBLASInt m, PetscHipBLASInt nrhs, PetscHipBLASInt k, PetscBool T)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP   *dA  = (Mat_SeqDenseHIP *)A->spptr;
  const PetscScalar *da;
  PetscHipBLASInt    lda;
  hipsolverHandle_t  handle;
  int                info;

  PetscFunctionBegin;
  PetscCall(MatDenseHIPGetArrayRead(A, &da));
  PetscCall(PetscHipBLASIntCast(mat->lda, &lda));
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "Cholesky solve %d x %d on backend\n", m, k));
  if (!dA->d_fact_ipiv) { /* spd */
    /* ========= Program hit hipErrorNotReady (error 34) due to "device not ready" on HIP API call to hipEventQuery. */
    PetscCallHIPSOLVER(hipsolverDnXpotrs(handle, HIPSOLVER_FILL_MODE_LOWER, m, nrhs, da, lda, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "hipsolverDnsytrs not implemented");
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArrayRead(A, &da));
  if (PetscDefined(USE_DEBUG)) {
    PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
  }
  PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Internal_QR(Mat A, PetscScalar *x, PetscHipBLASInt ldx, PetscHipBLASInt m, PetscHipBLASInt nrhs, PetscHipBLASInt k, PetscBool T)
{
  Mat_SeqDense        *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP     *dA  = (Mat_SeqDenseHIP *)A->spptr;
  const PetscScalar   *da;
  PetscHipBLASInt      lda, rank;
  hipsolverHandle_t    handle;
  hipblasHandle_t      bhandle;
  int                  info;
  hipsolverOperation_t trans;
  PetscScalar          one = 1.;

  PetscFunctionBegin;
  PetscCall(PetscHipBLASIntCast(mat->rank, &rank));
  PetscCall(MatDenseHIPGetArrayRead(A, &da));
  PetscCall(PetscHipBLASIntCast(mat->lda, &lda));
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(PetscHIPBLASGetHandle(&bhandle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "QR solve %d x %d on backend\n", m, k));
  if (!T) {
    if (PetscDefined(USE_COMPLEX)) trans = HIPSOLVER_OP_C;
    else trans = HIPSOLVER_OP_T;
    PetscCallHIPSOLVER(hipsolverDnXormqr(handle, HIPSOLVER_SIDE_LEFT, trans, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    if (PetscDefined(USE_DEBUG)) {
      PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
    }
    PetscCallHIPBLAS(hipblasXtrsm(bhandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
  } else {
    PetscCallHIPBLAS(hipblasXtrsm(bhandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
    PetscCallHIPSOLVER(hipsolverDnXormqr(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    if (PetscDefined(USE_DEBUG)) {
      PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArrayRead(A, &da));
  PetscCall(PetscLogFlops(nrhs * (4.0 * m * mat->rank - PetscSqr(mat->rank))));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_LU(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseHIP_LU(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseHIP_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_QR(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseHIP_QR(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseHIP_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseHIP_LU(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseHIP_LU(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseHIP_Cholesky(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseHIP_Cholesky(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseHIP_QR(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseHIP_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseHIP_QR(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseHIP_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseHIP_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseHIP(Mat A, IS rperm, IS cperm, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar      *da;
  PetscHipBLASInt   m, n, lda;
  hipsolverHandle_t handle;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(MatDenseHIPGetArray(A, &da));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(a->lda, &lda));
  PetscCall(PetscInfo(A, "LU factor %d x %d on backend\n", m, n));
  if (!dA->d_fact_ipiv) PetscCallHIP(hipMalloc((void **)&dA->d_fact_ipiv, n * sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    PetscCallHIPSOLVER(hipsolverDnXgetrf_bufferSize(handle, m, n, da, lda, &dA->fact_lwork));
    PetscCallHIP(hipMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) PetscCallHIP(hipMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPSOLVER(hipsolverDnXgetrf(handle, m, n, da, lda, dA->d_fact_work, dA->fact_lwork, dA->d_fact_ipiv, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
  PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
  PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
#endif
  A->factortype = MAT_FACTOR_LU;
  PetscCall(PetscLogGpuFlops(2.0 * n * n * m / 3.0));

  A->ops->solve             = MatSolve_SeqDenseHIP_LU;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseHIP_LU;
  A->ops->matsolve          = MatMatSolve_SeqDenseHIP_LU;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseHIP_LU;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHIP, &A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseHIP(Mat A, IS perm, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar      *da;
  PetscHipBLASInt   n, lda;
  hipsolverHandle_t handle;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &n));
  PetscCall(PetscInfo(A, "Cholesky factor %d x %d on backend\n", n, n));
  if (A->spd == PETSC_BOOL3_TRUE) {
    PetscCall(MatDenseHIPGetArray(A, &da));
    PetscCall(PetscHipBLASIntCast(a->lda, &lda));
    if (!dA->fact_lwork) {
      PetscCallHIPSOLVER(hipsolverDnXpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, &dA->fact_lwork));
      PetscCallHIP(hipMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
    }
    if (!dA->d_fact_info) PetscCallHIP(hipMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallHIPSOLVER(hipsolverDnXpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());

    PetscCall(MatDenseHIPRestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
    PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "hipsolverDnsytrs unavailable. Use MAT_FACTOR_LU");

  /* at the time of writing hipsolverDn has *sytrs and *hetr* routines implemented and the
       code below should work */
  if (!dA->d_fact_ipiv) PetscCallHIP(hipMalloc((void **)&dA->d_fact_ipiv, n * sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    PetscCallHIPSOLVER(hipsolverDnXsytrf_bufferSize(handle, n, da, lda, &dA->fact_lwork));
    PetscCallHIP(hipMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) PetscCallHIP(hipMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPSOLVER(hipsolverDnXsytrf(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, dA->d_fact_ipiv, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());

  A->ops->solve             = MatSolve_SeqDenseHIP_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseHIP_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDenseHIP_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseHIP_Cholesky;
  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHIP, &A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactor_SeqDenseHIP(Mat A, IS col, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar      *da;
  PetscHipBLASInt   m, min, max, n, lda;
  hipsolverHandle_t handle;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscHIPSOLVERGetHandle(&handle));
  PetscCall(MatDenseHIPGetArray(A, &da));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(a->lda, &lda));
  PetscCall(PetscInfo(A, "QR factor %d x %d on backend\n", m, n));
  max = PetscMax(m, n);
  min = PetscMin(m, n);
  if (!dA->d_fact_tau) PetscCallHIP(hipMalloc((void **)&dA->d_fact_tau, min * sizeof(*dA->d_fact_tau)));
  if (!dA->d_fact_ipiv) PetscCallHIP(hipMalloc((void **)&dA->d_fact_ipiv, n * sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    PetscCallHIPSOLVER(hipsolverDnXgeqrf_bufferSize(handle, m, n, da, lda, &dA->fact_lwork));
    PetscCallHIP(hipMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) PetscCallHIP(hipMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
  if (!dA->workvec) PetscCall(VecCreateSeqHIP(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPSOLVER(hipsolverDnXgeqrf(handle, m, n, da, lda, dA->d_fact_tau, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseHIPRestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
  PetscCallHIP(hipMemcpy(&info, dA->d_fact_info, sizeof(PetscHipBLASInt), hipMemcpyDeviceToHost));
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to hipSolver %d", -info);
#endif
  A->factortype = MAT_FACTOR_QR;
  a->rank       = min;
  PetscCall(PetscLogGpuFlops(2.0 * min * min * (max - min / 3.0)));

  A->ops->solve             = MatSolve_SeqDenseHIP_QR;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseHIP_QR;
  A->ops->matsolve          = MatMatSolve_SeqDenseHIP_QR;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseHIP_QR;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHIP, &A->solvertype));
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(Mat A, Mat B, Mat C, PetscBool tA, PetscBool tB)
{
  const PetscScalar *da, *db;
  PetscScalar       *dc;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscHipBLASInt    m, n, k;
  PetscInt           alda, blda, clda;
  hipblasHandle_t    hipblasv2handle;
  PetscBool          Aiship, Biship;

  PetscFunctionBegin;
  /* we may end up with SEQDENSE as one of the arguments */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSEHIP, &Aiship));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEHIP, &Biship));
  if (!Aiship) PetscCall(MatConvert(A, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &A));
  if (!Biship) PetscCall(MatConvert(B, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &B));
  PetscCall(PetscHipBLASIntCast(C->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(C->cmap->n, &n));
  if (tA) PetscCall(PetscHipBLASIntCast(A->rmap->n, &k));
  else PetscCall(PetscHipBLASIntCast(A->cmap->n, &k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  PetscCall(PetscInfo(C, "Matrix-Matrix product %d x %d x %d on backend\n", m, k, n));
  PetscCall(MatDenseHIPGetArrayRead(A, &da));
  PetscCall(MatDenseHIPGetArrayRead(B, &db));
  PetscCall(MatDenseHIPGetArrayWrite(C, &dc));
  PetscCall(MatDenseGetLDA(A, &alda));
  PetscCall(MatDenseGetLDA(B, &blda));
  PetscCall(MatDenseGetLDA(C, &clda));
  PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPBLAS(hipblasXgemm(hipblasv2handle, tA ? HIPBLAS_OP_T : HIPBLAS_OP_N, tB ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one, da, alda, db, blda, &zero, dc, clda));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(1.0 * m * n * k + 1.0 * m * n * (k - 1)));
  PetscCall(MatDenseHIPRestoreArrayRead(A, &da));
  PetscCall(MatDenseHIPRestoreArrayRead(B, &db));
  PetscCall(MatDenseHIPRestoreArrayWrite(C, &dc));
  if (!Aiship) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (!Biship) PetscCall(MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A, B, C, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A, B, C, PETSC_FALSE, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A, B, C, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqDenseHIP(Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatProductSetFromOptions_SeqDense(C));
  PetscFunctionReturn(0);
}

/* zz = op(A)*xx + yy
   if yy == NULL, only MatMult */
static PetscErrorCode MatMultAdd_SeqDenseHIP_Private(Mat A, Vec xx, Vec yy, Vec zz, PetscBool trans)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  const PetscScalar *xarray, *da;
  PetscScalar       *zarray;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscHipBLASInt    m, n, lda;
  hipblasHandle_t    hipblasv2handle;

  PetscFunctionBegin;
  if (yy && yy != zz) PetscCall(VecCopy_SeqHIP(yy, zz)); /* mult add */
  if (!A->rmap->n || !A->cmap->n) {
    if (!yy) PetscCall(VecSet_SeqHIP(zz, 0.0)); /* mult only */
    PetscFunctionReturn(0);
  }
  PetscCall(PetscInfo(A, "Matrix-vector product %d x %d on backend\n", A->rmap->n, A->cmap->n));
  PetscCall(PetscHipBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
  PetscCall(MatDenseHIPGetArrayRead(A, &da));
  PetscCall(PetscHipBLASIntCast(mat->lda, &lda));
  PetscCall(VecHIPGetArrayRead(xx, &xarray));
  PetscCall(VecHIPGetArray(zz, &zarray));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPBLAS(hipblasXgemv(hipblasv2handle, trans ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, &one, da, lda, xarray, 1, (yy ? &one : &zero), zarray, 1));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * A->rmap->n * A->cmap->n - (yy ? 0 : A->rmap->n)));
  PetscCall(VecHIPRestoreArrayRead(xx, &xarray));
  PetscCall(VecHIPRestoreArray(zz, &zarray));
  PetscCall(MatDenseHIPRestoreArrayRead(A, &da));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseHIP(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseHIP_Private(A, xx, yy, zz, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseHIP(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseHIP_Private(A, xx, yy, zz, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseHIP(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseHIP_Private(A, xx, NULL, yy, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseHIP(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseHIP_Private(A, xx, NULL, yy, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_SeqDenseHIP(Mat A, const PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseHIPCopyFromGPU(A));
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_SeqDenseHIP(Mat A, PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  if (!mat->v) PetscCall(MatSeqDenseSetPreallocation(A, NULL)); /* MatCreateSeqDenseHIP may not allocate CPU memory. Allocate if needed */
  *array         = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDenseHIP(Mat A, PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseHIPCopyFromGPU(A));
  *array         = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDenseHIP(Mat Y, PetscScalar alpha)
{
  Mat_SeqDense   *y = (Mat_SeqDense *)Y->data;
  PetscScalar    *dy;
  PetscHipBLASInt j, N, m, lday, one = 1;
  hipblasHandle_t hipblasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
  PetscCall(MatDenseHIPGetArray(Y, &dy));
  PetscCall(PetscHipBLASIntCast(Y->rmap->n * Y->cmap->n, &N));
  PetscCall(PetscHipBLASIntCast(Y->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(y->lda, &lday));
  PetscCall(PetscInfo(Y, "Performing Scale %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", Y->rmap->n, Y->cmap->n));
  PetscCall(PetscLogGpuTimeBegin());
  if (lday > m) {
    for (j = 0; j < Y->cmap->n; j++) PetscCallHIPBLAS(hipblasXscal(hipblasv2handle, m, &alpha, dy + lday * j, one));
  } else PetscCallHIPBLAS(hipblasXscal(hipblasv2handle, N, &alpha, dy, one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(N));
  PetscCall(MatDenseHIPRestoreArray(Y, &dy));
  PetscFunctionReturn(0);
}

struct petscshift : public thrust::unary_function<PetscScalar, PetscScalar> {
  const PetscScalar shift_;
  petscshift(PetscScalar shift) : shift_(shift) { }
  __device__ PetscScalar operator()(PetscScalar x) { return x + shift_; }
};

template <typename Iterator>
class strided_range {
public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  struct stride_functor : public thrust::unary_function<difference_type, difference_type> {
    difference_type stride;
    stride_functor(difference_type stride) : stride(stride) { }
    __device__ difference_type operator()(const difference_type &i) const { return stride * i; }
  };
  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator>    PermutationIterator;
  typedef PermutationIterator                                                   iterator; // type of the strided_range iterator
  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride) : first(first), last(last), stride(stride) { }
  iterator begin(void) const { return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride))); }
  iterator end(void) const { return begin() + ((last - first) + (stride - 1)) / stride; }

protected:
  Iterator        first;
  Iterator        last;
  difference_type stride;
};

PetscErrorCode MatShift_DenseHIP_Private(PetscScalar *da, PetscScalar alpha, PetscInt lda, PetscInt rstart, PetscInt rend, PetscInt cols)
{
  PetscFunctionBegin;
  PetscInt rend2 = PetscMin(rend, cols);
  if (rend2 > rstart) {
    PetscCall(PetscLogGpuTimeBegin());
    try {
      const auto                                                  dptr  = thrust::device_pointer_cast(da);
      size_t                                                      begin = rstart * lda;
      size_t                                                      end   = rend2 - rstart + rend2 * lda;
      strided_range<thrust::device_vector<PetscScalar>::iterator> diagonal(dptr + begin, dptr + end, lda + 1);
      thrust::transform(diagonal.begin(), diagonal.end(), diagonal.begin(), petscshift(alpha));
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Thrust error: %s", ex);
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(rend2 - rstart));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqDenseHIP(Mat A, PetscScalar alpha)
{
  PetscScalar *da;
  PetscInt     m = A->rmap->n, n = A->cmap->n, lda;

  PetscFunctionBegin;
  PetscCall(MatDenseHIPGetArray(A, &da));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(PetscInfo(A, "Performing Shift %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(MatShift_DenseHIP_Private(da, alpha, lda, 0, m, n));
  PetscCall(MatDenseHIPRestoreArray(A, &da));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseHIP(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense *)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense *)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  PetscHipBLASInt    j, N, m, ldax, lday, one = 1;
  hipblasHandle_t    hipblasv2handle;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
  PetscCall(MatDenseHIPGetArrayRead(X, &dx));
  if (alpha == 0.0) PetscCall(MatDenseHIPGetArrayWrite(Y, &dy));
  else PetscCall(MatDenseHIPGetArray(Y, &dy));
  PetscCall(PetscHipBLASIntCast(X->rmap->n * X->cmap->n, &N));
  PetscCall(PetscHipBLASIntCast(X->rmap->n, &m));
  PetscCall(PetscHipBLASIntCast(x->lda, &ldax));
  PetscCall(PetscHipBLASIntCast(y->lda, &lday));
  PetscCall(PetscInfo(Y, "Performing AXPY %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", Y->rmap->n, Y->cmap->n));
  PetscCall(PetscLogGpuTimeBegin());
  if (ldax > m || lday > m) {
    for (j = 0; j < X->cmap->n; j++) PetscCallHIPBLAS(hipblasXaxpy(hipblasv2handle, m, &alpha, dx + j * ldax, one, dy + j * lday, one));
  } else PetscCallHIPBLAS(hipblasXaxpy(hipblasv2handle, N, &alpha, dx, one, dy, one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(PetscMax(2. * N - 1, 0)));
  PetscCall(MatDenseHIPRestoreArrayRead(X, &dx));
  if (alpha == 0.0) PetscCall(MatDenseHIPRestoreArrayWrite(Y, &dy));
  else PetscCall(MatDenseHIPRestoreArray(Y, &dy));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseHIP(Mat A)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  if (dA) {
    PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseHIPResetArray() must be called first");
    if (!dA->user_alloc) PetscCallHIP(hipFree(dA->d_v));
    PetscCallHIP(hipFree(dA->d_fact_tau));
    PetscCallHIP(hipFree(dA->d_fact_ipiv));
    PetscCallHIP(hipFree(dA->d_fact_info));
    PetscCallHIP(hipFree(dA->d_fact_work));
    PetscCall(VecDestroy(&dA->workvec));
  }
  PetscCall(PetscFree(A->spptr));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseHIP(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(MatConvert_SeqDenseHIP_SeqDense(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseHIP(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  MatDuplicateOption hcpvalues = (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) ? MAT_DO_NOT_COPY_VALUES : cpvalues;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  PetscCall(MatSetType(*B, ((PetscObject)A)->type_name));
  PetscCall(MatDuplicateNoCreate_SeqDense(*B, A, hcpvalues));
  if (cpvalues == MAT_COPY_VALUES && hcpvalues != MAT_COPY_VALUES) PetscCall(MatCopy_SeqDenseHIP(A, *B, SAME_NONZERO_PATTERN));
  if (cpvalues != MAT_COPY_VALUES) { /* allocate memory if needed */
    Mat_SeqDenseHIP *dB = (Mat_SeqDenseHIP *)(*B)->spptr;
    if (!dB->d_v) PetscCall(MatSeqDenseHIPSetPreallocation(*B, NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_SeqDenseHIP(Mat A, Vec v, PetscInt col)
{
  Mat_SeqDense    *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscScalar     *x;
  PetscBool        viship;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)v, &viship, VECSEQHIP, VECMPIHIP, VECHIP, ""));
  if (viship && !v->boundtocpu) { /* update device data */
    PetscCall(VecHIPGetArrayWrite(v, &x));
    if (A->offloadmask & PETSC_OFFLOAD_GPU) PetscCallHIP(hipMemcpy(x, dA->d_v + col * a->lda, A->rmap->n * sizeof(PetscScalar), hipMemcpyHostToHost));
    else PetscCallHIP(hipMemcpy(x, a->v + col * a->lda, A->rmap->n * sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCall(VecHIPRestoreArrayWrite(v, &x));
  } else { /* update host data */
    PetscCall(VecGetArrayWrite(v, &x));
    if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask & PETSC_OFFLOAD_CPU) PetscCall(PetscArraycpy(x, a->v + col * a->lda, A->rmap->n));
    else if (A->offloadmask & PETSC_OFFLOAD_GPU) PetscCallHIP(hipMemcpy(x, dA->d_v + col * a->lda, A->rmap->n * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    PetscCall(VecRestoreArrayWrite(v, &x));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_hip(Mat A, MatFactorType ftype, Mat *fact)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), fact));
  PetscCall(MatSetSizes(*fact, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  PetscCall(MatSetType(*fact, MATSEQDENSEHIP));
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqDense;
    (*fact)->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_QR) {
    PetscCall(PetscObjectComposeFunction((PetscObject)(*fact), "MatQRFactor_C", MatQRFactor_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)(*fact), "MatQRFactorSymbolic_C", MatQRFactorSymbolic_SeqDense));
  }
  (*fact)->factortype = ftype;
  PetscCall(PetscFree((*fact)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHIP, &(*fact)->solvertype));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_LU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_ILU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_ICC]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseHIPGetArray(A, (PetscScalar **)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    PetscCall(VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A, (PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecHIPPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecHIPResetArray(a->cvec));
  PetscCall(MatDenseHIPRestoreArray(A, (PetscScalar **)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseHIPGetArrayRead(A, &a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    PetscCall(VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A, (PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecHIPPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  PetscCall(VecLockReadPush(a->cvec));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecHIPResetArray(a->cvec));
  PetscCall(MatDenseHIPRestoreArrayRead(A, &a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseHIPGetArrayWrite(A, (PetscScalar **)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    PetscCall(VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A, (PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecHIPPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDenseHIP(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecHIPResetArray(a->cvec));
  PetscCall(MatDenseHIPRestoreArrayWrite(A, (PetscScalar **)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseHIP(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *v)
{
  Mat_SeqDense    *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && (cend - cbegin != a->cmat->cmap->N || rend - rbegin != a->cmat->rmap->N)) PetscCall(MatDestroy(&a->cmat));
  PetscCall(MatSeqDenseHIPCopyToGPU(A));
  if (!a->cmat) {
    PetscCall(MatCreateDenseHIP(PetscObjectComm((PetscObject)A), rend - rbegin, PETSC_DECIDE, rend - rbegin, cend - cbegin, dA->d_v + rbegin + (size_t)cbegin * a->lda, &a->cmat));
    PetscCall(PetscLogObjectParent((PetscObject)A, (PetscObject)a->cmat));
  } else PetscCall(MatDenseHIPPlaceArray(a->cmat, dA->d_v + rbegin + (size_t)cbegin * a->lda));
  PetscCall(MatDenseSetLDA(a->cmat, a->lda));
  /* Place CPU array if present but not copy any data */
  a->cmat->offloadmask = PETSC_OFFLOAD_GPU;
  if (a->v) { PetscCall(MatDensePlaceArray(a->cmat, a->v + rbegin + (size_t)cbegin * a->lda)); }
  a->cmat->offloadmask = A->offloadmask;
  a->matinuse          = cbegin + 1;
  *v                   = a->cmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseHIP(Mat A, Mat *v)
{
  Mat_SeqDense    *a    = (Mat_SeqDense *)A->data;
  PetscBool        copy = PETSC_FALSE, reset;
  PetscOffloadMask suboff;

  PetscFunctionBegin;
  PetscCheck(a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetSubMatrix() first");
  PetscCheck(a->cmat, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column matrix");
  PetscCheck(*v == a->cmat, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  reset       = a->v ? PETSC_TRUE : PETSC_FALSE;
  suboff      = a->cmat->offloadmask; /* calls to ResetArray may change it, so save it here */
  if (suboff == PETSC_OFFLOAD_CPU && !a->v) {
    copy = PETSC_TRUE;
    PetscCall(MatSeqDenseSetPreallocation(A, NULL));
  }
  PetscCall(MatDenseHIPResetArray(a->cmat));
  if (reset) PetscCall(MatDenseResetArray(a->cmat));
  if (copy) {
    PetscCall(MatSeqDenseHIPCopyFromGPU(A));
  } else A->offloadmask = (suboff == PETSC_OFFLOAD_CPU) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  a->cmat->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseSetLDA_SeqDenseHIP(Mat A, PetscInt lda)
{
  Mat_SeqDense    *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP *)A->spptr;
  PetscBool        data;

  PetscFunctionBegin;
  data = (PetscBool)((A->rmap->n > 0 && A->cmap->n > 0) ? (dA->d_v ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE);
  PetscCheck(dA->user_alloc || data || cA->lda == lda, PETSC_COMM_SELF, PETSC_ERR_ORDER, "LDA cannot be changed after allocation of internal storage");
  PetscCheck(lda >= A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "LDA %" PetscInt_FMT " must be at least matrix dimension %" PetscInt_FMT, lda, A->rmap->n);
  cA->lda = lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_SeqDenseHIP(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) PetscCall(MatSeqDenseHIPSetPreallocation(A, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqDenseHIP(Mat A, PetscBool flg)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = flg;
  if (!flg) {
    PetscBool iship;

    PetscCall(PetscObjectTypeCompare((PetscObject)a->cvec, VECSEQHIP, &iship));
    if (!iship) PetscCall(VecDestroy(&a->cvec));
    PetscCall(PetscObjectTypeCompare((PetscObject)a->cmat, MATSEQDENSEHIP, &iship));
    if (!iship) PetscCall(MatDestroy(&a->cmat));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArray_C", MatDenseGetArray_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayRead_C", MatDenseGetArrayRead_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWrite_C", MatDenseGetArrayWrite_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDenseHIP));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatQRFactor_C", MatQRFactor_SeqDenseHIP));

    A->ops->duplicate               = MatDuplicate_SeqDenseHIP;
    A->ops->mult                    = MatMult_SeqDenseHIP;
    A->ops->multadd                 = MatMultAdd_SeqDenseHIP;
    A->ops->multtranspose           = MatMultTranspose_SeqDenseHIP;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDenseHIP;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->axpy                    = MatAXPY_SeqDenseHIP;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDenseHIP;
    A->ops->lufactor                = MatLUFactor_SeqDenseHIP;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDenseHIP;
    A->ops->getcolumnvector         = MatGetColumnVector_SeqDenseHIP;
    A->ops->scale                   = MatScale_SeqDenseHIP;
    A->ops->shift                   = MatShift_SeqDenseHIP;
    A->ops->copy                    = MatCopy_SeqDenseHIP;
    A->ops->zeroentries             = MatZeroEntries_SeqDenseHIP;
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    PetscCall(MatSeqDenseHIPCopyFromGPU(A));
    PetscCall(PetscFree(A->defaultrandtype));
    PetscCall(PetscStrallocpy(PETSCRANDER48, &A->defaultrandtype));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArray_C", MatDenseGetArray_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayRead_C", MatDenseGetArray_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWrite_C", MatDenseGetArray_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatQRFactor_C", MatQRFactor_SeqDense));

    A->ops->duplicate               = MatDuplicate_SeqDense;
    A->ops->mult                    = MatMult_SeqDense;
    A->ops->multadd                 = MatMultAdd_SeqDense;
    A->ops->multtranspose           = MatMultTranspose_SeqDense;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDense;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDense;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDense_SeqDense;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDense_SeqDense;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDense_SeqDense;
    A->ops->axpy                    = MatAXPY_SeqDense;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDense;
    A->ops->lufactor                = MatLUFactor_SeqDense;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDense;
    A->ops->getcolumnvector         = MatGetColumnVector_SeqDense;
    A->ops->scale                   = MatScale_SeqDense;
    A->ops->shift                   = MatShift_SeqDense;
    A->ops->copy                    = MatCopy_SeqDense;
    A->ops->zeroentries             = MatZeroEntries_SeqDense;
  }
  if (a->cmat) { PetscCall(MatBindToCPU(a->cmat, flg)); }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDenseHIP_SeqDense(Mat M, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqDense *a;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
    PetscFunctionReturn(0);
  }

  B = *newmat;
  PetscCall(MatBindToCPU_SeqDenseHIP(B, PETSC_TRUE));
  PetscCall(MatReset_SeqDenseHIP(B));
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECSTANDARD, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensehip_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArrayRead_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArrayWrite_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArrayRead_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArrayWrite_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPPlaceArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPResetArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPReplaceArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensehip_C", NULL));
  a = (Mat_SeqDense *)B->data;
  PetscCall(VecDestroy(&a->cvec)); /* cvec might be VECSEQHIP. Destroy it and rebuild a VECSEQ when needed */
  B->ops->bindtocpu = NULL;
  B->ops->destroy   = MatDestroy_SeqDense;
  B->offloadmask    = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseHIP(Mat M, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat_SeqDenseHIP *dB;
  Mat_SeqDense    *a;
  Mat              B;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
    PetscFunctionReturn(0);
  }

  B = *newmat;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECHIP, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSEHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensehip_seqdense_C", MatConvert_SeqDenseHIP_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArray_C", MatDenseHIPGetArray_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArrayRead_C", MatDenseHIPGetArrayRead_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPGetArrayWrite_C", MatDenseHIPGetArrayWrite_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArray_C", MatDenseHIPRestoreArray_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArrayRead_C", MatDenseHIPRestoreArrayRead_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPRestoreArrayWrite_C", MatDenseHIPRestoreArrayWrite_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPPlaceArray_C", MatDenseHIPPlaceArray_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPResetArray_C", MatDenseHIPResetArray_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseHIPReplaceArray_C", MatDenseHIPReplaceArray_SeqDenseHIP));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensehip_C", MatProductSetFromOptions_SeqAIJ_SeqDense));
  a = (Mat_SeqDense *)B->data;
  PetscCall(VecDestroy(&a->cvec)); /* cvec might be VECSEQ. Destroy it and rebuild a VECSEQHIP when needed */
  PetscCall(PetscNewLog(B, &dB));
  B->spptr       = dB;
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscCall(MatBindToCPU_SeqDenseHIP(B, PETSC_FALSE));
  B->ops->bindtocpu = MatBindToCPU_SeqDenseHIP;
  B->ops->destroy   = MatDestroy_SeqDenseHIP;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqDenseHIP - Creates a sequential matrix in dense format using HIP.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of rows
.  n - number of columns
-  data - optional location of GPU matrix data.  Set data=NULL for PETSc
   to control matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:

   Level: intermediate

.seealso: `MATSEQDENSE`, `MatCreate()`, `MatCreateSeqDense()`
@*/
PetscErrorCode MatCreateSeqDenseHIP(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, comm, PETSC_ERR_ARG_WRONG, "Invalid communicator size %d", size);
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQDENSEHIP));
  PetscCall(MatSeqDenseHIPSetPreallocation(*A, data));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSEHIP - MATSEQDENSEHIP = "seqdensehip" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensehip - sets the matrix type to `MATSEQDENSEHIP` during a call to `MatSetFromOptions()`

  Level: beginner

.seealso: `MATSEQDENSE`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseHIP(Mat B)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  PetscCall(MatCreate_SeqDense(B));
  PetscCall(MatConvert_SeqDense_SeqDenseHIP(B, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(0);
}
