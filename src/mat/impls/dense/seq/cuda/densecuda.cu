/*
     Defines the matrix operations for sequential dense with CUDA
*/
#include <petscpkg_version.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsc/private/cudavecimpl.h>        /* cublas definitions are here */
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnCpotrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
    #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCpotrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
    #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnCpotrs((a), (b), (c), (d), (cuComplex *)(e), (f), (cuComplex *)(g), (h), (i))
    #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnCpotri((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
    #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnCpotri_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
    #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnCsytrf((a), (b), (c), (cuComplex *)(d), (e), (f), (cuComplex *)(g), (h), (i))
    #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnCsytrf_bufferSize((a), (b), (cuComplex *)(c), (d), (e))
    #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnCgetrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
    #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCgetrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
    #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnCgetrs((a), (b), (c), (d), (cuComplex *)(e), (f), (g), (cuComplex *)(h), (i), (j))
    #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCgeqrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
    #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnCgeqrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (cuComplex *)(g), (h), (i))
    #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnCunmqr_bufferSize((a), (b), (c), (d), (e), (f), (cuComplex *)(g), (h), (cuComplex *)(i), (cuComplex *)(j), (k), (l))
    #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnCunmqr((a), (b), (c), (d), (e), (f), (cuComplex *)(g), (h), (cuComplex *)(i), (cuComplex *)(j), (k), (cuComplex *)(l), (m), (n))
    #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasCtrsm((a), (b), (c), (d), (e), (f), (g), (cuComplex *)(h), (cuComplex *)(i), (j), (cuComplex *)(k), (l))
  #else /* complex double */
    #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnZpotrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
    #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZpotrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
    #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnZpotrs((a), (b), (c), (d), (cuDoubleComplex *)(e), (f), (cuDoubleComplex *)(g), (h), (i))
    #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnZpotri((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
    #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnZpotri_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
    #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnZsytrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (f), (cuDoubleComplex *)(g), (h), (i))
    #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnZsytrf_bufferSize((a), (b), (cuDoubleComplex *)(c), (d), (e))
    #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnZgetrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
    #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZgetrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
    #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnZgetrs((a), (b), (c), (d), (cuDoubleComplex *)(e), (f), (g), (cuDoubleComplex *)(h), (i), (j))
    #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZgeqrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
    #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnZgeqrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (cuDoubleComplex *)(g), (h), (i))
    #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnZunmqr_bufferSize((a), (b), (c), (d), (e), (f), (cuDoubleComplex *)(g), (h), (cuDoubleComplex *)(i), (cuDoubleComplex *)(j), (k), (l))
    #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnZunmqr((a), (b), (c), (d), (e), (f), (cuDoubleComplex *)(g), (h), (cuDoubleComplex *)(i), (cuDoubleComplex *)(j), (k), (cuDoubleComplex *)(l), (m), (n))
    #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasZtrsm((a), (b), (c), (d), (e), (f), (g), (cuDoubleComplex *)(h), (cuDoubleComplex *)(i), (j), (cuDoubleComplex *)(k), (l))
  #endif
#else /* real single */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnSpotrf((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSpotrf_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnSpotrs((a), (b), (c), (d), (e), (f), (g), (h), (i))
    #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnSpotri((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnSpotri_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnSsytrf((a), (b), (c), (d), (e), (f), (g), (h), (i))
    #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnSsytrf_bufferSize((a), (b), (c), (d), (e))
    #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnSgetrf((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSgetrf_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnSgetrs((a), (b), (c), (d), (e), (f), (g), (h), (i), (j))
    #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSgeqrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
    #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnSgeqrf((a), (b), (c), (float *)(d), (e), (float *)(f), (float *)(g), (h), (i))
    #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnSormqr_bufferSize((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (l))
    #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnSormqr((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (float *)(l), (m), (n))
    #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasStrsm((a), (b), (c), (d), (e), (f), (g), (float *)(h), (float *)(i), (j), (float *)(k), (l))
  #else /* real double */
    #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnDpotrf((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDpotrf_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnDpotrs((a), (b), (c), (d), (e), (f), (g), (h), (i))
    #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnDpotri((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnDpotri_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnDsytrf((a), (b), (c), (d), (e), (f), (g), (h), (i))
    #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnDsytrf_bufferSize((a), (b), (c), (d), (e))
    #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnDgetrf((a), (b), (c), (d), (e), (f), (g), (h))
    #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDgetrf_bufferSize((a), (b), (c), (d), (e), (f))
    #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnDgetrs((a), (b), (c), (d), (e), (f), (g), (h), (i), (j))
    #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDgeqrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
    #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnDgeqrf((a), (b), (c), (double *)(d), (e), (double *)(f), (double *)(g), (h), (i))
    #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnDormqr_bufferSize((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (l))
    #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnDormqr((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (double *)(l), (m), (n))
    #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasDtrsm((a), (b), (c), (d), (e), (f), (g), (double *)(h), (double *)(i), (j), (double *)(k), (l))
  #endif
#endif

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool    user_alloc;
  PetscScalar *unplacedarray; /* if one called MatCUDADensePlaceArray(), this is where it stashed the original */
  PetscBool    unplaced_user_alloc;
  /* factorization support */
  PetscCuBLASInt *d_fact_ipiv; /* device pivots */
  PetscScalar    *d_fact_tau;  /* device QR tau vector */
  PetscScalar    *d_fact_work; /* device workspace */
  PetscCuBLASInt  fact_lwork;
  PetscCuBLASInt *d_fact_info; /* device info */
  /* workspace */
  Vec workvec;
} Mat_SeqDenseCUDA;

PetscErrorCode MatSeqDenseCUDASetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_SeqDense     *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscBool         iscuda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSECUDA, &iscuda));
  if (!iscuda) PetscFunctionReturn(0);
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  /* it may happen CPU preallocation has not been performed */
  if (cA->lda <= 0) cA->lda = A->rmap->n;
  if (!dA->user_alloc) PetscCallCUDA(cudaFree(dA->d_v));
  if (!d_data) { /* petsc-allocated storage */
    size_t sz;

    PetscCall(PetscIntMultError(cA->lda, A->cmap->n, NULL));
    sz = cA->lda * A->cmap->n * sizeof(PetscScalar);
    PetscCallCUDA(cudaMalloc((void **)&dA->d_v, sz));
    PetscCallCUDA(cudaMemset(dA->d_v, 0, sz));
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

PetscErrorCode MatSeqDenseCUDACopyFromGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQDENSECUDA);
  PetscCall(PetscInfo(A, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing", A->rmap->n, A->cmap->n));
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!cA->v) { /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
      PetscCall(MatSeqDenseSetPreallocation(A, NULL));
    }
    PetscCall(PetscLogEventBegin(MAT_DenseCopyFromGPU, A, 0, 0, 0));
    if (cA->lda > A->rmap->n) {
      PetscCallCUDA(cudaMemcpy2D(cA->v, cA->lda * sizeof(PetscScalar), dA->d_v, cA->lda * sizeof(PetscScalar), A->rmap->n * sizeof(PetscScalar), A->cmap->n, cudaMemcpyDeviceToHost));
    } else {
      PetscCallCUDA(cudaMemcpy(cA->v, dA->d_v, cA->lda * sizeof(PetscScalar) * A->cmap->n, cudaMemcpyDeviceToHost));
    }
    PetscCall(PetscLogGpuToCpu(cA->lda * sizeof(PetscScalar) * A->cmap->n));
    PetscCall(PetscLogEventEnd(MAT_DenseCopyFromGPU, A, 0, 0, 0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseCUDACopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscBool         copy;

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQDENSECUDA);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  PetscCall(PetscInfo(A, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", A->rmap->n, A->cmap->n));
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      PetscCall(MatSeqDenseCUDASetPreallocation(A, NULL));
    }
    PetscCall(PetscLogEventBegin(MAT_DenseCopyToGPU, A, 0, 0, 0));
    if (cA->lda > A->rmap->n) {
      PetscCallCUDA(cudaMemcpy2D(dA->d_v, cA->lda * sizeof(PetscScalar), cA->v, cA->lda * sizeof(PetscScalar), A->rmap->n * sizeof(PetscScalar), A->cmap->n, cudaMemcpyHostToDevice));
    } else {
      PetscCallCUDA(cudaMemcpy(dA->d_v, cA->v, cA->lda * sizeof(PetscScalar) * A->cmap->n, cudaMemcpyHostToDevice));
    }
    PetscCall(PetscLogCpuToGpu(cA->lda * sizeof(PetscScalar) * A->cmap->n));
    PetscCall(PetscLogEventEnd(MAT_DenseCopyToGPU, A, 0, 0, 0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_SeqDenseCUDA(Mat A, Mat B, MatStructure str)
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
  PetscCall(MatDenseCUDAGetArrayRead(A, &va));
  PetscCall(MatDenseCUDAGetArrayWrite(B, &vb));
  PetscCall(MatDenseGetLDA(A, &lda1));
  PetscCall(MatDenseGetLDA(B, &lda2));
  PetscCall(PetscLogGpuTimeBegin());
  if (lda1 > m || lda2 > m) {
    PetscCallCUDA(cudaMemcpy2D(vb, lda2 * sizeof(PetscScalar), va, lda1 * sizeof(PetscScalar), m * sizeof(PetscScalar), n, cudaMemcpyDeviceToDevice));
  } else {
    PetscCallCUDA(cudaMemcpy(vb, va, m * (n * sizeof(PetscScalar)), cudaMemcpyDeviceToDevice));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArrayWrite(B, &vb));
  PetscCall(MatDenseCUDARestoreArrayRead(A, &va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqDenseCUDA(Mat A)
{
  PetscScalar *va;
  PetscInt     lda, m = A->rmap->n, n = A->cmap->n;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArrayWrite(A, &va));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(PetscLogGpuTimeBegin());
  if (lda > m) {
    PetscCallCUDA(cudaMemset2D(va, lda * sizeof(PetscScalar), 0, m * sizeof(PetscScalar), n));
  } else {
    PetscCallCUDA(cudaMemset(va, 0, m * (n * sizeof(PetscScalar))));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArrayWrite(A, &va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAPlaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!aa->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!aa->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseCUDAResetArray() must be called first");
  if (aa->v) PetscCall(MatSeqDenseCUDACopyToGPU(A));
  dA->unplacedarray       = dA->d_v;
  dA->unplaced_user_alloc = dA->user_alloc;
  dA->d_v                 = (PetscScalar *)a;
  dA->user_alloc          = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAResetArray_SeqDenseCUDA(Mat A)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (a->v) PetscCall(MatSeqDenseCUDACopyToGPU(A));
  dA->d_v           = dA->unplacedarray;
  dA->user_alloc    = dA->unplaced_user_alloc;
  dA->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAReplaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!aa->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!aa->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseCUDAResetArray() must be called first");
  if (!dA->user_alloc) PetscCallCUDA(cudaFree(dA->d_v));
  dA->d_v        = (PetscScalar *)a;
  dA->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  if (!dA->d_v) PetscCall(MatSeqDenseCUDASetPreallocation(A, NULL));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayRead_SeqDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseCUDACopyToGPU(A));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayRead_SeqDenseCUDA(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArray_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseCUDACopyToGPU(A));
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArray_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseCUDAInvertFactors_Private(Mat A)
{
#if PETSC_PKG_CUDA_VERSION_GE(10, 1, 0)
  Mat_SeqDense      *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA  *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar       *da;
  cusolverDnHandle_t handle;
  PetscCuBLASInt     n, lda;
  #if defined(PETSC_USE_DEBUG)
  PetscCuBLASInt info;
  #endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscCuBLASIntCast(a->lda, &lda));
  PetscCheck(A->factortype != MAT_FACTOR_LU, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDngetri not implemented");
  if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      PetscCuBLASInt il;

      PetscCall(MatDenseCUDAGetArray(A, &da));
      PetscCallCUSOLVER(cusolverDnXpotri_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, da, lda, &il));
      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        PetscCallCUDA(cudaFree(dA->d_fact_work));
        PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
      }
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUSOLVER(cusolverDnXpotri(handle, CUBLAS_FILL_MODE_LOWER, n, da, lda, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(MatDenseCUDARestoreArray(A, &da));
      /* TODO (write cuda kernel) */
      PetscCall(MatSeqDenseSymmetrize_Private(A, PETSC_TRUE));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnsytri not implemented");
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Not implemented");
  #if defined(PETSC_USE_DEBUG)
  PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: leading minor of order %d is zero", info);
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
  #endif
  PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  PetscCall(PetscFree(A->solvertype));
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Upgrade to CUDA version 10.1.0 or higher");
#endif
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal(Mat A, Vec xx, Vec yy, PetscBool transpose, PetscErrorCode (*matsolve)(Mat, PetscScalar *, PetscCuBLASInt, PetscCuBLASInt, PetscCuBLASInt, PetscCuBLASInt, PetscBool))
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar      *y;
  PetscCuBLASInt    m = 0, k = 0;
  PetscBool         xiscuda, yiscuda, aiscuda;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &k));
  PetscCall(PetscObjectTypeCompare((PetscObject)xx, VECSEQCUDA, &xiscuda));
  PetscCall(PetscObjectTypeCompare((PetscObject)yy, VECSEQCUDA, &yiscuda));
  {
    const PetscScalar *x;
    PetscBool          xishost = PETSC_TRUE;

    /* The logic here is to try to minimize the amount of memory copying:
       if we call VecCUDAGetArrayRead(X,&x) every time xiscuda and the
       data is not offloaded to the GPU yet, then the data is copied to the
       GPU.  But we are only trying to get the data in order to copy it into the y
       array.  So the array x will be wherever the data already is so that
       only one memcpy is performed */
    if (xiscuda && xx->offloadmask & PETSC_OFFLOAD_GPU) {
      PetscCall(VecCUDAGetArrayRead(xx, &x));
      xishost = PETSC_FALSE;
    } else {
      PetscCall(VecGetArrayRead(xx, &x));
    }
    if (k < m || !yiscuda) {
      if (!dA->workvec) PetscCall(VecCreateSeqCUDA(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
      PetscCall(VecCUDAGetArrayWrite(dA->workvec, &y));
    } else {
      PetscCall(VecCUDAGetArrayWrite(yy, &y));
    }
    PetscCallCUDA(cudaMemcpy(y, x, m * sizeof(PetscScalar), xishost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSECUDA, &aiscuda));
  if (!aiscuda) PetscCall(MatConvert(A, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &A));
  PetscCall((*matsolve)(A, y, m, m, 1, k, transpose));
  if (!aiscuda) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (k < m || !yiscuda) {
    PetscScalar *yv;

    /* The logic here is that the data is not yet in either yy's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (yiscuda) {
      PetscCall(VecCUDAGetArrayWrite(yy, &yv));
    } else {
      PetscCall(VecGetArray(yy, &yv));
    }
    PetscCallCUDA(cudaMemcpy(yv, y, k * sizeof(PetscScalar), yiscuda ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));
    if (yiscuda) {
      PetscCall(VecCUDARestoreArrayWrite(yy, &yv));
    } else {
      PetscCall(VecRestoreArray(yy, &yv));
    }
    PetscCall(VecCUDARestoreArrayWrite(dA->workvec, &y));
  } else {
    PetscCall(VecCUDARestoreArrayWrite(yy, &y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_Internal(Mat A, Mat B, Mat X, PetscBool transpose, PetscErrorCode (*matsolve)(Mat, PetscScalar *, PetscCuBLASInt, PetscCuBLASInt, PetscCuBLASInt, PetscCuBLASInt, PetscBool))
{
  PetscScalar   *y;
  PetscInt       n, _ldb, _ldx;
  PetscBool      biscuda, xiscuda, aiscuda;
  PetscCuBLASInt nrhs = 0, m = 0, k = 0, ldb = 0, ldx = 0, ldy = 0;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &k));
  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(PetscCuBLASIntCast(n, &nrhs));
  PetscCall(MatDenseGetLDA(B, &_ldb));
  PetscCall(PetscCuBLASIntCast(_ldb, &ldb));
  PetscCall(MatDenseGetLDA(X, &_ldx));
  PetscCall(PetscCuBLASIntCast(_ldx, &ldx));

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &biscuda));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSECUDA, &xiscuda));
  {
    /* The logic here is to try to minimize the amount of memory copying:
       if we call MatDenseCUDAGetArrayRead(B,&b) every time biscuda and the
       data is not offloaded to the GPU yet, then the data is copied to the
       GPU.  But we are only trying to get the data in order to copy it into the y
       array.  So the array b will be wherever the data already is so that
       only one memcpy is performed */
    const PetscScalar *b;

    /* some copying from B will be involved */
    PetscBool bishost = PETSC_TRUE;

    if (biscuda && B->offloadmask & PETSC_OFFLOAD_GPU) {
      PetscCall(MatDenseCUDAGetArrayRead(B, &b));
      bishost = PETSC_FALSE;
    } else {
      PetscCall(MatDenseGetArrayRead(B, &b));
    }
    if (ldx < m || !xiscuda) {
      /* X's array cannot serve as the array (too small or not on device), B's
       * array cannot serve as the array (const), so allocate a new array  */
      ldy = m;
      PetscCallCUDA(cudaMalloc((void **)&y, nrhs * m * sizeof(PetscScalar)));
    } else {
      /* X's array should serve as the array */
      ldy = ldx;
      PetscCall(MatDenseCUDAGetArrayWrite(X, &y));
    }
    PetscCallCUDA(cudaMemcpy2D(y, ldy * sizeof(PetscScalar), b, ldb * sizeof(PetscScalar), m * sizeof(PetscScalar), nrhs, bishost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
    if (bishost) {
      PetscCall(MatDenseRestoreArrayRead(B, &b));
    } else {
      PetscCall(MatDenseCUDARestoreArrayRead(B, &b));
    }
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSECUDA, &aiscuda));
  if (!aiscuda) PetscCall(MatConvert(A, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &A));
  PetscCall((*matsolve)(A, y, ldy, m, nrhs, k, transpose));
  if (!aiscuda) PetscCall(MatConvert(A, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &A));
  if (ldx < m || !xiscuda) {
    PetscScalar *x;

    /* The logic here is that the data is not yet in either X's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (xiscuda) {
      PetscCall(MatDenseCUDAGetArrayWrite(X, &x));
    } else {
      PetscCall(MatDenseGetArray(X, &x));
    }
    PetscCallCUDA(cudaMemcpy2D(x, ldx * sizeof(PetscScalar), y, ldy * sizeof(PetscScalar), k * sizeof(PetscScalar), nrhs, xiscuda ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));
    if (xiscuda) {
      PetscCall(MatDenseCUDARestoreArrayWrite(X, &x));
    } else {
      PetscCall(MatDenseRestoreArray(X, &x));
    }
    PetscCallCUDA(cudaFree(y));
  } else {
    PetscCall(MatDenseCUDARestoreArrayWrite(X, &y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_LU(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA  *dA  = (Mat_SeqDenseCUDA *)A->spptr;
  const PetscScalar *da;
  PetscCuBLASInt     lda;
  cusolverDnHandle_t handle;
  int                info;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArrayRead(A, &da));
  PetscCall(PetscCuBLASIntCast(mat->lda, &lda));
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "LU solve %d x %d on backend\n", m, k));
  PetscCallCUSOLVER(cusolverDnXgetrs(handle, T ? CUBLAS_OP_T : CUBLAS_OP_N, m, nrhs, da, lda, dA->d_fact_ipiv, x, ldx, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArrayRead(A, &da));
  if (PetscDefined(USE_DEBUG)) {
    PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
  }
  PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_Cholesky(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA  *dA  = (Mat_SeqDenseCUDA *)A->spptr;
  const PetscScalar *da;
  PetscCuBLASInt     lda;
  cusolverDnHandle_t handle;
  int                info;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArrayRead(A, &da));
  PetscCall(PetscCuBLASIntCast(mat->lda, &lda));
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "Cholesky solve %d x %d on backend\n", m, k));
  if (!dA->d_fact_ipiv) { /* spd */
    /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
    PetscCallCUSOLVER(cusolverDnXpotrs(handle, CUBLAS_FILL_MODE_LOWER, m, nrhs, da, lda, x, ldx, dA->d_fact_info));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnsytrs not implemented");
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArrayRead(A, &da));
  if (PetscDefined(USE_DEBUG)) {
    PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
  }
  PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_QR(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA  *dA  = (Mat_SeqDenseCUDA *)A->spptr;
  const PetscScalar *da;
  PetscCuBLASInt     lda, rank;
  cusolverDnHandle_t handle;
  cublasHandle_t     bhandle;
  int                info;
  cublasOperation_t  trans;
  PetscScalar        one = 1.;

  PetscFunctionBegin;
  PetscCall(PetscCuBLASIntCast(mat->rank, &rank));
  PetscCall(MatDenseCUDAGetArrayRead(A, &da));
  PetscCall(PetscCuBLASIntCast(mat->lda, &lda));
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(PetscCUBLASGetHandle(&bhandle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscInfo(A, "QR solve %d x %d on backend\n", m, k));
  if (!T) {
    if (PetscDefined(USE_COMPLEX)) {
      trans = CUBLAS_OP_C;
    } else {
      trans = CUBLAS_OP_T;
    }
    PetscCallCUSOLVER(cusolverDnXormqr(handle, CUBLAS_SIDE_LEFT, trans, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    if (PetscDefined(USE_DEBUG)) {
      PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
    }
    PetscCallCUBLAS(cublasXtrsm(bhandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
  } else {
    PetscCallCUBLAS(cublasXtrsm(bhandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
    PetscCallCUSOLVER(cusolverDnXormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    if (PetscDefined(USE_DEBUG)) {
      PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArrayRead(A, &da));
  PetscCall(PetscLogFlops(nrhs * (4.0 * m * mat->rank - PetscSqr(mat->rank))));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_LU(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_LU(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_QR(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_QR(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_LU(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_LU(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_Cholesky(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_Cholesky(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_QR(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_QR(Mat A, Mat B, Mat X)
{
  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseCUDA(Mat A, IS rperm, IS cperm, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar      *da;
  PetscCuBLASInt    m, n, lda;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(MatDenseCUDAGetArray(A, &da));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(a->lda, &lda));
  PetscCall(PetscInfo(A, "LU factor %d x %d on backend\n", m, n));
  if (!dA->d_fact_ipiv) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_ipiv, n * sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    PetscCallCUSOLVER(cusolverDnXgetrf_bufferSize(handle, m, n, da, lda, &dA->fact_lwork));
    PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUSOLVER(cusolverDnXgetrf(handle, m, n, da, lda, dA->d_fact_work, dA->d_fact_ipiv, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
  PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
#endif
  A->factortype = MAT_FACTOR_LU;
  PetscCall(PetscLogGpuFlops(2.0 * n * n * m / 3.0));

  A->ops->solve             = MatSolve_SeqDenseCUDA_LU;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_LU;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_LU;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_LU;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDA, &A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseCUDA(Mat A, IS perm, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar      *da;
  PetscCuBLASInt    n, lda;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &n));
  PetscCall(PetscInfo(A, "Cholesky factor %d x %d on backend\n", n, n));
  if (A->spd == PETSC_BOOL3_TRUE) {
    PetscCall(MatDenseCUDAGetArray(A, &da));
    PetscCall(PetscCuBLASIntCast(a->lda, &lda));
    if (!dA->fact_lwork) {
      PetscCallCUSOLVER(cusolverDnXpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, da, lda, &dA->fact_lwork));
      PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
    }
    if (!dA->d_fact_info) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUSOLVER(cusolverDnXpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, da, lda, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());

    PetscCall(MatDenseCUDARestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
    PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "cusolverDnsytrs unavailable. Use MAT_FACTOR_LU");
#if 0
    /* at the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs and *hetr* routines
       The code below should work, and it can be activated when *sytrs routines will be available */
    if (!dA->d_fact_ipiv) {
      PetscCallCUDA(cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv)));
    }
    if (!dA->fact_lwork) {
      PetscCallCUSOLVER(cusolverDnXsytrf_bufferSize(handle,n,da,lda,&dA->fact_lwork));
      PetscCallCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
    }
    if (!dA->d_fact_info) {
      PetscCallCUDA(cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info)));
    }
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUSOLVER(cusolverDnXsytrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_ipiv,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());
#endif

  A->ops->solve             = MatSolve_SeqDenseCUDA_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_Cholesky;
  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDA, &A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactor_SeqDenseCUDA(Mat A, IS col, const MatFactorInfo *factinfo)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar      *da;
  PetscCuBLASInt    m, min, max, n, lda;
#if defined(PETSC_USE_DEBUG)
  int info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscCUSOLVERDnGetHandle(&handle));
  PetscCall(MatDenseCUDAGetArray(A, &da));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(a->lda, &lda));
  PetscCall(PetscInfo(A, "QR factor %d x %d on backend\n", m, n));
  max = PetscMax(m, n);
  min = PetscMin(m, n);
  if (!dA->d_fact_tau) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_tau, min * sizeof(*dA->d_fact_tau)));
  if (!dA->d_fact_ipiv) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_ipiv, n * sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    PetscCallCUSOLVER(cusolverDnXgeqrf_bufferSize(handle, m, n, da, lda, &dA->fact_lwork));
    PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_work, dA->fact_lwork * sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) PetscCallCUDA(cudaMalloc((void **)&dA->d_fact_info, sizeof(*dA->d_fact_info)));
  if (!dA->workvec) PetscCall(VecCreateSeqCUDA(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUSOLVER(cusolverDnXgeqrf(handle, m, n, da, lda, dA->d_fact_tau, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatDenseCUDARestoreArray(A, &da));
#if defined(PETSC_USE_DEBUG)
  PetscCallCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cuSolver %d", -info);
#endif
  A->factortype = MAT_FACTOR_QR;
  a->rank       = min;
  PetscCall(PetscLogGpuFlops(2.0 * min * min * (max - min / 3.0)));

  A->ops->solve             = MatSolve_SeqDenseCUDA_QR;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_QR;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_QR;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_QR;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDA, &A->solvertype));
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(Mat A, Mat B, Mat C, PetscBool tA, PetscBool tB)
{
  const PetscScalar *da, *db;
  PetscScalar       *dc;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscCuBLASInt     m, n, k;
  PetscInt           alda, blda, clda;
  cublasHandle_t     cublasv2handle;
  PetscBool          Aiscuda, Biscuda;
  cublasStatus_t     berr;

  PetscFunctionBegin;
  /* we may end up with SEQDENSE as one of the arguments */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSECUDA, &Aiscuda));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &Biscuda));
  if (!Aiscuda) PetscCall(MatConvert(A, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &A));
  if (!Biscuda) PetscCall(MatConvert(B, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &B));
  PetscCall(PetscCuBLASIntCast(C->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(C->cmap->n, &n));
  if (tA) PetscCall(PetscCuBLASIntCast(A->rmap->n, &k));
  else PetscCall(PetscCuBLASIntCast(A->cmap->n, &k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  PetscCall(PetscInfo(C, "Matrix-Matrix product %d x %d x %d on backend\n", m, k, n));
  PetscCall(MatDenseCUDAGetArrayRead(A, &da));
  PetscCall(MatDenseCUDAGetArrayRead(B, &db));
  PetscCall(MatDenseCUDAGetArrayWrite(C, &dc));
  PetscCall(MatDenseGetLDA(A, &alda));
  PetscCall(MatDenseGetLDA(B, &blda));
  PetscCall(MatDenseGetLDA(C, &clda));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscLogGpuTimeBegin());
  berr = cublasXgemm(cublasv2handle, tA ? CUBLAS_OP_T : CUBLAS_OP_N, tB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one, da, alda, db, blda, &zero, dc, clda);
  PetscCallCUBLAS(berr);
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(1.0 * m * n * k + 1.0 * m * n * (k - 1)));
  PetscCall(MatDenseCUDARestoreArrayRead(A, &da));
  PetscCall(MatDenseCUDARestoreArrayRead(B, &db));
  PetscCall(MatDenseCUDARestoreArrayWrite(C, &dc));
  if (!Aiscuda) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (!Biscuda) PetscCall(MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A, B, C, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A, B, C, PETSC_FALSE, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A, B, C, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqDenseCUDA(Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatProductSetFromOptions_SeqDense(C));
  PetscFunctionReturn(0);
}

/* zz = op(A)*xx + yy
   if yy == NULL, only MatMult */
static PetscErrorCode MatMultAdd_SeqDenseCUDA_Private(Mat A, Vec xx, Vec yy, Vec zz, PetscBool trans)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  const PetscScalar *xarray, *da;
  PetscScalar       *zarray;
  PetscScalar        one = 1.0, zero = 0.0;
  PetscCuBLASInt     m, n, lda;
  cublasHandle_t     cublasv2handle;
  cublasStatus_t     berr;

  PetscFunctionBegin;
  /* mult add */
  if (yy && yy != zz) PetscCall(VecCopy_SeqCUDA(yy, zz));
  if (!A->rmap->n || !A->cmap->n) {
    /* mult only */
    if (!yy) PetscCall(VecSet_SeqCUDA(zz, 0.0));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscInfo(A, "Matrix-vector product %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", A->rmap->n, A->cmap->n));
  PetscCall(PetscCuBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(A->cmap->n, &n));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(MatDenseCUDAGetArrayRead(A, &da));
  PetscCall(PetscCuBLASIntCast(mat->lda, &lda));
  PetscCall(VecCUDAGetArrayRead(xx, &xarray));
  PetscCall(VecCUDAGetArray(zz, &zarray));
  PetscCall(PetscLogGpuTimeBegin());
  berr = cublasXgemv(cublasv2handle, trans ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, &one, da, lda, xarray, 1, (yy ? &one : &zero), zarray, 1);
  PetscCallCUBLAS(berr);
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * A->rmap->n * A->cmap->n - (yy ? 0 : A->rmap->n)));
  PetscCall(VecCUDARestoreArrayRead(xx, &xarray));
  PetscCall(VecCUDARestoreArray(zz, &zarray));
  PetscCall(MatDenseCUDARestoreArrayRead(A, &da));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseCUDA(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseCUDA_Private(A, xx, yy, zz, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseCUDA(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseCUDA_Private(A, xx, yy, zz, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseCUDA(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseCUDA_Private(A, xx, NULL, yy, PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseCUDA(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseCUDA_Private(A, xx, NULL, yy, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_SeqDenseCUDA(Mat A, const PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseCUDACopyFromGPU(A));
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
  if (!mat->v) PetscCall(MatSeqDenseSetPreallocation(A, NULL));
  *array         = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDenseCUDA(Mat A, PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseCUDACopyFromGPU(A));
  *array         = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDenseCUDA(Mat Y, PetscScalar alpha)
{
  Mat_SeqDense  *y = (Mat_SeqDense *)Y->data;
  PetscScalar   *dy;
  PetscCuBLASInt j, N, m, lday, one = 1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(MatDenseCUDAGetArray(Y, &dy));
  PetscCall(PetscCuBLASIntCast(Y->rmap->n * Y->cmap->n, &N));
  PetscCall(PetscCuBLASIntCast(Y->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(y->lda, &lday));
  PetscCall(PetscInfo(Y, "Performing Scale %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", Y->rmap->n, Y->cmap->n));
  PetscCall(PetscLogGpuTimeBegin());
  if (lday > m) {
    for (j = 0; j < Y->cmap->n; j++) PetscCallCUBLAS(cublasXscal(cublasv2handle, m, &alpha, dy + lday * j, one));
  } else PetscCallCUBLAS(cublasXscal(cublasv2handle, N, &alpha, dy, one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(N));
  PetscCall(MatDenseCUDARestoreArray(Y, &dy));
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

PetscErrorCode MatShift_DenseCUDA_Private(PetscScalar *da, PetscScalar alpha, PetscInt lda, PetscInt rstart, PetscInt rend, PetscInt cols)
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

PetscErrorCode MatShift_SeqDenseCUDA(Mat A, PetscScalar alpha)
{
  PetscScalar *da;
  PetscInt     m = A->rmap->n, n = A->cmap->n, lda;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArray(A, &da));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(PetscInfo(A, "Performing Shift %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(MatShift_DenseCUDA_Private(da, alpha, lda, 0, m, n));
  PetscCall(MatDenseCUDARestoreArray(A, &da));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseCUDA(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense *)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense *)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  PetscCuBLASInt     j, N, m, ldax, lday, one = 1;
  cublasHandle_t     cublasv2handle;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(MatDenseCUDAGetArrayRead(X, &dx));
  if (alpha == 0.0) PetscCall(MatDenseCUDAGetArrayWrite(Y, &dy));
  else PetscCall(MatDenseCUDAGetArray(Y, &dy));
  PetscCall(PetscCuBLASIntCast(X->rmap->n * X->cmap->n, &N));
  PetscCall(PetscCuBLASIntCast(X->rmap->n, &m));
  PetscCall(PetscCuBLASIntCast(x->lda, &ldax));
  PetscCall(PetscCuBLASIntCast(y->lda, &lday));
  PetscCall(PetscInfo(Y, "Performing AXPY %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", Y->rmap->n, Y->cmap->n));
  PetscCall(PetscLogGpuTimeBegin());
  if (ldax > m || lday > m) {
    for (j = 0; j < X->cmap->n; j++) PetscCallCUBLAS(cublasXaxpy(cublasv2handle, m, &alpha, dx + j * ldax, one, dy + j * lday, one));
  } else PetscCallCUBLAS(cublasXaxpy(cublasv2handle, N, &alpha, dx, one, dy, one));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(PetscMax(2. * N - 1, 0)));
  PetscCall(MatDenseCUDARestoreArrayRead(X, &dx));
  if (alpha == 0.0) PetscCall(MatDenseCUDARestoreArrayWrite(Y, &dy));
  else PetscCall(MatDenseCUDARestoreArray(Y, &dy));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseCUDA(Mat A)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  if (dA) {
    PetscCheck(!dA->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDenseCUDAResetArray() must be called first");
    if (!dA->user_alloc) PetscCallCUDA(cudaFree(dA->d_v));
    PetscCallCUDA(cudaFree(dA->d_fact_tau));
    PetscCallCUDA(cudaFree(dA->d_fact_ipiv));
    PetscCallCUDA(cudaFree(dA->d_fact_info));
    PetscCallCUDA(cudaFree(dA->d_fact_work));
    PetscCall(VecDestroy(&dA->workvec));
  }
  PetscCall(PetscFree(A->spptr));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseCUDA(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(MatConvert_SeqDenseCUDA_SeqDense(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseCUDA(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  MatDuplicateOption hcpvalues = (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) ? MAT_DO_NOT_COPY_VALUES : cpvalues;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  PetscCall(MatSetType(*B, ((PetscObject)A)->type_name));
  PetscCall(MatDuplicateNoCreate_SeqDense(*B, A, hcpvalues));
  if (cpvalues == MAT_COPY_VALUES && hcpvalues != MAT_COPY_VALUES) PetscCall(MatCopy_SeqDenseCUDA(A, *B, SAME_NONZERO_PATTERN));
  if (cpvalues != MAT_COPY_VALUES) { /* allocate memory if needed */
    Mat_SeqDenseCUDA *dB = (Mat_SeqDenseCUDA *)(*B)->spptr;
    if (!dB->d_v) PetscCall(MatSeqDenseCUDASetPreallocation(*B, NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_SeqDenseCUDA(Mat A, Vec v, PetscInt col)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscScalar      *x;
  PetscBool         viscuda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)v, &viscuda, VECSEQCUDA, VECMPICUDA, VECCUDA, ""));
  if (viscuda && !v->boundtocpu) { /* update device data */
    PetscCall(VecCUDAGetArrayWrite(v, &x));
    if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      PetscCallCUDA(cudaMemcpy(x, dA->d_v + col * a->lda, A->rmap->n * sizeof(PetscScalar), cudaMemcpyHostToHost));
    } else {
      PetscCallCUDA(cudaMemcpy(x, a->v + col * a->lda, A->rmap->n * sizeof(PetscScalar), cudaMemcpyHostToDevice));
    }
    PetscCall(VecCUDARestoreArrayWrite(v, &x));
  } else { /* update host data */
    PetscCall(VecGetArrayWrite(v, &x));
    if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask & PETSC_OFFLOAD_CPU) {
      PetscCall(PetscArraycpy(x, a->v + col * a->lda, A->rmap->n));
    } else if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      PetscCallCUDA(cudaMemcpy(x, dA->d_v + col * a->lda, A->rmap->n * sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    }
    PetscCall(VecRestoreArrayWrite(v, &x));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat A, MatFactorType ftype, Mat *fact)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), fact));
  PetscCall(MatSetSizes(*fact, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  PetscCall(MatSetType(*fact, MATSEQDENSECUDA));
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
  PetscCall(PetscStrallocpy(MATSOLVERCUDA, &(*fact)->solvertype));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_LU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_ILU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&(*fact)->preferredordering[MAT_FACTOR_ICC]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAGetArray(A, (PetscScalar **)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    PetscCall(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecCUDAPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecCUDAResetArray(a->cvec));
  PetscCall(MatDenseCUDARestoreArray(A, (PetscScalar **)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAGetArrayRead(A, &a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    PetscCall(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecCUDAPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  PetscCall(VecLockReadPush(a->cvec));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecCUDAResetArray(a->cvec));
  PetscCall(MatDenseCUDARestoreArrayRead(A, &a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAGetArrayWrite(A, (PetscScalar **)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    PetscCall(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A), A->rmap->bs, A->rmap->n, a->ptrinuse, &a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(VecCUDAPlaceArray(a->cvec, a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDenseCUDA(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(VecCUDAResetArray(a->cvec));
  PetscCall(MatDenseCUDARestoreArrayWrite(A, (PetscScalar **)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseCUDA(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *v)
{
  Mat_SeqDense     *a  = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && (cend - cbegin != a->cmat->cmap->N || rend - rbegin != a->cmat->rmap->N)) PetscCall(MatDestroy(&a->cmat));
  PetscCall(MatSeqDenseCUDACopyToGPU(A));
  if (!a->cmat) {
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)A), rend - rbegin, PETSC_DECIDE, rend - rbegin, cend - cbegin, dA->d_v + rbegin + (size_t)cbegin * a->lda, &a->cmat));
  } else {
    PetscCall(MatDenseCUDAPlaceArray(a->cmat, dA->d_v + rbegin + (size_t)cbegin * a->lda));
  }
  PetscCall(MatDenseSetLDA(a->cmat, a->lda));
  /* Place CPU array if present but not copy any data */
  a->cmat->offloadmask = PETSC_OFFLOAD_GPU;
  if (a->v) PetscCall(MatDensePlaceArray(a->cmat, a->v + rbegin + (size_t)cbegin * a->lda));
  a->cmat->offloadmask = A->offloadmask;
  a->matinuse          = cbegin + 1;
  *v                   = a->cmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseCUDA(Mat A, Mat *v)
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
  PetscCall(MatDenseCUDAResetArray(a->cmat));
  if (reset) PetscCall(MatDenseResetArray(a->cmat));
  if (copy) {
    PetscCall(MatSeqDenseCUDACopyFromGPU(A));
  } else A->offloadmask = (suboff == PETSC_OFFLOAD_CPU) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  a->cmat->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseSetLDA_SeqDenseCUDA(Mat A, PetscInt lda)
{
  Mat_SeqDense     *cA = (Mat_SeqDense *)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA *)A->spptr;
  PetscBool         data;

  PetscFunctionBegin;
  data = (PetscBool)((A->rmap->n > 0 && A->cmap->n > 0) ? (dA->d_v ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE);
  PetscCheck(dA->user_alloc || !data || cA->lda == lda, PETSC_COMM_SELF, PETSC_ERR_ORDER, "LDA cannot be changed after allocation of internal storage");
  PetscCheck(lda >= A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "LDA %" PetscInt_FMT " must be at least matrix dimension %" PetscInt_FMT, lda, A->rmap->n);
  cA->lda = lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_SeqDenseCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) PetscCall(MatSeqDenseCUDASetPreallocation(A, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetRandom_SeqDenseCUDA(Mat A, PetscRandom r)
{
  PetscBool iscurand;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)r, PETSCCURAND, &iscurand));
  if (iscurand) {
    PetscScalar *a;
    PetscInt     lda, m, n, mn = 0;

    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatDenseGetLDA(A, &lda));
    PetscCall(MatDenseCUDAGetArrayWrite(A, &a));
    if (lda > m) {
      for (PetscInt i = 0; i < n; i++) PetscCall(PetscRandomGetValues(r, m, a + i * lda));
    } else {
      PetscCall(PetscIntMultError(m, n, &mn));
      PetscCall(PetscRandomGetValues(r, mn, a));
    }
    PetscCall(MatDenseCUDARestoreArrayWrite(A, &a));
  } else PetscCall(MatSetRandom_SeqDense(A, r));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqDenseCUDA(Mat A, PetscBool flg)
{
  Mat_SeqDense *a = (Mat_SeqDense *)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = flg;
  if (!flg) {
    PetscBool iscuda;

    PetscCall(PetscFree(A->defaultrandtype));
    PetscCall(PetscStrallocpy(PETSCCURAND, &A->defaultrandtype));
    PetscCall(PetscObjectTypeCompare((PetscObject)a->cvec, VECSEQCUDA, &iscuda));
    if (!iscuda) PetscCall(VecDestroy(&a->cvec));
    PetscCall(PetscObjectTypeCompare((PetscObject)a->cmat, MATSEQDENSECUDA, &iscuda));
    if (!iscuda) PetscCall(MatDestroy(&a->cmat));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArray_C", MatDenseGetArray_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayRead_C", MatDenseGetArrayRead_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWrite_C", MatDenseGetArrayWrite_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatQRFactor_C", MatQRFactor_SeqDenseCUDA));

    A->ops->duplicate               = MatDuplicate_SeqDenseCUDA;
    A->ops->mult                    = MatMult_SeqDenseCUDA;
    A->ops->multadd                 = MatMultAdd_SeqDenseCUDA;
    A->ops->multtranspose           = MatMultTranspose_SeqDenseCUDA;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDenseCUDA;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->axpy                    = MatAXPY_SeqDenseCUDA;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDenseCUDA;
    A->ops->lufactor                = MatLUFactor_SeqDenseCUDA;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDenseCUDA;
    A->ops->getcolumnvector         = MatGetColumnVector_SeqDenseCUDA;
    A->ops->scale                   = MatScale_SeqDenseCUDA;
    A->ops->shift                   = MatShift_SeqDenseCUDA;
    A->ops->copy                    = MatCopy_SeqDenseCUDA;
    A->ops->zeroentries             = MatZeroEntries_SeqDenseCUDA;
    A->ops->setup                   = MatSetUp_SeqDenseCUDA;
    A->ops->setrandom               = MatSetRandom_SeqDenseCUDA;
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    PetscCall(MatSeqDenseCUDACopyFromGPU(A));
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
    A->ops->setup                   = MatSetUp_SeqDense;
    A->ops->setrandom               = MatSetRandom_SeqDense;
  }
  if (a->cmat) PetscCall(MatBindToCPU(a->cmat, flg));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDenseCUDA_SeqDense(Mat M, MatType type, MatReuse reuse, Mat *newmat)
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
  PetscCall(MatBindToCPU_SeqDenseCUDA(B, PETSC_TRUE));
  PetscCall(MatReset_SeqDenseCUDA(B));
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECSTANDARD, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensecuda_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArrayRead_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArrayWrite_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArrayRead_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArrayWrite_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAPlaceArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAResetArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAReplaceArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensecuda_C", NULL));
  a = (Mat_SeqDense *)B->data;
  PetscCall(VecDestroy(&a->cvec)); /* cvec might be VECSEQCUDA. Destroy it and rebuild a VECSEQ when needed */
  B->ops->bindtocpu = NULL;
  B->ops->destroy   = MatDestroy_SeqDense;
  B->offloadmask    = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat M, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat_SeqDenseCUDA *dB;
  Mat               B;
  Mat_SeqDense     *a;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
    PetscFunctionReturn(0);
  }

  B = *newmat;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSECUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensecuda_seqdense_C", MatConvert_SeqDenseCUDA_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArray_C", MatDenseCUDAGetArray_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArrayRead_C", MatDenseCUDAGetArrayRead_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAGetArrayWrite_C", MatDenseCUDAGetArrayWrite_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArray_C", MatDenseCUDARestoreArray_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArrayRead_C", MatDenseCUDARestoreArrayRead_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDARestoreArrayWrite_C", MatDenseCUDARestoreArrayWrite_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAPlaceArray_C", MatDenseCUDAPlaceArray_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAResetArray_C", MatDenseCUDAResetArray_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseCUDAReplaceArray_C", MatDenseCUDAReplaceArray_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensecuda_C", MatProductSetFromOptions_SeqAIJ_SeqDense));
  a = (Mat_SeqDense *)B->data;
  PetscCall(VecDestroy(&a->cvec)); /* cvec might be VECSEQ. Destroy it and rebuild a VECSEQCUDA when needed */
  PetscCall(PetscNew(&dB));

  B->spptr       = dB;
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  PetscCall(MatBindToCPU_SeqDenseCUDA(B, PETSC_FALSE));
  B->ops->bindtocpu = MatBindToCPU_SeqDenseCUDA;
  B->ops->destroy   = MatDestroy_SeqDenseCUDA;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqDenseCUDA - Creates a sequential matrix in dense format using CUDA.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of rows
.  n - number of columns
-  data - optional location of GPU matrix data.  Set data=NULL for PETSc
   to control matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Level: intermediate

.seealso: `MATSEQDENSE`, `MatCreate()`, `MatCreateSeqDense()`
@*/
PetscErrorCode MatCreateSeqDenseCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, comm, PETSC_ERR_ARG_WRONG, "Invalid communicator size %d", size);
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQDENSECUDA));
  PetscCall(MatSeqDenseCUDASetPreallocation(*A, data));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSECUDA - MATSEQDENSECUDA = "seqdensecuda" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensecuda - sets the matrix type to `MATSEQDENSECUDA` during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MATSEQDENSE`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseCUDA(Mat B)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_SeqDense(B));
  PetscCall(MatConvert_SeqDense_SeqDenseCUDA(B, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(0);
}
