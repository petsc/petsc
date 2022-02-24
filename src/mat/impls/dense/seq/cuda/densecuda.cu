/*
     Defines the matrix operations for sequential dense with CUDA
*/
#include <petscpkg_version.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsc/private/cudavecimpl.h> /* cublas definitions are here */

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnCpotrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnCpotrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnCpotrs((a),(b),(c),(d),(cuComplex*)(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnCpotri((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnCpotri_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnCsytrf((a),(b),(c),(cuComplex*)(d),(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnCsytrf_bufferSize((a),(b),(cuComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnCgetrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnCgetrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnCgetrs((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(cuComplex*)(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnCgeqrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnCgeqrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnCunmqr_bufferSize((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(cuComplex*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnCunmqr((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(cuComplex*)(j),(k),(cuComplex*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasCtrsm((a),(b),(c),(d),(e),(f),(g),(cuComplex*)(h),(cuComplex*)(i),(j),(cuComplex*)(k),(l))
#else /* complex double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnZpotrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnZpotrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnZpotrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnZpotri((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnZpotri_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnZsytrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnZsytrf_bufferSize((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnZgetrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnZgetrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnZgetrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnZgeqrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnZgeqrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnZunmqr_bufferSize((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(cuDoubleComplex*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnZunmqr((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(cuDoubleComplex*)(j),(k),(cuDoubleComplex*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasZtrsm((a),(b),(c),(d),(e),(f),(g),(cuDoubleComplex*)(h),(cuDoubleComplex*)(i),(j),(cuDoubleComplex*)(k),(l))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnSpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnSpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnSpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnSpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnSpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnSsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnSsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnSgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnSgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnSgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnSgeqrf_bufferSize((a),(b),(c),(float*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnSgeqrf((a),(b),(c),(float*)(d),(e),(float*)(f),(float*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnSormqr_bufferSize((a),(b),(c),(d),(e),(f),(float*)(g),(h),(float*)(i),(float*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnSormqr((a),(b),(c),(d),(e),(f),(float*)(g),(h),(float*)(i),(float*)(j),(k),(float*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasStrsm((a),(b),(c),(d),(e),(f),(g),(float*)(h),(float*)(i),(j),(float*)(k),(l))
#else /* real double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnDpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnDpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnDpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXpotri(a,b,c,d,e,f,g,h)        cusolverDnDpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotri_bufferSize(a,b,c,d,e,f) cusolverDnDpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnDsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnDsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnDgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnDgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnDgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define cusolverDnXgeqrf_bufferSize(a,b,c,d,e,f) cusolverDnDgeqrf_bufferSize((a),(b),(c),(double*)(d),(e),(f))
#define cusolverDnXgeqrf(a,b,c,d,e,f,g,h,i)      cusolverDnDgeqrf((a),(b),(c),(double*)(d),(e),(double*)(f),(double*)(g),(h),(i))
#define cusolverDnXormqr_bufferSize(a,b,c,d,e,f,g,h,i,j,k,l) cusolverDnDormqr_bufferSize((a),(b),(c),(d),(e),(f),(double*)(g),(h),(double*)(i),(double*)(j),(k),(l))
#define cusolverDnXormqr(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cusolverDnDormqr((a),(b),(c),(d),(e),(f),(double*)(g),(h),(double*)(i),(double*)(j),(k),(double*)(l),(m),(n))
#define cublasXtrsm(a,b,c,d,e,f,g,h,i,j,k,l)     cublasDtrsm((a),(b),(c),(d),(e),(f),(g),(double*)(h),(double*)(i),(j),(double*)(k),(l))
#endif
#endif

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool   user_alloc;
  PetscScalar *unplacedarray; /* if one called MatCUDADensePlaceArray(), this is where it stashed the original */
  PetscBool   unplaced_user_alloc;
  /* factorization support */
  PetscCuBLASInt *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_tau;  /* device QR tau vector */
  PetscScalar *d_fact_work; /* device workspace */
  PetscCuBLASInt fact_lwork;
  PetscCuBLASInt *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseCUDA;

PetscErrorCode MatSeqDenseCUDASetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        iscuda;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&iscuda));
  if (!iscuda) PetscFunctionReturn(0);
  CHKERRQ(PetscLayoutSetUp(A->rmap));
  CHKERRQ(PetscLayoutSetUp(A->cmap));
  /* it may happen CPU preallocation has not been performed */
  if (cA->lda <= 0) cA->lda = A->rmap->n;
  if (!dA->user_alloc) CHKERRCUDA(cudaFree(dA->d_v));
  if (!d_data) { /* petsc-allocated storage */
    size_t sz;

    CHKERRQ(PetscIntMultError(cA->lda,A->cmap->n,NULL));
    sz   = cA->lda*A->cmap->n*sizeof(PetscScalar);
    CHKERRCUDA(cudaMalloc((void**)&dA->d_v,sz));
    CHKERRCUDA(cudaMemset(dA->d_v,0,sz));
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
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  CHKERRQ(PetscInfo(A,"%s matrix %d x %d\n",A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n));
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!cA->v) { /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
      CHKERRQ(MatSeqDenseSetPreallocation(A,NULL));
    }
    CHKERRQ(PetscLogEventBegin(MAT_DenseCopyFromGPU,A,0,0,0));
    if (cA->lda > A->rmap->n) {
      CHKERRCUDA(cudaMemcpy2D(cA->v,cA->lda*sizeof(PetscScalar),dA->d_v,cA->lda*sizeof(PetscScalar),A->rmap->n*sizeof(PetscScalar),A->cmap->n,cudaMemcpyDeviceToHost));
    } else {
      CHKERRCUDA(cudaMemcpy(cA->v,dA->d_v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyDeviceToHost));
    }
    CHKERRQ(PetscLogGpuToCpu(cA->lda*sizeof(PetscScalar)*A->cmap->n));
    CHKERRQ(PetscLogEventEnd(MAT_DenseCopyFromGPU,A,0,0,0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseCUDACopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        copy;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  CHKERRQ(PetscInfo(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n));
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      CHKERRQ(MatSeqDenseCUDASetPreallocation(A,NULL));
    }
    CHKERRQ(PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0));
    if (cA->lda > A->rmap->n) {
      CHKERRCUDA(cudaMemcpy2D(dA->d_v,cA->lda*sizeof(PetscScalar),cA->v,cA->lda*sizeof(PetscScalar),A->rmap->n*sizeof(PetscScalar),A->cmap->n,cudaMemcpyHostToDevice));
    } else {
      CHKERRCUDA(cudaMemcpy(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyHostToDevice));
    }
    CHKERRQ(PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n));
    CHKERRQ(PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0));

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_SeqDenseCUDA(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode    ierr;
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt          lda1,lda2,m=A->rmap->n,n=A->cmap->n;
  cudaError_t       cerr;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    CHKERRQ(MatCopy_Basic(A,B,str));
    PetscFunctionReturn(0);
  }
  PetscCheckFalse(m != B->rmap->n || n != B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&va));
  CHKERRQ(MatDenseCUDAGetArrayWrite(B,&vb));
  CHKERRQ(MatDenseGetLDA(A,&lda1));
  CHKERRQ(MatDenseGetLDA(B,&lda2));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (lda1>m || lda2>m) {
    CHKERRCUDA(cudaMemcpy2D(vb,lda2*sizeof(PetscScalar),va,lda1*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyDeviceToDevice));
  } else {
    CHKERRCUDA(cudaMemcpy(vb,va,m*(n*sizeof(PetscScalar)),cudaMemcpyDeviceToDevice));
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArrayWrite(B,&vb));
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqDenseCUDA(Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    *va;
  PetscInt       lda,m = A->rmap->n,n = A->cmap->n;
  cudaError_t    cerr;

  PetscFunctionBegin;
  CHKERRQ(MatDenseCUDAGetArrayWrite(A,&va));
  CHKERRQ(MatDenseGetLDA(A,&lda));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (lda>m) {
    CHKERRCUDA(cudaMemset2D(va,lda*sizeof(PetscScalar),0,m*sizeof(PetscScalar),n));
  } else {
    CHKERRCUDA(cudaMemset(va,0,m*(n*sizeof(PetscScalar))));
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArrayWrite(A,&va));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAPlaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  PetscCheckFalse(aa->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(aa->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCheckFalse(dA->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
  if (aa->v) CHKERRQ(MatSeqDenseCUDACopyToGPU(A));
  dA->unplacedarray = dA->d_v;
  dA->unplaced_user_alloc = dA->user_alloc;
  dA->d_v = (PetscScalar*)a;
  dA->user_alloc = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAResetArray_SeqDenseCUDA(Mat A)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->v) CHKERRQ(MatSeqDenseCUDACopyToGPU(A));
  dA->d_v = dA->unplacedarray;
  dA->user_alloc = dA->unplaced_user_alloc;
  dA->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAReplaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  PetscCheckFalse(aa->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(aa->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCheckFalse(dA->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
  if (!dA->user_alloc) CHKERRCUDA(cudaFree(dA->d_v));
  dA->d_v = (PetscScalar*)a;
  dA->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  if (!dA->d_v) {
    CHKERRQ(MatSeqDenseCUDASetPreallocation(A,NULL));
  }
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
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  CHKERRQ(MatSeqDenseCUDACopyToGPU(A));
  *a   = dA->d_v;
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
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  CHKERRQ(MatSeqDenseCUDACopyToGPU(A));
  *a   = dA->d_v;
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
#if PETSC_PKG_CUDA_VERSION_GE(10,1,0)
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  cusolverDnHandle_t handle;
  PetscCuBLASInt     n,lda;
#if defined(PETSC_USE_DEBUG)
  PetscCuBLASInt     info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&n));
  CHKERRQ(PetscCuBLASIntCast(a->lda,&lda));
  PetscCheckFalse(A->factortype == MAT_FACTOR_LU,PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDngetri not implemented");
  if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      PetscCuBLASInt il;

      CHKERRQ(MatDenseCUDAGetArray(A,&da));
      CHKERRCUSOLVER(cusolverDnXpotri_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&il));
      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        CHKERRCUDA(cudaFree(dA->d_fact_work));
        CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
      }
      CHKERRQ(PetscLogGpuTimeBegin());
      CHKERRCUSOLVER(cusolverDnXpotri(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info));
      CHKERRQ(PetscLogGpuTimeEnd());
      CHKERRQ(MatDenseCUDARestoreArray(A,&da));
      /* TODO (write cuda kernel) */
      CHKERRQ(MatSeqDenseSymmetrize_Private(A,PETSC_TRUE));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytri not implemented");
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Not implemented");
#if defined(PETSC_USE_DEBUG)
  CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: leading minor of order %d is zero",info);
  PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  CHKERRQ(PetscLogGpuFlops(1.0*n*n*n/3.0));
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  CHKERRQ(PetscFree(A->solvertype));
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Upgrade to CUDA version 10.1.0 or higher");
#endif
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal(Mat A, Vec xx, Vec yy, PetscBool transpose,
                                                     PetscErrorCode (*matsolve)(Mat,PetscScalar*,PetscCuBLASInt,PetscCuBLASInt,PetscCuBLASInt,PetscCuBLASInt,PetscBool))
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar      *y;
  PetscCuBLASInt   m=0, k=0;
  PetscBool        xiscuda, yiscuda, aiscuda;

  PetscFunctionBegin;
  PetscCheckFalse(A->factortype == MAT_FACTOR_NONE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&k));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)xx,VECSEQCUDA,&xiscuda));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)yy,VECSEQCUDA,&yiscuda));
  {
    const PetscScalar *x;
    PetscBool xishost = PETSC_TRUE;

    /* The logic here is to try to minimize the amount of memory copying:
       if we call VecCUDAGetArrayRead(X,&x) every time xiscuda and the
       data is not offloaded to the GPU yet, then the data is copied to the
       GPU.  But we are only trying to get the data in order to copy it into the y
       array.  So the array x will be wherever the data already is so that
       only one memcpy is performed */
    if (xiscuda && xx->offloadmask & PETSC_OFFLOAD_GPU) {
      CHKERRQ(VecCUDAGetArrayRead(xx, &x));
      xishost =  PETSC_FALSE;
    } else {
      CHKERRQ(VecGetArrayRead(xx, &x));
    }
    if (k < m || !yiscuda) {
      if (!dA->workvec) {
        CHKERRQ(VecCreateSeqCUDA(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
      }
      CHKERRQ(VecCUDAGetArrayWrite(dA->workvec, &y));
    } else {
      CHKERRQ(VecCUDAGetArrayWrite(yy,&y));
    }
    CHKERRCUDA(cudaMemcpy(y,x,m*sizeof(PetscScalar),xishost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&aiscuda));
  if (!aiscuda) {
    CHKERRQ(MatConvert(A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&A));
  }
  CHKERRQ((*matsolve) (A, y, m, m, 1, k, transpose));
  if (!aiscuda) {
    CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A));
  }
  if (k < m || !yiscuda) {
    PetscScalar *yv;

    /* The logic here is that the data is not yet in either yy's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (yiscuda) {
      CHKERRQ(VecCUDAGetArrayWrite(yy,&yv));
    } else {
      CHKERRQ(VecGetArray(yy,&yv));
    }
    CHKERRCUDA(cudaMemcpy(yv,y,k*sizeof(PetscScalar),yiscuda ? cudaMemcpyDeviceToDevice: cudaMemcpyDeviceToHost));
    if (yiscuda) {
      CHKERRQ(VecCUDARestoreArrayWrite(yy,&yv));
    } else {
      CHKERRQ(VecRestoreArray(yy,&yv));
    }
    CHKERRQ(VecCUDARestoreArrayWrite(dA->workvec, &y));
  } else {
    CHKERRQ(VecCUDARestoreArrayWrite(yy,&y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_Internal(Mat A, Mat B, Mat X, PetscBool transpose,
                                                        PetscErrorCode (*matsolve)(Mat,PetscScalar*,PetscCuBLASInt,PetscCuBLASInt,PetscCuBLASInt,PetscCuBLASInt,PetscBool))
{
  PetscScalar       *y;
  PetscInt          n, _ldb, _ldx;
  PetscBool         biscuda, xiscuda, aiscuda;
  PetscCuBLASInt    nrhs=0,m=0,k=0,ldb=0,ldx=0,ldy=0;

  PetscFunctionBegin;
  PetscCheckFalse(A->factortype == MAT_FACTOR_NONE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&k));
  CHKERRQ(MatGetSize(B,NULL,&n));
  CHKERRQ(PetscCuBLASIntCast(n,&nrhs));
  CHKERRQ(MatDenseGetLDA(B,&_ldb));
  CHKERRQ(PetscCuBLASIntCast(_ldb, &ldb));
  CHKERRQ(MatDenseGetLDA(X,&_ldx));
  CHKERRQ(PetscCuBLASIntCast(_ldx, &ldx));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)B,MATSEQDENSECUDA,&biscuda));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)X,MATSEQDENSECUDA,&xiscuda));
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
      CHKERRQ(MatDenseCUDAGetArrayRead(B,&b));
      bishost = PETSC_FALSE;
    } else {
      CHKERRQ(MatDenseGetArrayRead(B,&b));
    }
    if (ldx < m || !xiscuda) {
      /* X's array cannot serve as the array (too small or not on device), B's
       * array cannot serve as the array (const), so allocate a new array  */
      ldy = m;
      CHKERRCUDA(cudaMalloc((void**)&y,nrhs*m*sizeof(PetscScalar)));
    } else {
      /* X's array should serve as the array */
      ldy = ldx;
      CHKERRQ(MatDenseCUDAGetArrayWrite(X,&y));
    }
    CHKERRCUDA(cudaMemcpy2D(y,ldy*sizeof(PetscScalar),b,ldb*sizeof(PetscScalar),m*sizeof(PetscScalar),nrhs,bishost ? cudaMemcpyHostToDevice: cudaMemcpyDeviceToDevice));
    if (bishost) {
      CHKERRQ(MatDenseRestoreArrayRead(B,&b));
    } else {
      CHKERRQ(MatDenseCUDARestoreArrayRead(B,&b));
    }
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&aiscuda));
  if (!aiscuda) {
    CHKERRQ(MatConvert(A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&A));
  }
  CHKERRQ((*matsolve) (A, y, ldy, m, nrhs, k, transpose));
  if (!aiscuda) {
    CHKERRQ(MatConvert(A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&A));
  }
  if (ldx < m || !xiscuda) {
    PetscScalar *x;

    /* The logic here is that the data is not yet in either X's GPU array or its
       CPU array.  There is nothing in the interface to say where the user would like
       it to end up.  So we choose the GPU, because it is the faster option */
    if (xiscuda) {
      CHKERRQ(MatDenseCUDAGetArrayWrite(X,&x));
    } else {
      CHKERRQ(MatDenseGetArray(X,&x));
    }
    CHKERRCUDA(cudaMemcpy2D(x,ldx*sizeof(PetscScalar),y,ldy*sizeof(PetscScalar),k*sizeof(PetscScalar),nrhs,xiscuda ? cudaMemcpyDeviceToDevice: cudaMemcpyDeviceToHost));
    if (xiscuda) {
      CHKERRQ(MatDenseCUDARestoreArrayWrite(X,&x));
    } else {
      CHKERRQ(MatDenseRestoreArray(X,&x));
    }
    CHKERRCUDA(cudaFree(y));
  } else {
    CHKERRQ(MatDenseCUDARestoreArrayWrite(X,&y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_LU(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense       *mat = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscCuBLASInt     lda;
  cusolverDnHandle_t handle;
  int                info;

  PetscFunctionBegin;
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&da));
  CHKERRQ(PetscCuBLASIntCast(mat->lda,&lda));
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRQ(PetscInfo(A,"LU solve %d x %d on backend\n",m,k));
  CHKERRCUSOLVER(cusolverDnXgetrs(handle,T ? CUBLAS_OP_T : CUBLAS_OP_N,m,nrhs,da,lda,dA->d_fact_ipiv,x,ldx,dA->d_fact_info));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&da));
  if (PetscDefined(USE_DEBUG)) {
    CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  }
  CHKERRQ(PetscLogGpuFlops(nrhs*(2.0*m*m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_Cholesky(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense       *mat = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscCuBLASInt     lda;
  cusolverDnHandle_t handle;
  int                info;

  PetscFunctionBegin;
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&da));
  CHKERRQ(PetscCuBLASIntCast(mat->lda,&lda));
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRQ(PetscInfo(A,"Cholesky solve %d x %d on backend\n",m,k));
  if (!dA->d_fact_ipiv) { /* spd */
    /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
    CHKERRCUSOLVER(cusolverDnXpotrs(handle,CUBLAS_FILL_MODE_LOWER,m,nrhs,da,lda,x,ldx,dA->d_fact_info));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytrs not implemented");
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&da));
  if (PetscDefined(USE_DEBUG)) {
    CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  }
  CHKERRQ(PetscLogGpuFlops(nrhs*(2.0*m*m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_QR(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense       *mat = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscCuBLASInt     lda, rank;
  cusolverDnHandle_t handle;
  cublasHandle_t     bhandle;
  cusolverStatus_t   csrr;
  cublasStatus_t     cbrr;
  int                info;
  cublasOperation_t  trans;
  PetscScalar        one = 1.;

  PetscFunctionBegin;
  CHKERRQ(PetscCuBLASIntCast(mat->rank,&rank));
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&da));
  CHKERRQ(PetscCuBLASIntCast(mat->lda,&lda));
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(PetscCUBLASGetHandle(&bhandle));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRQ(PetscInfo(A,"QR solve %d x %d on backend\n",m,k));
  if (!T) {
    if (PetscDefined(USE_COMPLEX)) {
      trans = CUBLAS_OP_C;
    } else {
      trans = CUBLAS_OP_T;
    }
    csrr = cusolverDnXormqr(handle, CUBLAS_SIDE_LEFT, trans, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info);CHKERRCUSOLVER(csrr);
    if (PetscDefined(USE_DEBUG)) {
      CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
      PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
    }
    cbrr = cublasXtrsm(bhandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx);CHKERRCUBLAS(cbrr);
  } else {
    cbrr = cublasXtrsm(bhandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx);CHKERRCUBLAS(cbrr);
    csrr = cusolverDnXormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, nrhs, rank, da, lda, dA->d_fact_tau, x, ldx, dA->d_fact_work, dA->fact_lwork, dA->d_fact_info);CHKERRCUSOLVER(csrr);
    if (PetscDefined(USE_DEBUG)) {
      CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
      PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
    }
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&da));
  CHKERRQ(PetscLogFlops(nrhs*(4.0*m*mat->rank - PetscSqr(mat->rank))));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_LU(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_LU(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Cholesky(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_Cholesky(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_QR(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA_QR(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatSolve_SeqDenseCUDA_Internal(A, xx, yy, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_LU(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_LU(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_LU));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_Cholesky(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_Cholesky(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_Cholesky));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA_QR(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_FALSE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDenseCUDA_QR(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatSolve_SeqDenseCUDA_Internal(A, B, X, PETSC_TRUE, MatSolve_SeqDenseCUDA_Internal_QR));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseCUDA(Mat A,IS rperm,IS cperm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscCuBLASInt     m,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(MatDenseCUDAGetArray(A,&da));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&n));
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(a->lda,&lda));
  CHKERRQ(PetscInfo(A,"LU factor %d x %d on backend\n",m,n));
  if (!dA->d_fact_ipiv) {
    CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv)));
  }
  if (!dA->fact_lwork) {
    CHKERRCUSOLVER(cusolverDnXgetrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork));
    CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) {
    CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info)));
  }
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUSOLVER(cusolverDnXgetrf(handle,m,n,da,lda,dA->d_fact_work,dA->d_fact_ipiv,dA->d_fact_info));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArray(A,&da));
#if defined(PETSC_USE_DEBUG)
  CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  A->factortype = MAT_FACTOR_LU;
  CHKERRQ(PetscLogGpuFlops(2.0*n*n*m/3.0));

  A->ops->solve             = MatSolve_SeqDenseCUDA_LU;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_LU;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_LU;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_LU;

  CHKERRQ(PetscFree(A->solvertype));
  CHKERRQ(PetscStrallocpy(MATSOLVERCUDA,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseCUDA(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscCuBLASInt     n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&n));
  CHKERRQ(PetscInfo(A,"Cholesky factor %d x %d on backend\n",n,n));
  if (A->spd) {
    CHKERRQ(MatDenseCUDAGetArray(A,&da));
    CHKERRQ(PetscCuBLASIntCast(a->lda,&lda));
    if (!dA->fact_lwork) {
      CHKERRCUSOLVER(cusolverDnXpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&dA->fact_lwork));
      CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
    }
    if (!dA->d_fact_info) {
      CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info)));
    }
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUSOLVER(cusolverDnXpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info));
    CHKERRQ(PetscLogGpuTimeEnd());

    CHKERRQ(MatDenseCUDARestoreArray(A,&da));
#if defined(PETSC_USE_DEBUG)
    CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    CHKERRQ(PetscLogGpuFlops(1.0*n*n*n/3.0));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cusolverDnsytrs unavailable. Use MAT_FACTOR_LU");
#if 0
    /* at the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs and *hetr* routines
       The code below should work, and it can be activated when *sytrs routines will be available */
    if (!dA->d_fact_ipiv) {
      CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv)));
    }
    if (!dA->fact_lwork) {
      CHKERRCUSOLVER(cusolverDnXsytrf_bufferSize(handle,n,da,lda,&dA->fact_lwork));
      CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
    }
    if (!dA->d_fact_info) {
      CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info)));
    }
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUSOLVER(cusolverDnXsytrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_ipiv,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info));
    CHKERRQ(PetscLogGpuTimeEnd());
#endif

  A->ops->solve             = MatSolve_SeqDenseCUDA_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_Cholesky;
  CHKERRQ(PetscFree(A->solvertype));
  CHKERRQ(PetscStrallocpy(MATSOLVERCUDA,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactor_SeqDenseCUDA(Mat A,IS col,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscCuBLASInt     m,min,max,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  cusolverDnHandle_t handle;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  CHKERRQ(PetscCUSOLVERDnGetHandle(&handle));
  CHKERRQ(MatDenseCUDAGetArray(A,&da));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&n));
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(a->lda,&lda));
  CHKERRQ(PetscInfo(A,"QR factor %d x %d on backend\n",m,n));
  max = PetscMax(m,n);
  min = PetscMin(m,n);
  if (!dA->d_fact_tau) CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_tau,min*sizeof(*dA->d_fact_tau)));
  if (!dA->d_fact_ipiv) CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv)));
  if (!dA->fact_lwork) {
    CHKERRCUSOLVER(cusolverDnXgeqrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork));
    CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work)));
  }
  if (!dA->d_fact_info) CHKERRCUDA(cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info)));
  if (!dA->workvec) CHKERRQ(VecCreateSeqCUDA(PetscObjectComm((PetscObject)A), m, &(dA->workvec)));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUSOLVER(cusolverDnXgeqrf(handle,m,n,da,lda,dA->d_fact_tau,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(MatDenseCUDARestoreArray(A,&da));
#if defined(PETSC_USE_DEBUG)
  CHKERRCUDA(cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost));
  PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  A->factortype = MAT_FACTOR_QR;
  a->rank = min;
  CHKERRQ(PetscLogGpuFlops(2.0*min*min*(max-min/3.0)));

  A->ops->solve             = MatSolve_SeqDenseCUDA_QR;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDenseCUDA_QR;
  A->ops->matsolve          = MatMatSolve_SeqDenseCUDA_QR;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDenseCUDA_QR;

  CHKERRQ(PetscFree(A->solvertype));
  CHKERRQ(PetscStrallocpy(MATSOLVERCUDA,&A->solvertype));
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(Mat A,Mat B,Mat C,PetscBool tA,PetscBool tB)
{
  const PetscScalar *da,*db;
  PetscScalar       *dc;
  PetscScalar       one=1.0,zero=0.0;
  PetscCuBLASInt    m,n,k;
  PetscInt          alda,blda,clda;
  cublasHandle_t    cublasv2handle;
  PetscBool         Aiscuda,Biscuda;
  cublasStatus_t    berr;

  PetscFunctionBegin;
  /* we may end up with SEQDENSE as one of the arguments */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&Aiscuda));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)B,MATSEQDENSECUDA,&Biscuda));
  if (!Aiscuda) CHKERRQ(MatConvert(A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&A));
  if (!Biscuda) CHKERRQ(MatConvert(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B));
  CHKERRQ(PetscCuBLASIntCast(C->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(C->cmap->n,&n));
  if (tA) CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&k));
  else    CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  CHKERRQ(PetscInfo(C,"Matrix-Matrix product %d x %d x %d on backend\n",m,k,n));
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&da));
  CHKERRQ(MatDenseCUDAGetArrayRead(B,&db));
  CHKERRQ(MatDenseCUDAGetArrayWrite(C,&dc));
  CHKERRQ(MatDenseGetLDA(A,&alda));
  CHKERRQ(MatDenseGetLDA(B,&blda));
  CHKERRQ(MatDenseGetLDA(C,&clda));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscLogGpuTimeBegin());
  berr = cublasXgemm(cublasv2handle,tA ? CUBLAS_OP_T : CUBLAS_OP_N,tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,k,&one,da,alda,db,blda,&zero,dc,clda);CHKERRCUBLAS(berr);
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(1.0*m*n*k + 1.0*m*n*(k-1)));
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&da));
  CHKERRQ(MatDenseCUDARestoreArrayRead(B,&db));
  CHKERRQ(MatDenseCUDARestoreArrayWrite(C,&dc));
  if (!Aiscuda) CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A));
  if (!Biscuda) CHKERRQ(MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_TRUE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqDenseCUDA(Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatProductSetFromOptions_SeqDense(C));
  PetscFunctionReturn(0);
}

/* zz = op(A)*xx + yy
   if yy == NULL, only MatMult */
static PetscErrorCode MatMultAdd_SeqDenseCUDA_Private(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *xarray,*da;
  PetscScalar       *zarray;
  PetscScalar       one=1.0,zero=0.0;
  PetscCuBLASInt    m, n, lda;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;

  PetscFunctionBegin;
   /* mult add */
  if (yy && yy != zz) CHKERRQ(VecCopy_SeqCUDA(yy,zz));
  if (!A->rmap->n || !A->cmap->n) {
    /* mult only */
    if (!yy) CHKERRQ(VecSet_SeqCUDA(zz,0.0));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscInfo(A,"Matrix-vector product %d x %d on backend\n",A->rmap->n,A->cmap->n));
  CHKERRQ(PetscCuBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(A->cmap->n,&n));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&da));
  CHKERRQ(PetscCuBLASIntCast(mat->lda,&lda));
  CHKERRQ(VecCUDAGetArrayRead(xx,&xarray));
  CHKERRQ(VecCUDAGetArray(zz,&zarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  berr = cublasXgemv(cublasv2handle,trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,&one,da,lda,xarray,1,(yy ? &one : &zero),zarray,1);CHKERRCUBLAS(berr);
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(2.0*A->rmap->n*A->cmap->n - (yy ? 0 : A->rmap->n)));
  CHKERRQ(VecCUDARestoreArrayRead(xx,&xarray));
  CHKERRQ(VecCUDARestoreArray(zz,&zarray));
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&da));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_SeqDenseCUDA(Mat A,const PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatSeqDenseCUDACopyFromGPU(A));
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_SeqDenseCUDA(Mat A,PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
  if (!mat->v) CHKERRQ(MatSeqDenseSetPreallocation(A,NULL));
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDenseCUDA(Mat A,PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatSeqDenseCUDACopyFromGPU(A));
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDenseCUDA(Mat Y,PetscScalar alpha)
{
  Mat_SeqDense   *y = (Mat_SeqDense*)Y->data;
  PetscScalar    *dy;
  PetscCuBLASInt j,N,m,lday,one = 1;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(MatDenseCUDAGetArray(Y,&dy));
  CHKERRQ(PetscCuBLASIntCast(Y->rmap->n*Y->cmap->n,&N));
  CHKERRQ(PetscCuBLASIntCast(Y->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(y->lda,&lday));
  CHKERRQ(PetscInfo(Y,"Performing Scale %d x %d on backend\n",Y->rmap->n,Y->cmap->n));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (lday>m) {
    for (j=0; j<Y->cmap->n; j++) CHKERRCUBLAS(cublasXscal(cublasv2handle,m,&alpha,dy+lday*j,one));
  } else CHKERRCUBLAS(cublasXscal(cublasv2handle,N,&alpha,dy,one));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(N));
  CHKERRQ(MatDenseCUDARestoreArray(Y,&dy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseCUDA(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense*)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  PetscCuBLASInt    j,N,m,ldax,lday,one = 1;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(MatDenseCUDAGetArrayRead(X,&dx));
  if (alpha == 0.0) CHKERRQ(MatDenseCUDAGetArrayWrite(Y,&dy));
  else              CHKERRQ(MatDenseCUDAGetArray(Y,&dy));
  CHKERRQ(PetscCuBLASIntCast(X->rmap->n*X->cmap->n,&N));
  CHKERRQ(PetscCuBLASIntCast(X->rmap->n,&m));
  CHKERRQ(PetscCuBLASIntCast(x->lda,&ldax));
  CHKERRQ(PetscCuBLASIntCast(y->lda,&lday));
  CHKERRQ(PetscInfo(Y,"Performing AXPY %d x %d on backend\n",Y->rmap->n,Y->cmap->n));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      CHKERRCUBLAS(cublasXaxpy(cublasv2handle,m,&alpha,dx+j*ldax,one,dy+j*lday,one));
    }
  } else CHKERRCUBLAS(cublasXaxpy(cublasv2handle,N,&alpha,dx,one,dy,one));
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(PetscMax(2.*N-1,0)));
  CHKERRQ(MatDenseCUDARestoreArrayRead(X,&dx));
  if (alpha == 0.0) CHKERRQ(MatDenseCUDARestoreArrayWrite(Y,&dy));
  else              CHKERRQ(MatDenseCUDARestoreArray(Y,&dy));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseCUDA(Mat A)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  if (dA) {
    PetscCheckFalse(dA->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
    if (!dA->user_alloc) CHKERRCUDA(cudaFree(dA->d_v));
    CHKERRCUDA(cudaFree(dA->d_fact_tau));
    CHKERRCUDA(cudaFree(dA->d_fact_ipiv));
    CHKERRCUDA(cudaFree(dA->d_fact_info));
    CHKERRCUDA(cudaFree(dA->d_fact_work));
    CHKERRQ(VecDestroy(&dA->workvec));
  }
  CHKERRQ(PetscFree(A->spptr));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseCUDA(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) A->offloadmask = PETSC_OFFLOAD_CPU;
  CHKERRQ(MatConvert_SeqDenseCUDA_SeqDense(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A));
  CHKERRQ(MatDestroy_SeqDense(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  MatDuplicateOption hcpvalues = (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) ? MAT_DO_NOT_COPY_VALUES : cpvalues;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),B));
  CHKERRQ(MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  CHKERRQ(MatSetType(*B,((PetscObject)A)->type_name));
  CHKERRQ(MatDuplicateNoCreate_SeqDense(*B,A,hcpvalues));
  if (cpvalues == MAT_COPY_VALUES && hcpvalues != MAT_COPY_VALUES) {
    CHKERRQ(MatCopy_SeqDenseCUDA(A,*B,SAME_NONZERO_PATTERN));
  }
  if (cpvalues != MAT_COPY_VALUES) { /* allocate memory if needed */
    Mat_SeqDenseCUDA *dB = (Mat_SeqDenseCUDA*)(*B)->spptr;
    if (!dB->d_v) {
      CHKERRQ(MatSeqDenseCUDASetPreallocation(*B,NULL));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_SeqDenseCUDA(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar      *x;
  PetscBool        viscuda;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)v,&viscuda,VECSEQCUDA,VECMPICUDA,VECCUDA,""));
  if (viscuda && !v->boundtocpu) { /* update device data */
    CHKERRQ(VecCUDAGetArrayWrite(v,&x));
    if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      CHKERRCUDA(cudaMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyHostToHost));
    } else {
      CHKERRCUDA(cudaMemcpy(x,a->v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    }
    CHKERRQ(VecCUDARestoreArrayWrite(v,&x));
  } else { /* update host data */
    CHKERRQ(VecGetArrayWrite(v,&x));
    if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask & PETSC_OFFLOAD_CPU) {
      CHKERRQ(PetscArraycpy(x,a->v+col*a->lda,A->rmap->n));
    } else if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      CHKERRCUDA(cudaMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    }
    CHKERRQ(VecRestoreArrayWrite(v,&x));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),fact));
  CHKERRQ(MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  CHKERRQ(MatSetType(*fact,MATSEQDENSECUDA));
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
    (*fact)->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_QR) {
    CHKERRQ(PetscObjectComposeFunction((PetscObject)(*fact),"MatQRFactor_C",MatQRFactor_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)(*fact),"MatQRFactorSymbolic_C",MatQRFactorSymbolic_SeqDense));
  }
  (*fact)->factortype = ftype;
  CHKERRQ(PetscFree((*fact)->solvertype));
  CHKERRQ(PetscStrallocpy(MATSOLVERCUDA,&(*fact)->solvertype));
  CHKERRQ(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_LU]));
  CHKERRQ(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ILU]));
  CHKERRQ(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]));
  CHKERRQ(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ICC]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  CHKERRQ(MatDenseCUDAGetArray(A,(PetscScalar**)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec));
    CHKERRQ(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  CHKERRQ(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheckFalse(!a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  CHKERRQ(VecCUDAResetArray(a->cvec));
  CHKERRQ(MatDenseCUDARestoreArray(A,(PetscScalar**)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  CHKERRQ(MatDenseCUDAGetArrayRead(A,&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec));
    CHKERRQ(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  CHKERRQ(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  CHKERRQ(VecLockReadPush(a->cvec));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheckFalse(!a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  CHKERRQ(VecLockReadPop(a->cvec));
  CHKERRQ(VecCUDAResetArray(a->cvec));
  CHKERRQ(MatDenseCUDARestoreArrayRead(A,&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  CHKERRQ(MatDenseCUDAGetArrayWrite(A,(PetscScalar**)&a->ptrinuse));
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec));
    CHKERRQ(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  CHKERRQ(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheckFalse(!a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  CHKERRQ(VecCUDAResetArray(a->cvec));
  CHKERRQ(MatDenseCUDARestoreArrayWrite(A,(PetscScalar**)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseCUDA(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && cend-cbegin != a->cmat->cmap->N) {
    CHKERRQ(MatDestroy(&a->cmat));
  }
  CHKERRQ(MatSeqDenseCUDACopyToGPU(A));
  if (!a->cmat) {
    CHKERRQ(MatCreateDenseCUDA(PetscObjectComm((PetscObject)A),A->rmap->n,PETSC_DECIDE,A->rmap->N,cend-cbegin,dA->d_v+(size_t)cbegin*a->lda,&a->cmat));
    CHKERRQ(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat));
  } else {
    CHKERRQ(MatDenseCUDAPlaceArray(a->cmat,dA->d_v+(size_t)cbegin*a->lda));
  }
  CHKERRQ(MatDenseSetLDA(a->cmat,a->lda));
  if (a->v) CHKERRQ(MatDensePlaceArray(a->cmat,a->v+(size_t)cbegin*a->lda));
  a->cmat->offloadmask = A->offloadmask;
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseCUDA(Mat A,Mat *v)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  PetscCheck(a->cmat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column matrix");
  PetscCheck(*v == a->cmat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  A->offloadmask = (a->cmat->offloadmask == PETSC_OFFLOAD_CPU) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  CHKERRQ(MatDenseCUDAResetArray(a->cmat));
  if (a->unplacedarray) CHKERRQ(MatDenseResetArray(a->cmat));
  a->cmat->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatDenseSetLDA_SeqDenseCUDA(Mat A,PetscInt lda)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        data;

  PetscFunctionBegin;
  data = (PetscBool)((A->rmap->n > 0 && A->cmap->n > 0) ? (dA->d_v ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE);
  PetscCheckFalse(!dA->user_alloc && data && cA->lda!=lda,PETSC_COMM_SELF,PETSC_ERR_ORDER,"LDA cannot be changed after allocation of internal storage");
  PetscCheck(lda >= A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %" PetscInt_FMT " must be at least matrix dimension %" PetscInt_FMT,lda,A->rmap->n);
  cA->lda = lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_SeqDenseCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscLayoutSetUp(A->rmap));
  CHKERRQ(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) {
    CHKERRQ(MatSeqDenseCUDASetPreallocation(A,NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqDenseCUDA(Mat A,PetscBool flg)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheckFalse(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = flg;
  if (!flg) {
    PetscBool iscuda;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)a->cvec,VECSEQCUDA,&iscuda));
    if (!iscuda) {
      CHKERRQ(VecDestroy(&a->cvec));
    }
    CHKERRQ(PetscObjectTypeCompare((PetscObject)a->cmat,MATSEQDENSECUDA,&iscuda));
    if (!iscuda) {
      CHKERRQ(MatDestroy(&a->cmat));
    }
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArrayRead_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArrayWrite_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDenseCUDA));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatQRFactor_C",MatQRFactor_SeqDenseCUDA));

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
    A->ops->copy                    = MatCopy_SeqDenseCUDA;
    A->ops->zeroentries             = MatZeroEntries_SeqDenseCUDA;
    A->ops->setup                   = MatSetUp_SeqDenseCUDA;
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    CHKERRQ(MatSeqDenseCUDACopyFromGPU(A));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArray_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArray_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDense));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatQRFactor_C",MatQRFactor_SeqDense));

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
    A->ops->copy                    = MatCopy_SeqDense;
    A->ops->zeroentries             = MatZeroEntries_SeqDense;
    A->ops->setup                   = MatSetUp_SeqDense;
  }
  if (a->cmat) {
    CHKERRQ(MatBindToCPU(a->cmat,flg));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDenseCUDA_SeqDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat              B;
  Mat_SeqDense     *a;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    CHKERRQ(MatConvert_Basic(M,type,reuse,newmat));
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  CHKERRQ(MatBindToCPU_SeqDenseCUDA(B,PETSC_TRUE));
  CHKERRQ(MatReset_SeqDenseCUDA(B));
  CHKERRQ(PetscFree(B->defaultvectype));
  CHKERRQ(PetscStrallocpy(VECSTANDARD,&B->defaultvectype));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensecuda_seqdense_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensecuda_C",NULL));
  a    = (Mat_SeqDense*)B->data;
  CHKERRQ(VecDestroy(&a->cvec)); /* cvec might be VECSEQCUDA. Destroy it and rebuild a VECSEQ when needed */
  B->ops->bindtocpu = NULL;
  B->ops->destroy = MatDestroy_SeqDense;
  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat_SeqDenseCUDA *dB;
  Mat              B;
  Mat_SeqDense     *a;

  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    CHKERRQ(MatConvert_Basic(M,type,reuse,newmat));
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  CHKERRQ(PetscFree(B->defaultvectype));
  CHKERRQ(PetscStrallocpy(VECCUDA,&B->defaultvectype));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSECUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensecuda_seqdense_C",            MatConvert_SeqDenseCUDA_SeqDense));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",                        MatDenseCUDAGetArray_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",                    MatDenseCUDAGetArrayRead_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",                   MatDenseCUDAGetArrayWrite_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",                    MatDenseCUDARestoreArray_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",                MatDenseCUDARestoreArrayRead_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",               MatDenseCUDARestoreArrayWrite_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",                      MatDenseCUDAPlaceArray_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",                      MatDenseCUDAResetArray_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",                    MatDenseCUDAReplaceArray_SeqDenseCUDA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensecuda_C",MatProductSetFromOptions_SeqAIJ_SeqDense));
  a    = (Mat_SeqDense*)B->data;
  CHKERRQ(VecDestroy(&a->cvec)); /* cvec might be VECSEQ. Destroy it and rebuild a VECSEQCUDA when needed */
  CHKERRQ(PetscNewLog(B,&dB));

  B->spptr = dB;
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  CHKERRQ(MatBindToCPU_SeqDenseCUDA(B,PETSC_FALSE));
  B->ops->bindtocpu = MatBindToCPU_SeqDenseCUDA;
  B->ops->destroy  = MatDestroy_SeqDenseCUDA;
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

   Notes:

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqDense()
@*/
PetscErrorCode  MatCreateSeqDenseCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size <= 1,comm,PETSC_ERR_ARG_WRONG,"Invalid communicator size %d",size);
  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,m,n,m,n));
  CHKERRQ(MatSetType(*A,MATSEQDENSECUDA));
  CHKERRQ(MatSeqDenseCUDASetPreallocation(*A,data));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSECUDA - MATSEQDENSECUDA = "seqdensecuda" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensecuda - sets the matrix type to "seqdensecuda" during a call to MatSetFromOptions()

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseCUDA(Mat B)
{
  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  CHKERRQ(MatCreate_SeqDense(B));
  CHKERRQ(MatConvert_SeqDense_SeqDenseCUDA(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B));
  PetscFunctionReturn(0);
}
