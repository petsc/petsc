/*
     Defines the matrix operations for sequential dense with CUDA
*/
#include <petscpkg_version.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsccublas.h>

/* cublas definitions are here */
#include <petsc/private/cudavecimpl.h>

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
#endif
#endif

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool   user_alloc;
  PetscScalar *unplacedarray; /* if one called MatCUDADensePlaceArray(), this is where it stashed the original */
  PetscBool   unplaced_user_alloc;
  /* factorization support */
  int         *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_work; /* device workspace */
  int         fact_lwork;
  int         *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseCUDA;

PetscErrorCode MatSeqDenseCUDASetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;
  PetscBool        iscuda;
  cudaError_t      cerr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&iscuda);CHKERRQ(ierr);
  if (!iscuda) PetscFunctionReturn(0);
  /* it may happen CPU preallocation has not been performed */
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (cA->lda <= 0) cA->lda = A->rmap->n;
  if (!dA->user_alloc) { cerr = cudaFree(dA->d_v);CHKERRCUDA(cerr); }
  if (!d_data) { /* petsc-allocated storage */
    size_t sz;
    ierr = PetscIntMultError(cA->lda,A->cmap->n,NULL);CHKERRQ(ierr);
    sz   = cA->lda*A->cmap->n*sizeof(PetscScalar);
    cerr = cudaMalloc((void**)&dA->d_v,sz);CHKERRCUDA(cerr);
    cerr = cudaMemset(dA->d_v,0,sz);CHKERRCUDA(cerr);
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
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!cA->v) { /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
      ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogEventBegin(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* TODO: it can be done better */
        cerr = cudaMemcpy(cA->v + j*cA->lda,dA->d_v + j*cA->lda,m*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      }
    } else {
      cerr = cudaMemcpy(cA->v,dA->d_v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    }
    ierr = PetscLogGpuToCpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseCUDACopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        copy;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      ierr = MatSeqDenseCUDASetPreallocation(A,NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* TODO: it can be done better */
        cerr = cudaMemcpy(dA->d_v + j*cA->lda,cA->v + j*cA->lda,m*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      }
    } else {
      cerr = cudaMemcpy(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    }
    ierr = PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_SeqDenseCUDA(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense*)B->data;
  PetscErrorCode    ierr;
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt          lda1=a->lda,lda2=b->lda, m=A->rmap->n,n=A->cmap->n, j;
  cudaError_t       cerr;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (m != B->rmap->n || n != B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  ierr = MatDenseCUDAGetArrayRead(A,&va);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayWrite(B,&vb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      cerr = cudaMemcpy(vb+j*lda2,va+j*lda1,m*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    }
  } else {
    cerr = cudaMemcpy(vb,va,m*(n*sizeof(PetscScalar)),cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArray(B,&vb);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&va);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAPlaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (aa->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (aa->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
  if (aa->v) { ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr); }
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->v) { ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr); }
  dA->d_v = dA->unplacedarray;
  dA->user_alloc = dA->unplaced_user_alloc;
  dA->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAReplaceArray_SeqDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (aa->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (aa->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
  if (!dA->user_alloc) { cerr = cudaFree(dA->d_v);CHKERRCUDA(cerr); }
  dA->d_v = (PetscScalar*)a;
  dA->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!dA->d_v) {
    ierr = MatSeqDenseCUDASetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayWrite_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayRead_SeqDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayRead_SeqDenseCUDA(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArray_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArray_SeqDenseCUDA(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseCUDAInvertFactors_Private(Mat A)
{
#if PETSC_PKG_CUDA_VERSION_GE(10,1,0)
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscErrorCode     ierr;
  cudaError_t        ccer;
  cusolverStatus_t   cerr;
  cusolverDnHandle_t handle;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDngetri not implemented");
  else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      int il;

      ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
      cerr = cusolverDnXpotri_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&il);CHKERRCUSOLVER(cerr);
      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        ccer = cudaFree(dA->d_fact_work);CHKERRCUDA(ccer);
        ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
      }
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cusolverDnXpotri(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
      /* TODO (write cuda kernel) */
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytri not implemented");
  }
#if defined(PETSC_USE_DEBUG)
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: leading minor of order %d is zero",info);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Upgrade to CUDA version 10.1.0 or higher");
#endif
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA(Mat A,Mat B,Mat X)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense       *x = (Mat_SeqDense*)X->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *dx;
  cusolverDnHandle_t handle;
  PetscBool          iscuda;
  int                nrhs,n,lda,ldx;
#if defined(PETSC_USE_DEBUG)
  int                info;
  cudaError_t        ccer;
#endif
  cusolverStatus_t   cerr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&iscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (X != B) {
    ierr = MatCopy(B,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  /* MatMatSolve does not have a dispatching mechanism, we may end up with a MATSEQDENSE here */
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSECUDA,&iscuda);CHKERRQ(ierr);
  if (!iscuda) {
    ierr = MatConvert(X,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
  ierr = MatDenseCUDAGetArray(X,&dx);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(X->cmap->n,&nrhs);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(x->lda,&ldx);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = cusolverDnXgetrs(handle,CUBLAS_OP_N,n,nrhs,da,lda,dA->d_fact_ipiv,dx,ldx,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
      cerr = cusolverDnXpotrs(handle,CUBLAS_FILL_MODE_LOWER,n,nrhs,da,lda,dx,ldx,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArray(X,&dx);CHKERRQ(ierr);
  if (!iscuda) {
    ierr = MatConvert(X,MATSEQDENSE,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(nrhs*(2.0*n*n - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Private(Mat A,Vec xx,Vec yy,PetscBool trans)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *y;
  cusolverDnHandle_t handle;
  int                one = 1,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
  cudaError_t        ccer;
#endif
  cusolverStatus_t   cerr;
  PetscBool          iscuda;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  /* MatSolve does not have a dispatching mechanism, we may end up with a VECSTANDARD here */
  ierr = PetscObjectTypeCompareAny((PetscObject)yy,&iscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (iscuda) {
    ierr = VecCopy(xx,yy);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yy,&y);CHKERRQ(ierr);
  } else {
    if (!dA->workvec) {
      ierr = MatCreateVecs(A,&dA->workvec,NULL);CHKERRQ(ierr);
    }
    ierr = VecCopy(xx,dA->workvec);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(dA->workvec,&y);CHKERRQ(ierr);
  }
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = cusolverDnXgetrs(handle,trans ? CUBLAS_OP_T : CUBLAS_OP_N,n,one,da,lda,dA->d_fact_ipiv,y,n,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
      cerr = cusolverDnXpotrs(handle,CUBLAS_FILL_MODE_LOWER,n,one,da,lda,y,n,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (iscuda) {
    ierr = VecCUDARestoreArray(yy,&y);CHKERRQ(ierr);
  } else {
    ierr = VecCUDARestoreArray(dA->workvec,&y);CHKERRQ(ierr);
    ierr = VecCopy(dA->workvec,yy);CHKERRQ(ierr);
  }
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(2.0*n*n - n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseCUDA_Private(A,xx,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseCUDA_Private(A,xx,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseCUDA(Mat A,IS rperm,IS cperm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  int                m,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  cusolverStatus_t   cerr;
  cusolverDnHandle_t handle;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"LU factor %d x %d on backend\n",m,n);CHKERRQ(ierr);
  if (!dA->d_fact_ipiv) {
    ccer = cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRCUDA(ccer);
  }
  if (!dA->fact_lwork) {
    cerr = cusolverDnXgetrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
    ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
  }
  if (!dA->d_fact_info) {
    ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = cusolverDnXgetrf(handle,m,n,da,lda,dA->d_fact_work,dA->d_fact_ipiv,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
  A->factortype = MAT_FACTOR_LU;
  ierr = PetscLogGpuFlops(2.0*n*n*m/3.0);CHKERRQ(ierr);

  A->ops->solve          = MatSolve_SeqDenseCUDA;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseCUDA;
  A->ops->matsolve       = MatMatSolve_SeqDenseCUDA;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseCUDA(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  cusolverStatus_t   cerr;
  cusolverDnHandle_t handle;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"Cholesky factor %d x %d on backend\n",n,n);CHKERRQ(ierr);
  if (A->spd) {
    ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
    if (!dA->fact_lwork) {
      cerr = cusolverDnXpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
      ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cusolverDnXpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cusolverDnsytrs unavailable. Use MAT_FACTOR_LU");
#if 0
    /* at the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs and *hetr* routines
       The code below should work, and it can be activated when *sytrs routines will be available */
    if (!dA->d_fact_ipiv) {
      ccer = cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRCUDA(ccer);
    }
    if (!dA->fact_lwork) {
      cerr = cusolverDnXsytrf_bufferSize(handle,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
      ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cusolverDnXsytrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_ipiv,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif

  A->ops->solve          = MatSolve_SeqDenseCUDA;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseCUDA;
  A->ops->matsolve       = MatMatSolve_SeqDenseCUDA;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(Mat A,Mat B,Mat C,PetscBool tA,PetscBool tB)
{
  const PetscScalar *da,*db;
  PetscScalar       *dc;
  PetscScalar       one=1.0,zero=0.0;
  int               m,n,k;
  PetscInt          alda,blda,clda;
  PetscErrorCode    ierr;
  cublasHandle_t    cublasv2handle;
  PetscBool         Aiscuda,Biscuda;
  cublasStatus_t    berr;

  PetscFunctionBegin;
  /* we may end up with SEQDENSE as one of the arguments */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSECUDA,&Aiscuda);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSECUDA,&Biscuda);CHKERRQ(ierr);
  if (!Aiscuda) {
    ierr = MatConvert(A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  if (!Biscuda) {
    ierr = MatConvert(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
  ierr = PetscMPIIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  if (tA) {
    ierr = PetscMPIIntCast(A->rmap->n,&k);CHKERRQ(ierr);
  } else {
    ierr = PetscMPIIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  }
  if (!m || !n || !k) PetscFunctionReturn(0);
  ierr = PetscInfo3(C,"Matrix-Matrix product %d x %d x %d on backend\n",m,k,n);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayWrite(C,&dc);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(A,&alda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemm(cublasv2handle,tA ? CUBLAS_OP_T : CUBLAS_OP_N,tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,k,&one,da,alda,db,blda,&zero,dc,clda);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayWrite(C,&dc);CHKERRQ(ierr);
  if (!Aiscuda) {
    ierr = MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  if (!Biscuda) {
    ierr = MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqDenseCUDA(Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatProductSetFromOptions_SeqDense(C);CHKERRQ(ierr);
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
  int               m, n, lda; /* Use PetscMPIInt as it is typedef'ed to int */
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (yy && yy != zz) { /* mult add */
    ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) {
    if (!yy) { /* mult only */
      ierr = VecSet_SeqCUDA(zz,0.0);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo2(A,"Matrix-vector product %d x %d on backend\n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(mat->lda,&lda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemv(cublasv2handle,trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,&one,da,lda,xarray,1,(yy ? &one : &zero),zarray,1);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*A->rmap->n*A->cmap->n - (yy ? 0 : A->rmap->n));CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_SeqDenseCUDA(Mat A,const PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_SeqDenseCUDA(Mat A,PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mat->v) { /* MatCreateSeqDenseCUDA may not allocate CPU memory. Allocate if needed */
    ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDenseCUDA(Mat A,PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDenseCUDA(Mat Y,PetscScalar alpha)
{
  Mat_SeqDense   *y = (Mat_SeqDense*)Y->data;
  PetscScalar    *dy;
  int            j,N,m,lday,one = 1;
  cublasHandle_t cublasv2handle;
  cublasStatus_t berr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArray(Y,&dy);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(Y->rmap->n*Y->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(Y->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(y->lda,&lday);CHKERRQ(ierr);
  ierr = PetscInfo2(Y,"Performing Scale %d x %d on backend\n",Y->rmap->n,Y->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (lday>m) {
    for (j=0; j<Y->cmap->n; j++) {
      berr = cublasXscal(cublasv2handle,m,&alpha,dy+lday*j,one);CHKERRCUBLAS(berr);
    }
  } else {
    berr = cublasXscal(cublasv2handle,N,&alpha,dy,one);CHKERRCUBLAS(berr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(N);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArray(Y,&dy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseCUDA(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense*)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  int               j,N,m,ldax,lday,one = 1;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseCUDAGetArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDAGetArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  ierr = PetscMPIIntCast(X->rmap->n*X->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(X->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(x->lda,&ldax);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(y->lda,&lday);CHKERRQ(ierr);
  ierr = PetscInfo2(Y,"Performing AXPY %d x %d on backend\n",Y->rmap->n,Y->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      berr = cublasXaxpy(cublasv2handle,m,&alpha,dx+j*ldax,one,dy+j*lday,one);CHKERRCUBLAS(berr);
    }
  } else {
    berr = cublasXaxpy(cublasv2handle,N,&alpha,dx,one,dy,one);CHKERRCUBLAS(berr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2.*N-1,0));CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseCUDARestoreArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDARestoreArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseCUDA(Mat A)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  cudaError_t      cerr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dA) {
    if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseCUDAResetArray() must be called first");
    if (!dA->user_alloc) { cerr = cudaFree(dA->d_v);CHKERRCUDA(cerr); }
    cerr = cudaFree(dA->d_fact_ipiv);CHKERRCUDA(cerr);
    cerr = cudaFree(dA->d_fact_info);CHKERRCUDA(cerr);
    cerr = cudaFree(dA->d_fact_work);CHKERRCUDA(cerr);
    ierr = VecDestroy(&dA->workvec);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseCUDA(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) { A->offloadmask = PETSC_OFFLOAD_CPU; }
  ierr = MatConvert_SeqDenseCUDA_SeqDense(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatDestroy_SeqDense(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*B,A,cpvalues);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) {
    Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
    const PetscScalar *da;
    PetscScalar       *db;
    cudaError_t       cerr;
    PetscInt          ldb;

    ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseCUDAGetArrayWrite(*B,&db);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(*B,&ldb);CHKERRQ(ierr);
    if (a->lda > A->rmap->n || ldb > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* it can be done better */
        cerr = cudaMemcpy(db+j*ldb,da+j*a->lda,m*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
      }
    } else {
      cerr = cudaMemcpy(db,da,(sizeof(PetscScalar)*A->cmap->n)*A->rmap->n,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    }
    ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseCUDARestoreArrayWrite(*B,&db);CHKERRQ(ierr);
    (*B)->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

#include <petsc/private/vecimpl.h>

static PetscErrorCode MatGetColumnVector_SeqDenseCUDA(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;
  PetscScalar      *x;
  PetscBool        viscuda;
  cudaError_t      cerr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&viscuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (viscuda && !v->boundtocpu) { /* update device data */
    ierr = VecCUDAGetArrayWrite(v,&x);CHKERRQ(ierr);
    if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      cerr = cudaMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyHostToHost);CHKERRCUDA(cerr);
    } else {
      cerr = cudaMemcpy(x,a->v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    }
    ierr = VecCUDARestoreArrayWrite(v,&x);CHKERRQ(ierr);
  } else { /* update host data */
    ierr = VecGetArrayWrite(v,&x);CHKERRQ(ierr);
    if (A->offloadmask & PETSC_OFFLOAD_CPU) {
      ierr = PetscArraycpy(x,a->v+col*a->lda,A->rmap->n);CHKERRQ(ierr);
    } else {
      cerr = cudaMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    }
    ierr = VecRestoreArrayWrite(v,&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,MATSEQDENSECUDA);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;
  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&(*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ILU]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ICC]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAGetArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    ierr = VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAGetArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    ierr = VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  ierr = VecLockReadPush(a->cvec);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecLockReadPop(a->cvec);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAGetArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecCUDAPlaceArray is called */
    ierr = VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseCUDA(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && cend-cbegin != a->cmat->cmap->N) {
    ierr = MatDestroy(&a->cmat);CHKERRQ(ierr);
  }
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  if (!a->cmat) {
    ierr = MatCreateDenseCUDA(PetscObjectComm((PetscObject)A),A->rmap->n,PETSC_DECIDE,A->rmap->N,cend-cbegin,dA->d_v + (size_t)cbegin * (size_t)a->lda,&a->cmat);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDAPlaceArray(a->cmat,dA->d_v + (size_t)cbegin * (size_t)a->lda);CHKERRQ(ierr);
  }
  ierr = MatDenseSetLDA(a->cmat,a->lda);CHKERRQ(ierr);
  if (a->v) { ierr = MatDensePlaceArray(a->cmat,a->v + (size_t)cbegin * (size_t)a->lda);CHKERRQ(ierr); }
  a->cmat->offloadmask = A->offloadmask;
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseCUDA(Mat A,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  if (!a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column matrix");
  if (*v != a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatDenseCUDAResetArray(a->cmat);CHKERRQ(ierr);
  ierr = MatDenseResetArray(a->cmat);CHKERRQ(ierr);
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
  if (!dA->user_alloc && data && cA->lda!=lda) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"LDA cannot be changed after allocation of internal storage");
  if (lda < A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %D must be at least matrix dimension %D",lda,A->rmap->n);
  cA->lda = lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqDenseCUDA(Mat A,PetscBool flg)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = flg;
  if (!flg) {
    PetscBool iscuda;

    ierr = PetscObjectTypeCompare((PetscObject)a->cvec,VECSEQCUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = VecDestroy(&a->cvec);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)a->cmat,MATSEQDENSECUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = MatDestroy(&a->cmat);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArrayRead_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArrayWrite_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDenseCUDA);CHKERRQ(ierr);

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
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDense);CHKERRQ(ierr);

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
  }
  if (a->cmat) {
    ierr = MatBindToCPU(a->cmat,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDenseCUDA_SeqDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat              B;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    ierr = MatConvert_Basic(M,type,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  ierr = MatBindToCPU_SeqDenseCUDA(B,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatReset_SeqDenseCUDA(B);CHKERRQ(ierr);
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensecuda_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensecuda_C",NULL);CHKERRQ(ierr);

  B->ops->bindtocpu = NULL;
  B->ops->destroy = MatDestroy_SeqDense;
  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat_SeqDenseCUDA *dB;
  Mat              B;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    ierr = MatConvert_Basic(M,type,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSECUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensecuda_seqdense_C",            MatConvert_SeqDenseCUDA_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",                        MatDenseCUDAGetArray_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",                    MatDenseCUDAGetArrayRead_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",                   MatDenseCUDAGetArrayWrite_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",                    MatDenseCUDARestoreArray_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",                MatDenseCUDARestoreArrayRead_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",               MatDenseCUDARestoreArrayWrite_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",                      MatDenseCUDAPlaceArray_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",                      MatDenseCUDAResetArray_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",                    MatDenseCUDAReplaceArray_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensecuda_C",MatProductSetFromOptions_SeqAIJ_SeqDense);CHKERRQ(ierr);

  ierr     = PetscNewLog(B,&dB);CHKERRQ(ierr);
  B->spptr = dB;

  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  ierr = MatBindToCPU_SeqDenseCUDA(B,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Invalid communicator size %d",size);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSECUDA);CHKERRQ(ierr);
  ierr = MatSeqDenseCUDASetPreallocation(*A,data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_SeqDense(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqDense_SeqDenseCUDA(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
