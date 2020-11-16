#if !defined(__CUSPARSEMATIMPL)
#define __CUSPARSEMATIMPL

#include <petscpkg_version.h>
#include <petsc/private/cudavecimpl.h>

#include <cusparse_v2.h>

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#if (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) /* According to cuda/10.1.168 on OLCF Summit */
#define CHKERRCUSPARSE(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name  = cusparseGetErrorName(stat); \
      const char *descr = cusparseGetErrorString(stat); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuSPARSE error %d (%s) : %s",(int)stat,name,descr); \
   } \
} while (0)
#else
#define CHKERRCUSPARSE(stat) do {if (PetscUnlikely(stat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cusparse error %d",(int)stat);} while (0)
#endif

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    const cuComplex PETSC_CUSPARSE_ONE        = {1.0f, 0.0f};
    const cuComplex PETSC_CUSPARSE_ZERO       = {0.0f, 0.0f};
  #elif defined(PETSC_USE_REAL_DOUBLE)
    const cuDoubleComplex PETSC_CUSPARSE_ONE  = {1.0, 0.0};
    const cuDoubleComplex PETSC_CUSPARSE_ZERO = {0.0, 0.0};
  #endif
#else
  const PetscScalar PETSC_CUSPARSE_ONE        = 1.0;
  const PetscScalar PETSC_CUSPARSE_ZERO       = 0.0;
#endif

#if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  #define cusparse_create_analysis_info  cusparseCreateCsrsv2Info
  #define cusparse_destroy_analysis_info cusparseDestroyCsrsv2Info
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) cusparseCcsrsv2_bufferSize(a,b,c,d,e,(cuComplex*)(f),g,h,i,j)
      #define cusparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     cusparseCcsrsv2_analysis(a,b,c,d,e,(const cuComplex*)(f),g,h,i,j,k)
      #define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  cusparseCcsrsv2_solve(a,b,c,d,(const cuComplex*)(e),f,(const cuComplex*)(g),h,i,j,(const cuComplex*)(k),(cuComplex*)(l),m,n)
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) cusparseZcsrsv2_bufferSize(a,b,c,d,e,(cuDoubleComplex*)(f),g,h,i,j)
      #define cusparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     cusparseZcsrsv2_analysis(a,b,c,d,e,(const cuDoubleComplex*)(f),g,h,i,j,k)
      #define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  cusparseZcsrsv2_solve(a,b,c,d,(const cuDoubleComplex*)(e),f,(const cuDoubleComplex*)(g),h,i,j,(const cuDoubleComplex*)(k),(cuDoubleComplex*)(l),m,n)
    #endif
  #else /* not complex */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_get_svbuffsize cusparseScsrsv2_bufferSize
      #define cusparse_analysis       cusparseScsrsv2_analysis
      #define cusparse_solve          cusparseScsrsv2_solve
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_get_svbuffsize cusparseDcsrsv2_bufferSize
      #define cusparse_analysis       cusparseDcsrsv2_analysis
      #define cusparse_solve          cusparseDcsrsv2_solve
    #endif
  #endif
#else
  #define cusparse_create_analysis_info  cusparseCreateSolveAnalysisInfo
  #define cusparse_destroy_analysis_info cusparseDestroySolveAnalysisInfo
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k) cusparseCcsrsv_solve((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h),(i),(cuComplex*)(j),(cuComplex*)(k))
      #define cusparse_analysis(a,b,c,d,e,f,g,h,i)  cusparseCcsrsv_analysis((a),(b),(c),(d),(e),(cuComplex*)(f),(g),(h),(i))
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k) cusparseZcsrsv_solve((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k))
      #define cusparse_analysis(a,b,c,d,e,f,g,h,i)  cusparseZcsrsv_analysis((a),(b),(c),(d),(e),(cuDoubleComplex*)(f),(g),(h),(i))
    #endif
  #else /* not complex */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_solve    cusparseScsrsv_solve
      #define cusparse_analysis cusparseScsrsv_analysis
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_solve    cusparseDcsrsv_solve
      #define cusparse_analysis cusparseDcsrsv_analysis
    #endif
  #endif
#endif

#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  #define cusparse_csr2csc cusparseCsr2cscEx2
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_scalartype CUDA_C_32F
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_scalartype CUDA_C_64F
    #endif
  #else /* not complex */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_scalartype CUDA_R_32F
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_scalartype CUDA_R_64F
    #endif
  #endif
#else
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       cusparseCcsrmv((a),(b),(c),(d),(e),(cuComplex*)(f),(g),(cuComplex*)(h),(i),(j),(cuComplex*)(k),(cuComplex*)(l),(cuComplex*)(m))
      #define cusparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusparseCcsrmm((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(j),(k),(cuComplex*)(l),(m),(cuComplex*)(n),(cuComplex*)(o),(p))
      #define cusparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          cusparseCcsr2csc((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(cuComplex*)(h),(i),(j),(k),(l))
      #define cusparse_hyb_spmv(a,b,c,d,e,f,g,h)                 cusparseChybmv((a),(b),(cuComplex*)(c),(d),(e),(cuComplex*)(f),(cuComplex*)(g),(cuComplex*)(h))
      #define cusparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              cusparseCcsr2hyb((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(h),(i),(j))
      #define cusparse_hyb2csr(a,b,c,d,e,f)                      cusparseChyb2csr((a),(b),(c),(cuComplex*)(d),(e),(f))
      #define cusparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) cusparseCcsrgemm(a,b,c,d,e,f,g,h,(cuComplex*)i,j,k,l,m,(cuComplex*)n,o,p,q,(cuComplex*)r,s,t)
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       cusparseZcsrmv((a),(b),(c),(d),(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(j),(cuDoubleComplex*)(k),(cuDoubleComplex*)(l),(cuDoubleComplex*)(m))
      #define cusparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusparseZcsrmm((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(j),(k),(cuDoubleComplex*)(l),(m),(cuDoubleComplex*)(n),(cuDoubleComplex*)(o),(p))
      #define cusparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          cusparseZcsr2csc((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j),(k),(l))
      #define cusparse_hyb_spmv(a,b,c,d,e,f,g,h)                 cusparseZhybmv((a),(b),(cuDoubleComplex*)(c),(d),(e),(cuDoubleComplex*)(f),(cuDoubleComplex*)(g),(cuDoubleComplex*)(h))
      #define cusparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              cusparseZcsr2hyb((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(h),(i),(j))
      #define cusparse_hyb2csr(a,b,c,d,e,f)                      cusparseZhyb2csr((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
      #define cusparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) cusparseZcsrgemm(a,b,c,d,e,f,g,h,(cuDoubleComplex*)i,j,k,l,m,(cuDoubleComplex*)n,o,p,q,(cuDoubleComplex*)r,s,t)
    #endif
  #else
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusparse_csr_spmv cusparseScsrmv
      #define cusparse_csr_spmm cusparseScsrmm
      #define cusparse_csr2csc  cusparseScsr2csc
      #define cusparse_hyb_spmv cusparseShybmv
      #define cusparse_csr2hyb  cusparseScsr2hyb
      #define cusparse_hyb2csr  cusparseShyb2csr
      #define cusparse_csr_spgemm  cusparseScsrgemm
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define cusparse_csr_spmv cusparseDcsrmv
      #define cusparse_csr_spmm cusparseDcsrmm
      #define cusparse_csr2csc  cusparseDcsr2csc
      #define cusparse_hyb_spmv cusparseDhybmv
      #define cusparse_csr2hyb  cusparseDcsr2hyb
      #define cusparse_hyb2csr  cusparseDhyb2csr
      #define cusparse_csr_spgemm cusparseDcsrgemm
    #endif
  #endif
#endif

#define THRUSTINTARRAY32 thrust::device_vector<int>
#define THRUSTINTARRAY thrust::device_vector<PetscInt>
#define THRUSTARRAY thrust::device_vector<PetscScalar>

/* A CSR matrix structure */
struct CsrMatrix {
  PetscInt         num_rows;
  PetscInt         num_cols;
  PetscInt         num_entries;
  THRUSTINTARRAY32 *row_offsets;
  THRUSTINTARRAY32 *column_indices;
  THRUSTARRAY      *values;
};

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJCUSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  cusparseMatDescr_t          descr;
  cusparseOperation_t         solveOp;
  CsrMatrix                   *csrMat;
 #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  csrsv2Info_t                solveInfo;
 #else
  cusparseSolveAnalysisInfo_t solveInfo;
 #endif
  cusparseSolvePolicy_t       solvePolicy;     /* whether level information is generated and used */
  int                         solveBufferSize;
  void                        *solveBuffer;
  size_t                      csr2cscBufferSize; /* to transpose the triangular factor (only used for CUDA >= 11.0) */
  void                        *csr2cscBuffer;
  PetscScalar                 *AA_h; /* managed host buffer for moving values to the GPU */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and any indices used in a reordering */
struct Mat_SeqAIJCUSPARSETriFactors {
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactorPtrTranspose; /* pointer for lower triangular (factored matrix) on GPU for the transpose (useful for BiCG) */
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactorPtrTranspose; /* pointer for upper triangular (factored matrix) on GPU for the transpose (useful for BiCG)*/
  THRUSTINTARRAY                    *rpermIndices;  /* indices used for any reordering */
  THRUSTINTARRAY                    *cpermIndices;  /* indices used for any reordering */
  THRUSTARRAY                       *workVector;
  cusparseHandle_t                  handle;   /* a handle to the cusparse library */
  PetscInt                          nnz;      /* number of nonzeros ... need this for accurate logging between ICC and ILU */
};

struct Mat_CusparseSpMV {
  PetscBool             initialized;    /* Don't rely on spmvBuffer != NULL to test if the struct is initialized, */
  size_t                spmvBufferSize; /* since I'm not sure if smvBuffer can be NULL even after cusparseSpMV_bufferSize() */
  void                  *spmvBuffer;
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)  /* these are present from CUDA 10.1, but PETSc code makes use of them from CUDA 11 on */
  cusparseDnVecDescr_t  vecXDescr,vecYDescr; /* descriptor for the dense vectors in y=op(A)x */
 #endif
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJCUSPARSEMultStruct {
  void               *mat;  /* opaque pointer to a matrix. This could be either a cusparseHybMat_t or a CsrMatrix */
  cusparseMatDescr_t descr; /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY     *cprowIndices;   /* compressed row indices used in the parallel SpMV */
  PetscScalar        *alpha_one; /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar        *beta_zero; /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar        *beta_one; /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseSpMatDescr_t  matDescr;  /* descriptor for the matrix, used by SpMV and SpMM */
  Mat_CusparseSpMV      cuSpMV[3]; /* different Mat_CusparseSpMV structs for non-transpose, transpose, conj-transpose */
  Mat_SeqAIJCUSPARSEMultStruct() : matDescr(NULL) {
    for (int i=0; i<3; i++) cuSpMV[i].initialized = PETSC_FALSE;
  }
 #endif
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Tranpose */
struct Mat_SeqAIJCUSPARSE {
  Mat_SeqAIJCUSPARSEMultStruct *mat;            /* pointer to the matrix on the GPU */
  Mat_SeqAIJCUSPARSEMultStruct *matTranspose;   /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                  *workVector;     /* pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  THRUSTINTARRAY32             *rowoffsets_gpu; /* rowoffsets on GPU in non-compressed-row format. It is used to convert CSR to CSC */
  PetscInt                     nrows;           /* number of rows of the matrix seen by GPU */
  MatCUSPARSEStorageFormat     format;          /* the storage format for the matrix on the device */
  cudaStream_t                 stream;          /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  cusparseHandle_t             handle;          /* a handle to the cusparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState             nonzerostate;    /* track nonzero state to possibly recreate the GPU matrix */
  PetscBool                    transgen;        /* whether or not to generate explicit transpose for MatMultTranspose operations */
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  size_t                       csr2cscBufferSize; /* stuff used to compute the matTranspose above */
  void                         *csr2cscBuffer;    /* This is used as a C struct and is calloc'ed by PetscNewLog() */
  cusparseCsr2CscAlg_t         csr2cscAlg;        /* algorithms can be selected from command line options */
  cusparseSpMVAlg_t            spmvAlg;
  cusparseSpMMAlg_t            spmmAlg;
 #endif
  PetscSplitCSRDataStructure   *deviceMat;       /* Matrix on device for, eg, assembly */
  THRUSTINTARRAY               *cooPerm;
  THRUSTINTARRAY               *cooPerm_a;
};

PETSC_INTERN PetscErrorCode MatCUSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatCUSPARSESetStream(Mat, const cudaStream_t stream);
PETSC_INTERN PetscErrorCode MatCUSPARSESetHandle(Mat, const cusparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatCUSPARSEClearHandle(Mat);
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat,const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSEGetArrayRead(Mat,const PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSERestoreArrayRead(Mat,const PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSEGetArrayWrite(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSERestoreArrayWrite(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSEMergeMats(Mat,Mat,MatReuse,Mat*);

PETSC_STATIC_INLINE bool isCudaMem(const void *data)
{
  cudaError_t                  cerr;
  struct cudaPointerAttributes attr;
  enum cudaMemoryType          mtype;
  cerr = cudaPointerGetAttributes(&attr,data); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
  cudaGetLastError(); /* Reset the last error */
  #if (CUDART_VERSION < 10000)
    mtype = attr.memoryType;
  #else
    mtype = attr.type;
  #endif
  if (cerr == cudaSuccess && mtype == cudaMemoryTypeDevice) return true;
  else return false;
}

#endif
