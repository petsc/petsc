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
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"cuSPARSE error %d (%s) : %s",(int)stat,name,descr); \
   } \
} while (0)
#else
#define CHKERRCUSPARSE(stat) do {if (PetscUnlikely(stat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusparse error %d",(int)stat);} while (0)
#endif

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
const cuComplex PETSC_CUSPARSE_ONE  = {1.0f, 0.0f};
const cuComplex PETSC_CUSPARSE_ZERO = {0.0f, 0.0f};
#define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k)              cusparseCcsrsv_solve((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h),(i),(cuComplex*)(j),(cuComplex*)(k))
#define cusparse_analysis(a,b,c,d,e,f,g,h,i)               cusparseCcsrsv_analysis((a),(b),(c),(d),(e),(cuComplex*)(f),(g),(h),(i))
#define cusparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       cusparseCcsrmv((a),(b),(c),(d),(e),(cuComplex*)(f),(g),(cuComplex*)(h),(i),(j),(cuComplex*)(k),(cuComplex*)(l),(cuComplex*)(m))
#define cusparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusparseCcsrmm((a),(b),(c),(d),(e),(f),(cuComplex*)(g),(h),(cuComplex*)(i),(j),(k),(cuComplex*)(l),(m),(cuComplex*)(n),(cuComplex*)(o),(p))
#define cusparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          cusparseCcsr2csc((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(cuComplex*)(h),(i),(j),(k),(l))
#define cusparse_hyb_spmv(a,b,c,d,e,f,g,h)                 cusparseChybmv((a),(b),(cuComplex*)(c),(d),(e),(cuComplex*)(f),(cuComplex*)(g),(cuComplex*)(h))
#define cusparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              cusparseCcsr2hyb((a),(b),(c),(d),(cuComplex*)(e),(f),(g),(h),(i),(j))
#define cusparse_hyb2csr(a,b,c,d,e,f)                      cusparseChyb2csr((a),(b),(c),(cuComplex*)(d),(e),(f))
#elif defined(PETSC_USE_REAL_DOUBLE)
const cuDoubleComplex PETSC_CUSPARSE_ONE  = {1.0, 0.0};
const cuDoubleComplex PETSC_CUSPARSE_ZERO = {0.0, 0.0};
#define cusparse_solve(a,b,c,d,e,f,g,h,i,j,k)              cusparseZcsrsv_solve((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k))
#define cusparse_analysis(a,b,c,d,e,f,g,h,i)               cusparseZcsrsv_analysis((a),(b),(c),(d),(e),(cuDoubleComplex*)(f),(g),(h),(i))
#define cusparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       cusparseZcsrmv((a),(b),(c),(d),(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(j),(cuDoubleComplex*)(k),(cuDoubleComplex*)(l),(cuDoubleComplex*)(m))
#define cusparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusparseZcsrmm((a),(b),(c),(d),(e),(f),(cuDoubleComplex*)(g),(h),(cuDoubleComplex*)(i),(j),(k),(cuDoubleComplex*)(l),(m),(cuDoubleComplex*)(n),(cuDoubleComplex*)(o),(p))
#define cusparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          cusparseZcsr2csc((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j),(k),(l))
#define cusparse_hyb_spmv(a,b,c,d,e,f,g,h)                 cusparseZhybmv((a),(b),(cuDoubleComplex*)(c),(d),(e),(cuDoubleComplex*)(f),(cuDoubleComplex*)(g),(cuDoubleComplex*)(h))
#define cusparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              cusparseZcsr2hyb((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(h),(i),(j))
#define cusparse_hyb2csr(a,b,c,d,e,f)                      cusparseZhyb2csr((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#endif
#else
const PetscScalar PETSC_CUSPARSE_ONE  = 1.0;
const PetscScalar PETSC_CUSPARSE_ZERO = 0.0;
#if defined(PETSC_USE_REAL_SINGLE)
#define cusparse_solve    cusparseScsrsv_solve
#define cusparse_analysis cusparseScsrsv_analysis
#define cusparse_csr_spmv cusparseScsrmv
#define cusparse_csr_spmm cusparseScsrmm
#define cusparse_csr2csc  cusparseScsr2csc
#define cusparse_hyb_spmv cusparseShybmv
#define cusparse_csr2hyb  cusparseScsr2hyb
#define cusparse_hyb2csr  cusparseShyb2csr
#elif defined(PETSC_USE_REAL_DOUBLE)
#define cusparse_solve    cusparseDcsrsv_solve
#define cusparse_analysis cusparseDcsrsv_analysis
#define cusparse_csr_spmv cusparseDcsrmv
#define cusparse_csr_spmm cusparseDcsrmm
#define cusparse_csr2csc  cusparseDcsr2csc
#define cusparse_hyb_spmv cusparseDhybmv
#define cusparse_csr2hyb  cusparseDcsr2hyb
#define cusparse_hyb2csr  cusparseDhyb2csr
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

#if PETSC_PKG_CUDA_VERSION_LT(11,0,0)
/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJCUSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  cusparseMatDescr_t          descr;
  cusparseSolveAnalysisInfo_t solveInfo;
  cusparseOperation_t         solveOp;
  CsrMatrix                   *csrMat;
};
#endif

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJCUSPARSEMultStruct {
  void               *mat;  /* opaque pointer to a matrix. This could be either a cusparseHybMat_t or a CsrMatrix */
  cusparseMatDescr_t descr; /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY     *cprowIndices;   /* compressed row indices used in the parallel SpMV */
  PetscScalar        *alpha; /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar        *beta_zero; /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar        *beta_one; /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
};

#if PETSC_PKG_CUDA_VERSION_LT(11,0,0)

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and
 any indices used in a reordering */
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
#endif

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
};

PETSC_INTERN PetscErrorCode MatCUSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatCUSPARSESetStream(Mat, const cudaStream_t stream);
PETSC_INTERN PetscErrorCode MatCUSPARSESetHandle(Mat, const cusparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatCUSPARSEClearHandle(Mat);
#endif
