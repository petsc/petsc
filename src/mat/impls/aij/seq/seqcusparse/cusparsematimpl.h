#pragma once

#include <petscpkg_version.h>
#include <../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp> /* for VecSeq_CUPM */
#include <../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp>
#include <petsc/private/petsclegacycupmblas.h>

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/system/system_error.h>

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
const cuComplex PETSC_CUSPARSE_ONE  = {1.0f, 0.0f};
const cuComplex PETSC_CUSPARSE_ZERO = {0.0f, 0.0f};
    #define cusparseXcsrilu02_bufferSize(a, b, c, d, e, f, g, h, i)  cusparseCcsrilu02_bufferSize(a, b, c, d, (cuComplex *)e, f, g, h, i)
    #define cusparseXcsrilu02_analysis(a, b, c, d, e, f, g, h, i, j) cusparseCcsrilu02_analysis(a, b, c, d, (cuComplex *)e, f, g, h, i, j)
    #define cusparseXcsrilu02(a, b, c, d, e, f, g, h, i, j)          cusparseCcsrilu02(a, b, c, d, (cuComplex *)e, f, g, h, i, j)
    #define cusparseXcsric02_bufferSize(a, b, c, d, e, f, g, h, i)   cusparseCcsric02_bufferSize(a, b, c, d, (cuComplex *)e, f, g, h, i)
    #define cusparseXcsric02_analysis(a, b, c, d, e, f, g, h, i, j)  cusparseCcsric02_analysis(a, b, c, d, (cuComplex *)e, f, g, h, i, j)
    #define cusparseXcsric02(a, b, c, d, e, f, g, h, i, j)           cusparseCcsric02(a, b, c, d, (cuComplex *)e, f, g, h, i, j)
  #elif defined(PETSC_USE_REAL_DOUBLE)
const cuDoubleComplex PETSC_CUSPARSE_ONE  = {1.0, 0.0};
const cuDoubleComplex PETSC_CUSPARSE_ZERO = {0.0, 0.0};
    #define cusparseXcsrilu02_bufferSize(a, b, c, d, e, f, g, h, i)  cusparseZcsrilu02_bufferSize(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i)
    #define cusparseXcsrilu02_analysis(a, b, c, d, e, f, g, h, i, j) cusparseZcsrilu02_analysis(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i, j)
    #define cusparseXcsrilu02(a, b, c, d, e, f, g, h, i, j)          cusparseZcsrilu02(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i, j)
    #define cusparseXcsric02_bufferSize(a, b, c, d, e, f, g, h, i)   cusparseZcsric02_bufferSize(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i)
    #define cusparseXcsric02_analysis(a, b, c, d, e, f, g, h, i, j)  cusparseZcsric02_analysis(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i, j)
    #define cusparseXcsric02(a, b, c, d, e, f, g, h, i, j)           cusparseZcsric02(a, b, c, d, (cuDoubleComplex *)e, f, g, h, i, j)
  #endif
#else
const PetscScalar PETSC_CUSPARSE_ONE  = 1.0;
const PetscScalar PETSC_CUSPARSE_ZERO = 0.0;
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusparseXcsrilu02_bufferSize cusparseScsrilu02_bufferSize
    #define cusparseXcsrilu02_analysis   cusparseScsrilu02_analysis
    #define cusparseXcsrilu02            cusparseScsrilu02
    #define cusparseXcsric02_bufferSize  cusparseScsric02_bufferSize
    #define cusparseXcsric02_analysis    cusparseScsric02_analysis
    #define cusparseXcsric02             cusparseScsric02
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define cusparseXcsrilu02_bufferSize cusparseDcsrilu02_bufferSize
    #define cusparseXcsrilu02_analysis   cusparseDcsrilu02_analysis
    #define cusparseXcsrilu02            cusparseDcsrilu02
    #define cusparseXcsric02_bufferSize  cusparseDcsric02_bufferSize
    #define cusparseXcsric02_analysis    cusparseDcsric02_analysis
    #define cusparseXcsric02             cusparseDcsric02
  #endif
#endif

#define csrsvInfo_t              csrsv2Info_t
#define cusparseCreateCsrsvInfo  cusparseCreateCsrsv2Info
#define cusparseDestroyCsrsvInfo cusparseDestroyCsrsv2Info
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusparseXcsrsv_buffsize(a, b, c, d, e, f, g, h, i, j)          cusparseCcsrsv2_bufferSize(a, b, c, d, e, (cuComplex *)(f), g, h, i, j)
    #define cusparseXcsrsv_analysis(a, b, c, d, e, f, g, h, i, j, k)       cusparseCcsrsv2_analysis(a, b, c, d, e, (const cuComplex *)(f), g, h, i, j, k)
    #define cusparseXcsrsv_solve(a, b, c, d, e, f, g, h, i, j, k, l, m, n) cusparseCcsrsv2_solve(a, b, c, d, (const cuComplex *)(e), f, (const cuComplex *)(g), h, i, j, (const cuComplex *)(k), (cuComplex *)(l), m, n)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define cusparseXcsrsv_buffsize(a, b, c, d, e, f, g, h, i, j)          cusparseZcsrsv2_bufferSize(a, b, c, d, e, (cuDoubleComplex *)(f), g, h, i, j)
    #define cusparseXcsrsv_analysis(a, b, c, d, e, f, g, h, i, j, k)       cusparseZcsrsv2_analysis(a, b, c, d, e, (const cuDoubleComplex *)(f), g, h, i, j, k)
    #define cusparseXcsrsv_solve(a, b, c, d, e, f, g, h, i, j, k, l, m, n) cusparseZcsrsv2_solve(a, b, c, d, (const cuDoubleComplex *)(e), f, (const cuDoubleComplex *)(g), h, i, j, (const cuDoubleComplex *)(k), (cuDoubleComplex *)(l), m, n)
  #endif
#else /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusparseXcsrsv_buffsize cusparseScsrsv2_bufferSize
    #define cusparseXcsrsv_analysis cusparseScsrsv2_analysis
    #define cusparseXcsrsv_solve    cusparseScsrsv2_solve
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define cusparseXcsrsv_buffsize cusparseDcsrsv2_bufferSize
    #define cusparseXcsrsv_analysis cusparseDcsrsv2_analysis
    #define cusparseXcsrsv_solve    cusparseDcsrsv2_solve
  #endif
#endif

#define cusparse_csr2csc cusparseCsr2cscEx2
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusparse_scalartype                                                             CUDA_C_32F
    #define cusparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) cusparseCcsrgeam2(a, b, c, (cuComplex *)d, e, f, (cuComplex *)g, h, i, (cuComplex *)j, k, l, (cuComplex *)m, n, o, p, (cuComplex *)q, r, s, t)
    #define cusparse_csr_spgeam_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
      cusparseCcsrgeam2_bufferSizeExt(a, b, c, (cuComplex *)d, e, f, (cuComplex *)g, h, i, (cuComplex *)j, k, l, (cuComplex *)m, n, o, p, (cuComplex *)q, r, s, t)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define cusparse_scalartype CUDA_C_64F
    #define cusparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
      cusparseZcsrgeam2(a, b, c, (cuDoubleComplex *)d, e, f, (cuDoubleComplex *)g, h, i, (cuDoubleComplex *)j, k, l, (cuDoubleComplex *)m, n, o, p, (cuDoubleComplex *)q, r, s, t)
    #define cusparse_csr_spgeam_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
      cusparseZcsrgeam2_bufferSizeExt(a, b, c, (cuDoubleComplex *)d, e, f, (cuDoubleComplex *)g, h, i, (cuDoubleComplex *)j, k, l, (cuDoubleComplex *)m, n, o, p, (cuDoubleComplex *)q, r, s, t)
  #endif
#else /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define cusparse_scalartype            CUDA_R_32F
    #define cusparse_csr_spgeam            cusparseScsrgeam2
    #define cusparse_csr_spgeam_bufferSize cusparseScsrgeam2_bufferSizeExt
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define cusparse_scalartype            CUDA_R_64F
    #define cusparse_csr_spgeam            cusparseDcsrgeam2
    #define cusparse_csr_spgeam_bufferSize cusparseDcsrgeam2_bufferSizeExt
  #endif
#endif

#define THRUSTINTARRAY32 thrust::device_vector<int>
#define THRUSTINTARRAY   thrust::device_vector<PetscInt>
#define THRUSTARRAY      thrust::device_vector<PetscScalar>

/* A CSR matrix nonzero structure */
struct CsrMatrix {
  PetscInt          num_rows;
  PetscInt          num_cols;
  PetscInt          num_entries;
  THRUSTINTARRAY32 *row_offsets;
  THRUSTINTARRAY32 *column_indices;
  THRUSTARRAY      *values;
};

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJCUSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  cusparseMatDescr_t  descr;
  cusparseOperation_t solveOp;
  CsrMatrix          *csrMat;
  int                 solveBufferSize;
  void               *solveBuffer;
  size_t              csr2cscBufferSize; /* to transpose the triangular factor (only used for CUDA >= 11.0) */
  void               *csr2cscBuffer;
  PetscScalar        *AA_h; /* managed host buffer for moving values to the GPU */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and any indices used in a reordering */
struct Mat_SeqAIJCUSPARSETriFactors {
  THRUSTINTARRAY  *rpermIndices; /* indices used for any reordering */
  THRUSTINTARRAY  *cpermIndices; /* indices used for any reordering */
  THRUSTARRAY     *workVector;
  cusparseHandle_t handle; /* a handle to the cusparse library */
  PetscInt         nnz;    /* number of nonzeros ... need this for accurate logging between ICC and ILU */
  cudaDeviceProp   dev_prop;
  PetscBool        init_dev_prop;

  PetscScalar *csrVal, *diag;             // the diagonal D in UtDU of Cholesky
  int         *csrRowPtr32, *csrColIdx32; // i,j of M. cusparseScsrilu02/ic02() etc require 32-bit indices
  PetscInt    *csrRowPtr, *csrColIdx;     // i, j of M on device for CUDA APIs that support 64-bit indices
  PetscScalar *csrVal_h, *diag_h;         // Since LU is done on host, we prepare a factored matrix in regular csr format on host and then copy it to device
  PetscInt    *csrRowPtr_h;               // csrColIdx_h is temporary, so it is not here

  /* Mixed mat descriptor types? yes, different cusparse APIs use different types */
  cusparseMatDescr_t   matDescr_M;
  cusparseSpMatDescr_t spMatDescr_L, spMatDescr_U;
  cusparseSpSVDescr_t  spsvDescr_L, spsvDescr_Lt, spsvDescr_U, spsvDescr_Ut;
  cusparseDnVecDescr_t dnVecDescr_X, dnVecDescr_Y;
  PetscScalar         *X, *Y; /* data array of dnVec X and Y */

  /* Mixed size types? yes, CUDA-11.7.0 declared cusparseDcsrilu02_bufferSizeExt() that returns size_t but did not implement it! */
  int    factBufferSize_M; /* M ~= LU or LLt */
  size_t spsvBufferSize_L, spsvBufferSize_Lt, spsvBufferSize_U, spsvBufferSize_Ut;
  /* cusparse needs various buffers for factorization and solve of L, U, Lt, or Ut.
     So save memory, we share the factorization buffer with one of spsvBuffer_L/U.
  */
  void *factBuffer_M, *spsvBuffer_L, *spsvBuffer_U, *spsvBuffer_Lt, *spsvBuffer_Ut;

  csrilu02Info_t        ilu0Info_M;
  csric02Info_t         ic0Info_M;
  int                   structural_zero, numerical_zero;
  cusparseSolvePolicy_t policy_M;

  /* In MatSolveTranspose() for ILU0, we use the two flags to do on-demand solve */
  PetscBool createdTransposeSpSVDescr;    /* Have we created SpSV descriptors for Lt, Ut? */
  PetscBool updatedTransposeSpSVAnalysis; /* Have we ever updated (done) SpSV analysis for Lt, Ut */
  PetscBool updatedSpSVAnalysis;          /* Have we ever updated (done) SpSV Analysis for L, U? */

  PetscLogDouble numericFactFlops; /* Estimated FLOPs in ILU0/ICC0 numeric factorization */
};

struct Mat_CusparseSpMV {
  PetscBool            initialized;    /* Don't rely on spmvBuffer != NULL to test if the struct is initialized, */
  size_t               spmvBufferSize; /* since I'm not sure if smvBuffer can be NULL even after cusparseSpMV_bufferSize() */
  void                *spmvBuffer;
  cusparseDnVecDescr_t vecXDescr, vecYDescr; /* descriptor for the dense vectors in y=op(A)x */
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJCUSPARSEMultStruct {
  void                *mat;          /* opaque pointer to a matrix. This could be either a cusparseHybMat_t or a CsrMatrix */
  cusparseMatDescr_t   descr;        /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY      *cprowIndices; /* compressed row indices used in the parallel SpMV */
  PetscScalar         *alpha_one;    /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar         *beta_zero;    /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar         *beta_one;     /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
  cusparseSpMatDescr_t matDescr;     /* descriptor for the matrix, used by SpMV and SpMM */
#if PETSC_PKG_CUDA_VERSION_GE(12, 4, 0)
  cusparseSpMatDescr_t matDescr_SpMV[3]; // Use separate MatDescr for opA's, to workaround cusparse bugs after 12.4, see https://github.com/NVIDIA/CUDALibrarySamples/issues/212,
  cusparseSpMatDescr_t matDescr_SpMM[3]; // and known issues https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cusparse-release-12-6
#endif
  Mat_CusparseSpMV cuSpMV[3]; /* different Mat_CusparseSpMV structs for non-transpose, transpose, conj-transpose */
  Mat_SeqAIJCUSPARSEMultStruct() : matDescr(NULL)
  {
    for (int i = 0; i < 3; i++) {
      cuSpMV[i].initialized = PETSC_FALSE;
#if PETSC_PKG_CUDA_VERSION_GE(12, 4, 0)
      matDescr_SpMV[i] = NULL;
      matDescr_SpMM[i] = NULL;
#endif
    }
  }
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Transpose */
struct Mat_SeqAIJCUSPARSE {
  Mat_SeqAIJCUSPARSEMultStruct *mat;               /* pointer to the matrix on the GPU */
  Mat_SeqAIJCUSPARSEMultStruct *matTranspose;      /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                  *workVector;        /* pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  THRUSTINTARRAY32             *rowoffsets_gpu;    /* rowoffsets on GPU in non-compressed-row format. It is used to convert CSR to CSC */
  PetscInt                      nrows;             /* number of rows of the matrix seen by GPU */
  MatCUSPARSEStorageFormat      format;            /* the storage format for the matrix on the device */
  PetscBool                     use_cpu_solve;     /* Use AIJ_Seq (I)LU solve */
  cudaStream_t                  stream;            /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  cusparseHandle_t              handle;            /* a handle to the cusparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState              nonzerostate;      /* track nonzero state to possibly recreate the GPU matrix */
  size_t                        csr2cscBufferSize; /* stuff used to compute the matTranspose above */
  void                         *csr2cscBuffer;     /* This is used as a C struct and is calloc'ed by PetscNew() */
  cusparseCsr2CscAlg_t          csr2cscAlg;        /* algorithms can be selected from command line options */
  cusparseSpMVAlg_t             spmvAlg;
  cusparseSpMMAlg_t             spmmAlg;
  THRUSTINTARRAY               *csr2csc_i;
  THRUSTINTARRAY               *coords; /* permutation array used in MatSeqAIJCUSPARSEMergeMats */
};

typedef struct Mat_SeqAIJCUSPARSETriFactors *Mat_SeqAIJCUSPARSETriFactors_p;

PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSEMergeMats(Mat, Mat, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSETriFactors_Reset(Mat_SeqAIJCUSPARSETriFactors_p *);

using VecSeq_CUDA = Petsc::vec::cupm::impl::VecSeq_CUPM<Petsc::device::cupm::DeviceType::CUDA>;

static inline bool isCudaMem(const void *data)
{
  using namespace Petsc::device::cupm;
  auto mtype = PETSC_MEMTYPE_HOST;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, impl::Interface<DeviceType::CUDA>::PetscCUPMGetMemType(data, &mtype));
  PetscFunctionReturn(PetscMemTypeDevice(mtype));
}
