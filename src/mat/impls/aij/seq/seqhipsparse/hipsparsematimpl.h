/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#ifndef PETSC_HIPSPARSEMATIMPL_H
#define PETSC_HIPSPARSEMATIMPL_H

#include <petscpkg_version.h>
#include <petsc/private/hipvecimpl.h>
#include <petscaijdevice.h>

#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  #include <hipsparse/hipsparse.h>
#else /* PETSC_PKG_HIP_VERSION_GE(5,2,0) */
  #include <hipsparse.h>
#endif /* PETSC_PKG_HIP_VERSION_GE(5,2,0) */
#include "hip/hip_runtime.h"

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/system/system_error.h>

#define PetscCallThrust(body) \
  do { \
    try { \
      body; \
    } catch (thrust::system_error & e) { \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in Thrust %s", e.what()); \
    } \
  } while (0)

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
const hipComplex PETSC_HIPSPARSE_ONE  = {1.0f, 0.0f};
const hipComplex PETSC_HIPSPARSE_ZERO = {0.0f, 0.0f};
    #define hipsparseXcsrilu02_bufferSize(a, b, c, d, e, f, g, h, i)  hipsparseCcsrilu02_bufferSize(a, b, c, d, (hipComplex *)e, f, g, h, i)
    #define hipsparseXcsrilu02_analysis(a, b, c, d, e, f, g, h, i, j) hipsparseCcsrilu02_analysis(a, b, c, d, (hipComplex *)e, f, g, h, i, j)
    #define hipsparseXcsrilu02(a, b, c, d, e, f, g, h, i, j)          hipsparseCcsrilu02(a, b, c, d, (hipComplex *)e, f, g, h, i, j)
    #define hipsparseXcsric02_bufferSize(a, b, c, d, e, f, g, h, i)   hipsparseCcsric02_bufferSize(a, b, c, d, (hipComplex *)e, f, g, h, i)
    #define hipsparseXcsric02_analysis(a, b, c, d, e, f, g, h, i, j)  hipsparseCcsric02_analysis(a, b, c, d, (hipComplex *)e, f, g, h, i, j)
    #define hipsparseXcsric02(a, b, c, d, e, f, g, h, i, j)           hipsparseCcsric02(a, b, c, d, (hipComplex *)e, f, g, h, i, j)
  #elif defined(PETSC_USE_REAL_DOUBLE)
const hipDoubleComplex PETSC_HIPSPARSE_ONE  = {1.0, 0.0};
const hipDoubleComplex PETSC_HIPSPARSE_ZERO = {0.0, 0.0};
    #define hipsparseXcsrilu02_bufferSize(a, b, c, d, e, f, g, h, i)  hipsparseZcsrilu02_bufferSize(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i)
    #define hipsparseXcsrilu02_analysis(a, b, c, d, e, f, g, h, i, j) hipsparseZcsrilu02_analysis(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i, j)
    #define hipsparseXcsrilu02(a, b, c, d, e, f, g, h, i, j)          hipsparseZcsrilu02(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i, j)
    #define hipsparseXcsric02_bufferSize(a, b, c, d, e, f, g, h, i)   hipsparseZcsric02_bufferSize(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i)
    #define hipsparseXcsric02_analysis(a, b, c, d, e, f, g, h, i, j)  hipsparseZcsric02_analysis(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i, j)
    #define hipsparseXcsric02(a, b, c, d, e, f, g, h, i, j)           hipsparseZcsric02(a, b, c, d, (hipDoubleComplex *)e, f, g, h, i, j)
  #endif /* Single or double */
#else    /* not complex */
const PetscScalar PETSC_HIPSPARSE_ONE  = 1.0;
const PetscScalar PETSC_HIPSPARSE_ZERO = 0.0;
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsparseXcsrilu02_bufferSize hipsparseScsrilu02_bufferSize
    #define hipsparseXcsrilu02_analysis   hipsparseScsrilu02_analysis
    #define hipsparseXcsrilu02            hipsparseScsrilu02
    #define hipsparseXcsric02_bufferSize  hipsparseScsric02_bufferSize
    #define hipsparseXcsric02_analysis    hipsparseScsric02_analysis
    #define hipsparseXcsric02             hipsparseScsric02
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define hipsparseXcsrilu02_bufferSize hipsparseDcsrilu02_bufferSize
    #define hipsparseXcsrilu02_analysis   hipsparseDcsrilu02_analysis
    #define hipsparseXcsrilu02            hipsparseDcsrilu02
    #define hipsparseXcsric02_bufferSize  hipsparseDcsric02_bufferSize
    #define hipsparseXcsric02_analysis    hipsparseDcsric02_analysis
    #define hipsparseXcsric02             hipsparseDcsric02
  #endif /* Single or double */
#endif   /* complex or not */

#define csrsvInfo_t               csrsv2Info_t
#define hipsparseCreateCsrsvInfo  hipsparseCreateCsrsv2Info
#define hipsparseDestroyCsrsvInfo hipsparseDestroyCsrsv2Info
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsparseXcsrsv_buffsize(a, b, c, d, e, f, g, h, i, j)          hipsparseCcsrsv2_bufferSize(a, b, c, d, e, (hipComplex *)(f), g, h, i, j)
    #define hipsparseXcsrsv_analysis(a, b, c, d, e, f, g, h, i, j, k)       hipsparseCcsrsv2_analysis(a, b, c, d, e, (const hipComplex *)(f), g, h, i, j, k)
    #define hipsparseXcsrsv_solve(a, b, c, d, e, f, g, h, i, j, k, l, m, n) hipsparseCcsrsv2_solve(a, b, c, d, (const hipComplex *)(e), f, (const hipComplex *)(g), h, i, j, (const hipComplex *)(k), (hipComplex *)(l), m, n)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define hipsparseXcsrsv_buffsize(a, b, c, d, e, f, g, h, i, j)          hipsparseZcsrsv2_bufferSize(a, b, c, d, e, (hipDoubleComplex *)(f), g, h, i, j)
    #define hipsparseXcsrsv_analysis(a, b, c, d, e, f, g, h, i, j, k)       hipsparseZcsrsv2_analysis(a, b, c, d, e, (const hipDoubleComplex *)(f), g, h, i, j, k)
    #define hipsparseXcsrsv_solve(a, b, c, d, e, f, g, h, i, j, k, l, m, n) hipsparseZcsrsv2_solve(a, b, c, d, (const hipDoubleComplex *)(e), f, (const hipDoubleComplex *)(g), h, i, j, (const hipDoubleComplex *)(k), (hipDoubleComplex *)(l), m, n)
  #endif /* Single or double */
#else    /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsparseXcsrsv_buffsize hipsparseScsrsv2_bufferSize
    #define hipsparseXcsrsv_analysis hipsparseScsrsv2_analysis
    #define hipsparseXcsrsv_solve    hipsparseScsrsv2_solve
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define hipsparseXcsrsv_buffsize hipsparseDcsrsv2_bufferSize
    #define hipsparseXcsrsv_analysis hipsparseDcsrsv2_analysis
    #define hipsparseXcsrsv_solve    hipsparseDcsrsv2_solve
  #endif /* Single or double */
#endif   /* not complex */

#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
  // #define cusparse_csr2csc cusparseCsr2cscEx2
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipsparse_scalartype                                                             HIP_C_32F
      #define hipsparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) hipsparseCcsrgeam2(a, b, c, (hipComplex *)d, e, f, (hipComplex *)g, h, i, (hipComplex *)j, k, l, (hipComplex *)m, n, o, p, (hipComplex *)q, r, s, t)
      #define hipsparse_csr_spgeam_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
        hipsparseCcsrgeam2_bufferSizeExt(a, b, c, (hipComplex *)d, e, f, (hipComplex *)g, h, i, (hipComplex *)j, k, l, (hipComplex *)m, n, o, p, (hipComplex *)q, r, s, t)
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define hipsparse_scalartype HIP_C_64F
      #define hipsparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
        hipsparseZcsrgeam2(a, b, c, (hipDoubleComplex *)d, e, f, (hipDoubleComplex *)g, h, i, (hipDoubleComplex *)j, k, l, (hipDoubleComplex *)m, n, o, p, (hipDoubleComplex *)q, r, s, t)
      #define hipsparse_csr_spgeam_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) \
        hipsparseZcsrgeam2_bufferSizeExt(a, b, c, (hipDoubleComplex *)d, e, f, (hipDoubleComplex *)g, h, i, (hipDoubleComplex *)j, k, l, (hipDoubleComplex *)m, n, o, p, (hipDoubleComplex *)q, r, s, t)
    #endif /* Single or double */
  #else    /* not complex */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipsparse_scalartype            HIP_R_32F
      #define hipsparse_csr_spgeam            hipsparseScsrgeam2
      #define hipsparse_csr_spgeam_bufferSize hipsparseScsrgeam2_bufferSizeExt
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define hipsparse_scalartype            HIP_R_64F
      #define hipsparse_csr_spgeam            hipsparseDcsrgeam2
      #define hipsparse_csr_spgeam_bufferSize hipsparseDcsrgeam2_bufferSizeExt
    #endif /* Single or double */
  #endif   /* complex or not */
#endif     /* PETSC_PKG_HIP_VERSION_GE(4, 5, 0) */

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsparse_scalartype                                                             HIP_C_32F
    #define hipsparse_csr_spmv(a, b, c, d, e, f, g, h, i, j, k, l, m)                        hipsparseCcsrmv((a), (b), (c), (d), (e), (hipComplex *)(f), (g), (hipComplex *)(h), (i), (j), (hipComplex *)(k), (hipComplex *)(l), (hipComplex *)(m))
    #define hipsparse_csr_spmm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)               hipsparseCcsrmm((a), (b), (c), (d), (e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (j), (k), (hipComplex *)(l), (m), (hipComplex *)(n), (hipComplex *)(o), (p))
    #define hipsparse_csr2csc(a, b, c, d, e, f, g, h, i, j, k, l)                            hipsparseCcsr2csc((a), (b), (c), (d), (hipComplex *)(e), (f), (g), (hipComplex *)(h), (i), (j), (k), (l))
    #define hipsparse_hyb_spmv(a, b, c, d, e, f, g, h)                                       hipsparseChybmv((a), (b), (hipComplex *)(c), (d), (e), (hipComplex *)(f), (hipComplex *)(g), (hipComplex *)(h))
    #define hipsparse_csr2hyb(a, b, c, d, e, f, g, h, i, j)                                  hipsparseCcsr2hyb((a), (b), (c), (d), (hipComplex *)(e), (f), (g), (h), (i), (j))
    #define hipsparse_hyb2csr(a, b, c, d, e, f)                                              hipsparseChyb2csr((a), (b), (c), (hipComplex *)(d), (e), (f))
    #define hipsparse_csr_spgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) hipsparseCcsrgemm(a, b, c, d, e, f, g, h, (hipComplex *)i, j, k, l, m, (hipComplex *)n, o, p, q, (hipComplex *)r, s, t)
  // #define hipsparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)    hipsparseCcsrgeam(a, b, c, (hipComplex *)d, e, f, (hipComplex *)g, h, i, (hipComplex *)j, k, l, (hipComplex *)m, n, o, p, (hipComplex *)q, r, s)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define hipsparse_scalartype                                      HIP_C_64F
    #define hipsparse_csr_spmv(a, b, c, d, e, f, g, h, i, j, k, l, m) hipsparseZcsrmv((a), (b), (c), (d), (e), (hipDoubleComplex *)(f), (g), (hipDoubleComplex *)(h), (i), (j), (hipDoubleComplex *)(k), (hipDoubleComplex *)(l), (hipDoubleComplex *)(m))
    #define hipsparse_csr_spmm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) \
      hipsparseZcsrmm((a), (b), (c), (d), (e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (j), (k), (hipDoubleComplex *)(l), (m), (hipDoubleComplex *)(n), (hipDoubleComplex *)(o), (p))
    #define hipsparse_csr2csc(a, b, c, d, e, f, g, h, i, j, k, l)                            hipsparseZcsr2csc((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (g), (hipDoubleComplex *)(h), (i), (j), (k), (l))
    #define hipsparse_hyb_spmv(a, b, c, d, e, f, g, h)                                       hipsparseZhybmv((a), (b), (hipDoubleComplex *)(c), (d), (e), (hipDoubleComplex *)(f), (hipDoubleComplex *)(g), (hipDoubleComplex *)(h))
    #define hipsparse_csr2hyb(a, b, c, d, e, f, g, h, i, j)                                  hipsparseZcsr2hyb((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (g), (h), (i), (j))
    #define hipsparse_hyb2csr(a, b, c, d, e, f)                                              hipsparseZhyb2csr((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
    #define hipsparse_csr_spgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t) hipsparseZcsrgemm(a, b, c, d, e, f, g, h, (hipDoubleComplex *)i, j, k, l, m, (hipDoubleComplex *)n, o, p, q, (hipDoubleComplex *)r, s, t)
  // #define hipsparse_csr_spgeam(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)    hipsparseZcsrgeam(a, b, c, (hipDoubleComplex *)d, e, f, (hipDoubleComplex *)g, h, i, (hipDoubleComplex *)j, k, l, (hipDoubleComplex *)m, n, o, p, (hipDoubleComplex *)q, r, s)
  #endif /* Single or double */
#else    /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define hipsparse_scalartype HIP_R_32F
    #define hipsparse_csr_spmv   hipsparseScsrmv
    #define hipsparse_csr_spmm   hipsparseScsrmm
    #define hipsparse_csr2csc    hipsparseScsr2csc
    #define hipsparse_hyb_spmv   hipsparseShybmv
    #define hipsparse_csr2hyb    hipsparseScsr2hyb
    #define hipsparse_hyb2csr    hipsparseShyb2csr
    #define hipsparse_csr_spgemm hipsparseScsrgemm
  // #define hipsparse_csr_spgeam hipsparseScsrgeam
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define hipsparse_scalartype HIP_R_64F
    #define hipsparse_csr_spmv   hipsparseDcsrmv
    #define hipsparse_csr_spmm   hipsparseDcsrmm
    #define hipsparse_csr2csc    hipsparseDcsr2csc
    #define hipsparse_hyb_spmv   hipsparseDhybmv
    #define hipsparse_csr2hyb    hipsparseDcsr2hyb
    #define hipsparse_hyb2csr    hipsparseDhyb2csr
    #define hipsparse_csr_spgemm hipsparseDcsrgemm
  // #define hipsparse_csr_spgeam hipsparseDcsrgeam
  #endif /* Single or double */
#endif   /* complex or not */

#define THRUSTINTARRAY32 thrust::device_vector<int>
#define THRUSTINTARRAY   thrust::device_vector<PetscInt>
#define THRUSTARRAY      thrust::device_vector<PetscScalar>

/* A CSR matrix structure */
struct CsrMatrix {
  PetscInt          num_rows;
  PetscInt          num_cols;
  PetscInt          num_entries;
  THRUSTINTARRAY32 *row_offsets;
  THRUSTINTARRAY32 *column_indices;
  THRUSTARRAY      *values;
};

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJHIPSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  hipsparseMatDescr_t    descr;
  hipsparseOperation_t   solveOp;
  CsrMatrix             *csrMat;
  csrsvInfo_t            solveInfo;
  hipsparseSolvePolicy_t solvePolicy; /* whether level information is generated and used */
  int                    solveBufferSize;
  void                  *solveBuffer;
  size_t                 csr2cscBufferSize; /* to transpose the triangular factor (only used for CUDA >= 11.0) */
  void                  *csr2cscBuffer;
  PetscScalar           *AA_h; /* managed host buffer for moving values to the GPU */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and any indices used in a reordering */
struct Mat_SeqAIJHIPSPARSETriFactors {
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtr;          /* pointer for lower triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtr;          /* pointer for upper triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtrTranspose; /* pointer for lower triangular (factored matrix) on GPU for the transpose (useful for BiCG) */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtrTranspose; /* pointer for upper triangular (factored matrix) on GPU for the transpose (useful for BiCG)*/
  THRUSTINTARRAY                     *rpermIndices;            /* indices used for any reordering */
  THRUSTINTARRAY                     *cpermIndices;            /* indices used for any reordering */
  THRUSTARRAY                        *workVector;
  hipsparseHandle_t                   handle;   /* a handle to the hipsparse library */
  PetscInt                            nnz;      /* number of nonzeros ... need this for accurate logging between ICC and ILU */
  PetscScalar                        *a_band_d; /* GPU data for banded CSR LU factorization matrix diag(L)=1 */
  int                                *i_band_d; /* this could be optimized away */
  hipDeviceProp_t                     dev_prop;
  PetscBool                           init_dev_prop;

  /* csrilu0/csric0 appeared in earlier versions of AMD ROCm^{TM}, but we use it along with hipsparseSpSV,
     which first appeared in hipsparse with ROCm-4.5.0.
  */
  PetscBool factorizeOnDevice; /* Do factorization on device or not */
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
  PetscScalar *csrVal;
  int         *csrRowPtr, *csrColIdx; /* a,i,j of M. Using int since some hipsparse APIs only support 32-bit indices */

  /* Mixed mat descriptor types? yes, different hipsparse APIs use different types */
  hipsparseMatDescr_t   matDescr_M;
  hipsparseSpMatDescr_t spMatDescr_L, spMatDescr_U;
  hipsparseSpSVDescr_t  spsvDescr_L, spsvDescr_Lt, spsvDescr_U, spsvDescr_Ut;

  hipsparseDnVecDescr_t dnVecDescr_X, dnVecDescr_Y;
  PetscScalar          *X, *Y; /* data array of dnVec X and Y */

  /* Mixed size types? yes */
  int    factBufferSize_M; /* M ~= LU or LLt */
  size_t spsvBufferSize_L, spsvBufferSize_Lt, spsvBufferSize_U, spsvBufferSize_Ut;
  /* hipsparse needs various buffers for factorization and solve of L, U, Lt, or Ut.
     To save memory, we share the factorization buffer with one of spsvBuffer_L/U.
  */
  void *factBuffer_M, *spsvBuffer_L, *spsvBuffer_U, *spsvBuffer_Lt, *spsvBuffer_Ut;

  csrilu02Info_t         ilu0Info_M;
  csric02Info_t          ic0Info_M;
  int                    structural_zero, numerical_zero;
  hipsparseSolvePolicy_t policy_M;

  /* In MatSolveTranspose() for ILU0, we use the two flags to do on-demand solve */
  PetscBool createdTransposeSpSVDescr;    /* Have we created SpSV descriptors for Lt, Ut? */
  PetscBool updatedTransposeSpSVAnalysis; /* Have we updated SpSV analysis with the latest L, U values? */

  PetscLogDouble numericFactFlops; /* Estimated FLOPs in ILU0/ICC0 numeric factorization */
#endif
};

struct Mat_HipsparseSpMV {
  PetscBool             initialized;    /* Don't rely on spmvBuffer != NULL to test if the struct is initialized, */
  size_t                spmvBufferSize; /* since I'm not sure if smvBuffer can be NULL even after hipsparseSpMV_bufferSize() */
  void                 *spmvBuffer;
  hipsparseDnVecDescr_t vecXDescr, vecYDescr; /* descriptor for the dense vectors in y=op(A)x */
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJHIPSPARSEMultStruct {
  void                 *mat;          /* opaque pointer to a matrix. This could be either a hipsparseHybMat_t or a CsrMatrix */
  hipsparseMatDescr_t   descr;        /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY       *cprowIndices; /* compressed row indices used in the parallel SpMV */
  PetscScalar          *alpha_one;    /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar          *beta_zero;    /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar          *beta_one;     /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
  hipsparseSpMatDescr_t matDescr;     /* descriptor for the matrix, used by SpMV and SpMM */
  Mat_HipsparseSpMV     hipSpMV[3];   /* different Mat_CusparseSpMV structs for non-transpose, transpose, conj-transpose */
  Mat_SeqAIJHIPSPARSEMultStruct() : matDescr(NULL)
  {
    for (int i = 0; i < 3; i++) hipSpMV[i].initialized = PETSC_FALSE;
  }
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Transpose */
struct Mat_SeqAIJHIPSPARSE {
  Mat_SeqAIJHIPSPARSEMultStruct *mat;               /* pointer to the matrix on the GPU */
  Mat_SeqAIJHIPSPARSEMultStruct *matTranspose;      /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                   *workVector;        /* pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  THRUSTINTARRAY32              *rowoffsets_gpu;    /* rowoffsets on GPU in non-compressed-row format. It is used to convert CSR to CSC */
  PetscInt                       nrows;             /* number of rows of the matrix seen by GPU */
  MatHIPSPARSEStorageFormat      format;            /* the storage format for the matrix on the device */
  PetscBool                      use_cpu_solve;     /* Use AIJ_Seq (I)LU solve */
  hipStream_t                    stream;            /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  hipsparseHandle_t              handle;            /* a handle to the cusparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState               nonzerostate;      /* track nonzero state to possibly recreate the GPU matrix */
  size_t                         csr2cscBufferSize; /* stuff used to compute the matTranspose above */
  void                          *csr2cscBuffer;     /* This is used as a C struct and is calloc'ed by PetscNewLog() */
                                                    //  hipsparseCsr2CscAlg_t         csr2cscAlg; /* algorithms can be selected from command line options */
  hipsparseSpMVAlg_t         spmvAlg;
  hipsparseSpMMAlg_t         spmmAlg;
  THRUSTINTARRAY            *csr2csc_i;
  PetscSplitCSRDataStructure deviceMat; /* Matrix on device for, eg, assembly */
  THRUSTINTARRAY            *cooPerm;   /* permutation array that sorts the input coo entris by row and col */
  THRUSTINTARRAY            *cooPerm_a; /* ordered array that indicate i-th nonzero (after sorting) is the j-th unique nonzero */

  /* Stuff for extended COO support */
  PetscBool   use_extended_coo; /* Use extended COO format */
  PetscCount *jmap_d;           /* perm[disp+jmap[i]..disp+jmap[i+1]) gives indices of entries in v[] associated with i-th nonzero of the matrix */
  PetscCount *perm_d;

  Mat_SeqAIJHIPSPARSE() : use_extended_coo(PETSC_FALSE), perm_d(NULL), jmap_d(NULL) { }
};

typedef struct Mat_SeqAIJHIPSPARSETriFactors *Mat_SeqAIJHIPSPARSETriFactors_p;

PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE_Basic(Mat, PetscCount, PetscInt[], PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE_Basic(Mat, const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEMergeMats(Mat, Mat, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Reset(Mat_SeqAIJHIPSPARSETriFactors_p *);

static inline bool isHipMem(const void *data)
{
  hipError_t                   cerr;
  struct hipPointerAttribute_t attr;
  enum hipMemoryType           mtype;
  cerr = hipPointerGetAttributes(&attr, data); /* Do not check error since before CUDA 11.0, passing a host pointer returns hipErrorInvalidValue */
  hipGetLastError();                           /* Reset the last error */
  mtype = attr.memoryType;
  if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) return true;
  else return false;
}

#endif // PETSC_HIPSPARSEIMPL_H
