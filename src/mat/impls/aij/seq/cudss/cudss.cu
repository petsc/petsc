/*
     This file implements a subclass of the SeqAIJ matrix class that uses
     the NVIDIA cuDSS GPU-accelerated sparse direct solver library.
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>
#include <cudss.h>
#include <nvrtc.h>
#include <climits>

// Internal data structure for the cuDSS factored matrix
typedef struct {
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t   data;
  cudssMatrix_t cudss_A; // sparse matrix descriptor
  cudssMatrix_t cudss_b; // dense RHS descriptor
  cudssMatrix_t cudss_x; // dense solution descriptor

  // Device CSR arrays (owned by us when input is MATSEQAIJ)
  PetscInt    *d_row_offsets;
  PetscInt    *d_col_indices;
  int         *d_row_i32; // narrowed 32-bit copy when PetscInt is 64-bit
  int         *d_col_i32;
  PetscScalar *d_values;

  // User-supplied permutation (host array of 32-bit ints, owned by us)
  int      *h_user_perm;
  PetscBool userPermSet;

  PetscInt  n; // matrix dimension
  PetscInt  nnz;
  PetscBool ownDeviceCSR;

  cudssReorderingAlg_t reorderAlg;
  cudssPivotType_t     pivotType;
  double               pivotThreshold; // cuDSS expects double for CUDSS_CONFIG_PIVOT_THRESHOLD
  double               pivotEpsilon;   // cuDSS expects double for CUDSS_CONFIG_PIVOT_EPSILON
  int                  useMatching;    // cuDSS expects int, translated to CUDSS_CONFIG_MATCHING_ALG
  int                  irNSteps;       // cuDSS expects int for CUDSS_CONFIG_IR_N_STEPS
  int                  deterministic;  // cuDSS expects int for CUDSS_CONFIG_DETERMINISTIC_MODE
  PetscBool            mempool;        // register a stream-ordered cudaMemPool_t as cuDSS's device workspace allocator
} Mat_cuDSS;

// Map PetscScalar to the cuDSS data type
#if PetscDefined(USE_COMPLEX)
  #if PetscDefined(USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDSS_C_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDSS_C_64F
  #endif
#else
  #if PetscDefined(USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDSS_R_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDSS_R_64F
  #endif
#endif

#define PetscCallCUDSS(...) \
  do { \
    cudssStatus_t cudss_status_ = __VA_ARGS__; \
    PetscCheck(cudss_status_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)cudss_status_, PetscStringize(__VA_ARGS__)); \
  } while (0)

static PetscErrorCode MatCUDSSGetContext(PetscDeviceContext *dctx, cudaStream_t *stream)
{
  cudaStream_t *handle;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetStreamHandle(*dctx, (void **)&handle));
  *stream = *handle;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int MatCUDSSPoolAlloc(void *ctx, void **ptr, size_t size, cudaStream_t stream)
{
  cudaError_t cerr;

  (void)ctx;
  *ptr = NULL;
  if (size == 0) return 0;
  cerr = cudaMallocAsync(ptr, size, stream);
  return (cerr == cudaSuccess) ? 0 : (int)cerr;
}

static int MatCUDSSPoolFree(void *ctx, void *ptr, size_t size, cudaStream_t stream)
{
  cudaError_t cerr;

  (void)ctx;
  (void)size;
  if (ptr == NULL) return 0;
  cerr = cudaFreeAsync(ptr, stream);
  return (cerr == cudaSuccess) ? 0 : (int)cerr;
}

static PetscErrorCode MatCUDSSSetMemHandler(cudssHandle_t handle, PetscBool use_mempool)
{
  PetscFunctionBegin;
  if (use_mempool == PETSC_TRUE) {
    cudssDeviceMemHandler_t handler;

    PetscCall(PetscMemzero(&handler, sizeof(handler)));
    handler.ctx          = NULL;
    handler.device_alloc = MatCUDSSPoolAlloc;
    handler.device_free  = MatCUDSSPoolFree;
    PetscCall(PetscStrncpy(handler.name, "petsc_cudss_pool", sizeof(handler.name)));
    PetscCallCUDSS(cudssSetDeviceMemHandler(handle, &handler));
  } else PetscCallCUDSS(cudssSetDeviceMemHandler(handle, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* These arrays are indexed by the cuDSS enum values directly.
   Mapping (cuDSS 0.8.0):
     cudssReorderingAlg_t: CUDSS_REORDERING_ALG_DEFAULT==0, ..._BTF_COLAMD==1, ..._COLAMD==2,
                           ..._AMD==3, ..._NESTED_DISSECTION==4, ..._NONE==5
     cudssPivotType_t: CUDSS_PIVOT_AUTO==0, ..._NONE==1, ..._GLOBAL_COL==2, ..._GLOBAL_ROW==3,
                       ..._DIAGONAL==4, ..._LOCAL_BLOCK==5, ..._BUNCH_KAUFMAN==6
   If NVIDIA renumbers these enumerators in a future release, the option parsing
   and view output will silently mismap. Update both arrays and this comment. */
static const char *const MatCUDSSReorderAlgs[] = {"default", "btf_colamd", "colamd", "amd", "nested_dissection", "none"};
static const char *const MatCUDSSPivotTypes[]  = {"auto", "none", "col", "row", "diagonal", "local_block", "bunch_kaufman"};

// Forward declarations
static PetscErrorCode MatFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSetFromOptions_cuDSS(Mat);
static PetscErrorCode MatFactorSymbolic_cuDSS(Mat, Mat, IS, cudssMatrixType_t, cudssMatrixViewType_t);
static PetscErrorCode MatSetPermutation_cuDSS(Mat, IS);

static PetscErrorCode MatView_Info_cuDSS(Mat A, PetscViewer viewer)
{
  Mat_cuDSS  *lu          = (Mat_cuDSS *)A->data;
  const char *reorderName = ((int)lu->reorderAlg >= 0 && (size_t)lu->reorderAlg < PETSC_STATIC_ARRAY_LENGTH(MatCUDSSReorderAlgs)) ? MatCUDSSReorderAlgs[lu->reorderAlg] : "unknown";
  const char *pivotName   = ((int)lu->pivotType >= 0 && (size_t)lu->pivotType < PETSC_STATIC_ARRAY_LENGTH(MatCUDSSPivotTypes)) ? MatCUDSSPivotTypes[lu->pivotType] : "unknown";

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "cuDSS run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Reorder algorithm: %s\n", reorderName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  User permutation: %s\n", lu->userPermSet == PETSC_TRUE ? "yes" : "no"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot type: %s\n", pivotName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot threshold: %g\n", lu->pivotThreshold));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot epsilon: %g\n", lu->pivotEpsilon));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Use matching: %s\n", lu->useMatching ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  IR steps: %d\n", lu->irNSteps));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Deterministic mode: %s\n", lu->deterministic ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Device workspace mempool: %s\n", lu->mempool == PETSC_TRUE ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_cuDSS(Mat A, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii == PETSC_TRUE) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_cuDSS(A, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_cuDSS(Mat_cuDSS *lu)
{
  PetscFunctionBegin;
  if (lu->cudss_x != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_x));
    lu->cudss_x = NULL;
  }
  if (lu->cudss_b != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_b));
    lu->cudss_b = NULL;
  }
  if (lu->cudss_A != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_A));
    lu->cudss_A = NULL;
  }
  if (lu->data != NULL) {
    PetscCallCUDSS(cudssDataDestroy(lu->handle, lu->data));
    lu->data = NULL;
  }
  if (lu->ownDeviceCSR == PETSC_TRUE) {
    PetscCallCUDA(cudaFree(lu->d_row_offsets));
    lu->d_row_offsets = NULL;
    PetscCallCUDA(cudaFree(lu->d_col_indices));
    lu->d_col_indices = NULL;
    PetscCallCUDA(cudaFree(lu->d_values));
    lu->d_values     = NULL;
    lu->ownDeviceCSR = PETSC_FALSE;
  }
  PetscCallCUDA(cudaFree(lu->d_row_i32));
  lu->d_row_i32 = NULL;
  PetscCallCUDA(cudaFree(lu->d_col_i32));
  lu->d_col_i32 = NULL;
  PetscCall(PetscFree(lu->h_user_perm));
  lu->userPermSet = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_cuDSS(Mat A)
{
  Mat_cuDSS *lu = (Mat_cuDSS *)A->data;

  PetscFunctionBegin;
  PetscCall(MatReset_cuDSS(lu));
  if (lu->config != NULL) PetscCallCUDSS(cudssConfigDestroy(lu->config));
  if (lu->handle != NULL) PetscCallCUDSS(cudssDestroy(lu->handle));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_cuDSS(Mat F)
{
  Mat_cuDSS *lu         = (Mat_cuDSS *)F->data;
  PetscInt   reorderAlg = (PetscInt)lu->reorderAlg, pivotType = (PetscInt)lu->pivotType, irNSteps = (PetscInt)lu->irNSteps;
  PetscReal  pivotThreshold = (PetscReal)lu->pivotThreshold, pivotEpsilon = (PetscReal)lu->pivotEpsilon;
  PetscBool  useMatching = (PetscBool)lu->useMatching, deterministic = (PetscBool)lu->deterministic;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "cuDSS Options", "Mat");
  PetscCall(PetscOptionsEList("-mat_cudss_reorder_alg", "Reordering algorithm", "None", MatCUDSSReorderAlgs, PETSC_STATIC_ARRAY_LENGTH(MatCUDSSReorderAlgs), MatCUDSSReorderAlgs[reorderAlg], &reorderAlg, NULL));
  lu->reorderAlg = (cudssReorderingAlg_t)reorderAlg;
  PetscCall(PetscOptionsEList("-mat_cudss_pivot_type", "Pivot type", "None", MatCUDSSPivotTypes, PETSC_STATIC_ARRAY_LENGTH(MatCUDSSPivotTypes), MatCUDSSPivotTypes[pivotType], &pivotType, NULL));
  lu->pivotType = (cudssPivotType_t)pivotType;
  PetscCall(PetscOptionsReal("-mat_cudss_pivot_threshold", "Pivot threshold", "None", pivotThreshold, &pivotThreshold, NULL));
  lu->pivotThreshold = (double)pivotThreshold;
  PetscCall(PetscOptionsReal("-mat_cudss_pivot_epsilon", "Pivot epsilon", "None", pivotEpsilon, &pivotEpsilon, NULL));
  lu->pivotEpsilon = (double)pivotEpsilon;
  PetscCall(PetscOptionsBool("-mat_cudss_use_matching", "Enable matching", "None", useMatching, &useMatching, NULL));
  lu->useMatching = (int)useMatching;
  PetscCall(PetscOptionsInt("-mat_cudss_ir_n_steps", "Number of iterative refinement steps", "None", irNSteps, &irNSteps, NULL));
  PetscCall(PetscCIntCast(irNSteps, &lu->irNSteps));
  PetscCall(PetscOptionsBool("-mat_cudss_deterministic", "Enable bitwise-reproducible factorization and solve", "None", deterministic, &deterministic, NULL));
  lu->deterministic = (int)deterministic;
  PetscCall(PetscOptionsBool("-mat_cudss_mempool", "Use a stream-ordered cudaMemPool_t for cuDSS device workspace allocations", "None", lu->mempool, &lu->mempool, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatApplyConfig_cuDSS(Mat_cuDSS *lu)
{
  cudssMatchingAlg_t matchingAlg = lu->useMatching ? CUDSS_MATCHING_ALG_AUTO : CUDSS_MATCHING_ALG_NONE;

  PetscFunctionBegin;
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_REORDERING_ALG, &lu->reorderAlg, sizeof(lu->reorderAlg)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_TYPE, &lu->pivotType, sizeof(lu->pivotType)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_THRESHOLD, &lu->pivotThreshold, sizeof(lu->pivotThreshold)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_EPSILON, &lu->pivotEpsilon, sizeof(lu->pivotEpsilon)));
  // Matching is controlled via CUDSS_CONFIG_MATCHING_ALG (CUDSS_MATCHING_ALG_NONE disables it)
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_MATCHING_ALG, &matchingAlg, sizeof(matchingAlg)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_IR_N_STEPS, &lu->irNSteps, sizeof(lu->irNSteps)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_DETERMINISTIC_MODE, &lu->deterministic, sizeof(lu->deterministic)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(USE_64BIT_INDICES)
__global__ static void MatNarrowIndices_Kernel(PetscInt n, const PetscInt *src, int *dst)
{
  for (PetscInt tid = (PetscInt)blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += (PetscInt)gridDim.x * blockDim.x) dst[tid] = (int)src[tid];
}
#endif

/* Return the device CSR arrays of A in d_row_32/d_col_32 (MATSEQAIJCUSPARSE with 32-bit PetscInt) or d_row_pi/d_col_pi, and d_val.
   Leave the unused index pair unchanged. Reuse the arrays of a MATSEQAIJCUSPARSE matrix not bound to the CPU, else copy into buffers of lu.
   With valuesOnly, copy only the values because the structure is unchanged since the symbolic phase. */
static PetscErrorCode MatEnsureOnDevice_cuDSS(Mat A, Mat_cuDSS *lu, PetscBool valuesOnly, int **d_row_32, int **d_col_32, PetscInt **d_row_pi, PetscInt **d_col_pi, PetscScalar **d_val)
{
  PetscFunctionBegin;
  if (lu->ownDeviceCSR == PETSC_FALSE) {
    Mat_SeqAIJCUSPARSE           *cusparsestruct;
    Mat_SeqAIJCUSPARSEMultStruct *matstruct;
    CsrMatrix                    *csr;

    PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    cusparsestruct = (Mat_SeqAIJCUSPARSE *)A->spptr;
    matstruct      = (Mat_SeqAIJCUSPARSEMultStruct *)cusparsestruct->mat;
    csr            = (CsrMatrix *)matstruct->mat;
#if PetscDefined(USE_64BIT_INDICES)
    *d_row_pi = thrust::raw_pointer_cast(csr->row_offsets->data());
    *d_col_pi = thrust::raw_pointer_cast(csr->column_indices->data());
#else
    *d_row_32 = thrust::raw_pointer_cast(csr->row_offsets->data());
    *d_col_32 = thrust::raw_pointer_cast(csr->column_indices->data());
#endif
    *d_val = (PetscScalar *)thrust::raw_pointer_cast(csr->values->data());
  } else {
    Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data;
    const PetscScalar *aa;
    PetscDeviceContext dctx;
    cudaStream_t       stream;

    PetscCall(MatCUDSSGetContext(&dctx, &stream));
    if (valuesOnly == PETSC_FALSE) {
      // Symbolic phase: upload structure (row offsets + column indices)
      PetscCallCUDA(cudaMemcpyAsync(lu->d_row_offsets, a->i, (lu->n + 1) * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
      PetscCallCUDA(cudaMemcpyAsync(lu->d_col_indices, a->j, lu->nnz * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
      PetscCall(PetscLogCpuToGpu((lu->n + 1) * sizeof(PetscInt) + lu->nnz * sizeof(PetscInt)));
    }
    // MatSeqAIJGetArrayRead(), unlike a->a, syncs device values of derived types to the host
    PetscCall(MatSeqAIJGetArrayRead(A, &aa));
    PetscCallCUDA(cudaMemcpyAsync(lu->d_values, aa, lu->nnz * sizeof(PetscScalar), cudaMemcpyHostToDevice, stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
    PetscCall(PetscLogCpuToGpu(lu->nnz * sizeof(PetscScalar)));
    *d_row_pi = lu->d_row_offsets;
    *d_col_pi = lu->d_col_indices;
    *d_val    = lu->d_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, cudssMatrixType_t mtype, cudssMatrixViewType_t mview)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ *)A->data;
  PetscInt           m = A->rmap->n, n = A->cmap->n, nnz = a->nz;
  PetscInt          *d_row_pi = NULL, *d_col_pi = NULL; // PetscInt-width path
  int               *d_row_32 = NULL, *d_col_32 = NULL; // MATSEQAIJCUSPARSE path with 32-bit PetscInt
  void              *d_row, *d_col;
  PetscScalar       *d_val = NULL;
  PetscBool          isCUSPARSE;
  PetscDeviceContext dctx;
  cudaStream_t       stream;

  PetscFunctionBegin;
  PetscCall(MatReset_cuDSS(lu));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "cuDSS requires a square matrix, got %" PetscInt_FMT "x%" PetscInt_FMT, m, n);
  if (m == 0) PetscFunctionReturn(PETSC_SUCCESS);

  lu->n   = n;
  lu->nnz = nnz;

  if (lu->handle == NULL) PetscCallCUDSS(cudssCreate(&lu->handle));
  if (lu->config == NULL) PetscCallCUDSS(cudssConfigCreate(&lu->config));
  PetscCall(MatSetFromOptions_cuDSS(F));
  PetscCall(MatCUDSSSetMemHandler(lu->handle, lu->mempool));
  PetscCallCUDSS(cudssDataCreate(lu->handle, &lu->data));
  PetscCall(MatApplyConfig_cuDSS(lu));
  if (perm != NULL) PetscCall(MatSetPermutation_cuDSS(F, perm));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(isCUSPARSE == PETSC_FALSE || A->boundtocpu == PETSC_TRUE);
  if (lu->ownDeviceCSR == PETSC_TRUE) {
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  PetscCall(MatCUDSSGetContext(&dctx, &stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));

  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, PETSC_FALSE, &d_row_32, &d_col_32, &d_row_pi, &d_col_pi, &d_val));
#if PetscDefined(USE_64BIT_INDICES)
  // Narrow the 64-bit PetscInt CSR index arrays to 32-bit on the device
  PetscCheck(nnz <= INT_MAX && m < INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "cuDSS requires 32-bit CSR indices; matrix has m=%" PetscInt_FMT ", nnz=%" PetscInt_FMT " > INT_MAX", m, nnz);
  PetscCallCUDA(cudaMalloc((void **)&lu->d_row_i32, (m + 1) * sizeof(int)));
  PetscCallCUDA(cudaMalloc((void **)&lu->d_col_i32, nnz * sizeof(int)));
  PetscCall(Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>::PetscCUPMLaunchKernel1D(m + 1, 0, stream, MatNarrowIndices_Kernel, m + 1, d_row_pi, lu->d_row_i32));
  if (nnz > 0) PetscCall(Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>::PetscCUPMLaunchKernel1D(nnz, 0, stream, MatNarrowIndices_Kernel, nnz, d_col_pi, lu->d_col_i32));
  d_row = lu->d_row_i32;
  d_col = lu->d_col_i32;
#else
  d_row = d_row_32 != NULL ? (void *)d_row_32 : (void *)d_row_pi;
  d_col = d_col_32 != NULL ? (void *)d_col_32 : (void *)d_col_pi;
#endif
  PetscCallCUDSS(cudssMatrixCreateCsr(&lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, CUDSS_R_32I, CUDSS_R_32I, CUDSS_SCALAR_TYPE, mtype, mview, CUDSS_BASE_ZERO));

  PetscCallCUDSS(cudssMatrixCreateDn(&lu->cudss_b, n, 1, n, NULL, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
  PetscCallCUDSS(cudssMatrixCreateDn(&lu->cudss_x, n, 1, n, NULL, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
  // cuDSS needs the user permutation before the analysis phase
  if (lu->userPermSet == PETSC_TRUE) {
    PetscCallCUDSS(cudssDataSet(lu->handle, lu->data, CUDSS_DATA_USER_PERM, lu->h_user_perm, (size_t)n * sizeof(int)));
    PetscCall(PetscInfo(F, "cuDSS: using user-supplied permutation of size %" PetscInt_FMT "\n", n));
  }
  // Analysis phase: reordering and symbolic factorization
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
  PetscCall(PetscDeviceContextSynchronizeIfWithBarrier_Internal(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_cuDSS(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (r != NULL && c != NULL) PetscCall(PetscInfo(F, "cuDSS accepts only a single permutation; using row ordering IS, column ordering IS is ignored\n"));
  PetscCall(MatFactorSymbolic_cuDSS(F, A, r != NULL ? r : c, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL));
  F->ops->lufactornumeric = MatFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, const MatFactorInfo *info)
{
  cudssMatrixType_t mtype;

  PetscFunctionBegin;
#if PetscDefined(USE_COMPLEX)
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HPD;
  else if (A->hermitian == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HERMITIAN;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#else
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_SPD;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#endif
  PetscCall(MatFactorSymbolic_cuDSS(F, A, perm, mtype, CUDSS_MVIEW_UPPER));
  F->ops->choleskyfactornumeric = MatFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_cuDSS(Mat F, Vec b, Vec x)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscMemType       bmtype, xmtype;
  Vec                bcu = NULL, xcu = NULL;
  PetscDeviceContext dctx;
  cudaStream_t       stream;

  PetscFunctionBegin;
  if (F->rmap->n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  /* Accept any vector whose array is device-resident (VECCUDA, VECKOKKOS on a CUDA
     backend, etc.) directly without a temporary Vec or device-to-device copy. */
  PetscCall(VecGetArrayReadAndMemType(b, &barray, &bmtype));
  if (!PetscMemTypeDevice(bmtype)) {
    PetscCall(VecRestoreArrayReadAndMemType(b, &barray));
    PetscCall(PetscInfo(F, "Copying b from host to temporary VECCUDA; consider using a device Vec for better performance\n"));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)b), &bcu));
    PetscCall(VecSetSizes(bcu, b->map->n, b->map->N));
    PetscCall(VecSetType(bcu, VECCUDA));
    PetscCall(VecCopy(b, bcu));
    PetscCall(VecCUDAGetArrayRead(bcu, &barray));
  }
  PetscCall(VecGetArrayAndMemType(x, &xarray, &xmtype));
  if (!PetscMemTypeDevice(xmtype)) {
    PetscCall(VecRestoreArrayAndMemType(x, &xarray));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)x), &xcu));
    PetscCall(VecSetSizes(xcu, x->map->n, x->map->N));
    PetscCall(VecSetType(xcu, VECCUDA));
    PetscCall(VecCUDAGetArrayWrite(xcu, &xarray));
  }
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_b, (void *)barray));
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_x, xarray));
  PetscCall(MatCUDSSGetContext(&dctx, &stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
  PetscCall(PetscDeviceContextSynchronizeIfWithBarrier_Internal(dctx));
  if (bcu != NULL) {
    PetscCall(VecCUDARestoreArrayRead(bcu, &barray));
    PetscCall(VecDestroy(&bcu));
  } else PetscCall(VecRestoreArrayReadAndMemType(b, &barray));
  if (xcu != NULL) {
    PetscCall(VecCUDARestoreArrayWrite(xcu, &xarray));
    PetscCall(VecCopy(xcu, x));
    PetscCall(VecDestroy(&xcu));
  } else PetscCall(VecRestoreArrayAndMemType(x, &xarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_cuDSS(Mat F, Mat B, Mat X)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscInt           n, nrhs, nX, nrhsX, ldb, ldx, ncol;
  cudssMatrix_t      cudss_B = NULL, cudss_X = NULL;
  Mat                Bcuda = NULL, Xcuda = NULL;
  PetscBool          BisCUDA = PETSC_FALSE, XisCUDA = PETSC_FALSE;
  PetscDeviceContext dctx;
  cudaStream_t       stream;

  PetscFunctionBegin;
  if (F->rmap->n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &BisCUDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSECUDA, &XisCUDA));
  if (BisCUDA == PETSC_FALSE) {
    PetscCall(PetscInfo(F, "Converting B from host to MATDENSECUDA; consider using MATSEQDENSECUDA for better performance\n"));
    PetscCall(MatConvert(B, MATDENSECUDA, MAT_INITIAL_MATRIX, &Bcuda));
  } else if (B == X) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &Bcuda)); // solve out of place from a copy of B
  else Bcuda = B;
  if (XisCUDA == PETSC_FALSE) {
    PetscInt m_x, n_x, M_x, N_x;

    PetscCall(MatGetLocalSize(X, &m_x, &n_x));
    PetscCall(MatGetSize(X, &M_x, &N_x));
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)X), m_x, n_x, M_x, N_x, NULL, &Xcuda));
  } else Xcuda = X;
  PetscCall(MatGetSize(Bcuda, &n, &nrhs));
  PetscCall(MatGetSize(Xcuda, &nX, &nrhsX));
  PetscCheck(n == nX && nrhs == nrhsX, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible matrix dimensions: B is %" PetscInt_FMT " x %" PetscInt_FMT ", X is %" PetscInt_FMT " x %" PetscInt_FMT, n, nrhs, nX, nrhsX);
  PetscCheck(n == lu->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "B row count %" PetscInt_FMT " does not match factored matrix size %" PetscInt_FMT, n, lu->n);
  PetscCall(MatDenseGetLDA(Bcuda, &ldb));
  PetscCall(MatDenseGetLDA(Xcuda, &ldx));
  PetscCall(MatDenseCUDAGetArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDAGetArrayWrite(Xcuda, &xarray));
  PetscCall(MatCUDSSGetContext(&dctx, &stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  // cuDSS deterministic mode requires nrhs == 1, so solve column-by-column when enabled
  ncol = lu->deterministic ? 1 : nrhs;
  for (PetscInt j = 0; j < nrhs; j += ncol) {
    PetscCallCUDSS(cudssMatrixCreateDn(&cudss_B, n, ncol, ldb, (void *)(barray + j * ldb), CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    PetscCallCUDSS(cudssMatrixCreateDn(&cudss_X, n, ncol, ldx, xarray + j * ldx, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, cudss_X, cudss_B));
    PetscCallCUDSS(cudssMatrixDestroy(cudss_B));
    PetscCallCUDSS(cudssMatrixDestroy(cudss_X));
  }
  PetscCall(PetscDeviceContextSynchronizeIfWithBarrier_Internal(dctx));
  PetscCall(MatDenseCUDARestoreArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDARestoreArrayWrite(Xcuda, &xarray));
  if (Bcuda != B) PetscCall(MatDestroy(&Bcuda));
  if (XisCUDA == PETSC_FALSE) {
    PetscCall(MatCopy(Xcuda, X, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Xcuda));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS         *lu       = (Mat_cuDSS *)F->data;
  int               *d_row_32 = NULL, *d_col_32 = NULL;
  int                cudss_info = 0;
  PetscInt          *d_row_pi = NULL, *d_col_pi = NULL;
  PetscScalar       *d_val = NULL;
  size_t             nwritten;
  PetscDeviceContext dctx;
  cudaStream_t       stream;

  PetscFunctionBegin;
  if (F->rmap->n == 0) {
    F->assembled = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, PETSC_TRUE, &d_row_32, &d_col_32, &d_row_pi, &d_col_pi, &d_val));
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_A, d_val));
  PetscCall(MatCUDSSGetContext(&dctx, &stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
  PetscCall(PetscDeviceContextSynchronize(dctx)); // CUDSS_DATA_INFO is valid only after the factorization completes
  PetscCallCUDSS(cudssDataGet(lu->handle, lu->data, CUDSS_DATA_INFO, &cudss_info, sizeof(cudss_info), &nwritten));
  if (cudss_info != 0) {
    PetscCall(PetscInfo(F, "cuDSS factorization failed with CUDSS_DATA_INFO %d\n", cudss_info));
    F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }
  F->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_cudss(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPermutation_cuDSS(Mat F, IS perm)
{
  Mat_cuDSS      *lu = (Mat_cuDSS *)F->data;
  PetscInt        nrows, ncols, n;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscCall(PetscFree(lu->h_user_perm));
  PetscCall(MatGetLocalSize(F, &nrows, &ncols));
  PetscCheck(nrows <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "cuDSS requires 32-bit permutation indices; matrix has m=%" PetscInt_FMT " > INT_MAX", nrows);
  PetscCall(ISGetLocalSize(perm, &n));
  PetscCheck(n == nrows && n == ncols, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible matrix and permutation sizes");
  PetscCall(PetscMalloc1(n, &lu->h_user_perm));
  PetscCall(ISGetIndices(perm, &idx));
  for (PetscInt i = 0; i < n; i++) lu->h_user_perm[i] = (int)idx[i];
  PetscCall(ISRestoreIndices(perm, &idx));
  lu->userPermSet = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERCUDSS = "cudss" - A solver package providing LU and Cholesky factorization for
  sequential sparse matrices via the NVIDIA cuDSS GPU-accelerated sparse direct solver library.

  Options Database Keys:
+ -mat_cudss_reorder_alg (default|btf_colamd|colamd|amd|nested_dissection|none) - reordering algorithm
. -mat_cudss_pivot_type (auto|none|col|row|diagonal|local_block|bunch_kaufman)  - pivoting type
. -mat_cudss_pivot_threshold threshold                                          - Pivot threshold, default is 1.0
. -mat_cudss_pivot_epsilon epsilon                                              - Pivot epsilon, default is 0.0
. -mat_cudss_use_matching (true|false)                                          - Enable bipartite matching for heavy-diagonal scaling and permutation, default is false
. -mat_cudss_ir_n_steps nsteps                                                  - Number of iterative refinement steps, default is 0
. -mat_cudss_deterministic (true|false)                                         - Enable bitwise-reproducible factorization and solve, default is false
- -mat_cudss_mempool (true|false)                                               - Use a stream-ordered `cudaMemPool_t` for cuDSS device workspace allocations, default is true

  Level: intermediate

  Notes:
    Registered for both `MATSEQAIJ` (host) and `MATSEQAIJCUSPARSE` (device) matrix types.
    When the input matrix is `MATSEQAIJ`, the matrix data is transparently copied to the GPU.
    `MatSolveTranspose()` is not supported. The `bunch_kaufman` value of `-mat_cudss_pivot_type`
    applies to symmetric or Hermitian indefinite matrices.

    By default cuDSS performs its own internal reordering (`MATORDERINGEXTERNAL`) during the symbolic phase,
    selectable via `-mat_cudss_reorder_alg default`. A PETSc ordering can be selected with
    `-pc_factor_mat_ordering_type` or `PCFactorSetMatOrderingType()`, or by passing a non-`NULL` `IS`
    to `MatLUFactorSymbolic()` or `MatCholeskyFactorSymbolic()`.

    `MatSolve()` and `MatMatSolve()` operate on the GPU. Host right-hand-side and solution
    objects (`VECSEQ` vectors, `MATSEQDENSE` matrices) are accepted and transparently copied
    to and from the device, but this incurs extra copies; for best performance supply
    device vectors (e.g., `VECCUDA` or `VECKOKKOS`) and `MATSEQDENSECUDA` matrices.

.seealso: [](ch_matrices), `Mat`, `PCLU`, `PCCHOLESKY`, `PCFactorSetMatSolverType()`, `PCFactorSetMatOrderingType()`, `MatSolverType`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`
M*/

static PetscErrorCode MatGetFactor_seqaij_cudss(Mat A, MatFactorType ftype, Mat *F)
{
  Mat        B;
  Mat_cuDSS *lu;
  PetscInt   m = A->rmap->n, n = A->cmap->n;
  int        nvrtc_major, nvrtc_minor;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported by cuDSS");
  // Call nvrtcVersion() so --as-needed linkers record a direct libnvrtc dependency on libpetsc
  PetscCheck(nvrtcVersion(&nvrtc_major, &nvrtc_minor) == NVRTC_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "nvrtcVersion() failed");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  B->trivialsymbolic = PETSC_FALSE;
  B->factortype      = ftype;
  B->canuseordering  = PETSC_TRUE;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_LU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_cuDSS;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_cuDSS;
  B->ops->solve    = MatSolve_cuDSS;
  B->ops->matsolve = MatMatSolve_cuDSS;
  B->ops->destroy  = MatDestroy_cuDSS;
  B->ops->view     = MatView_cuDSS;
  B->ops->getinfo  = MatGetInfo_External;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &B->solvertype));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_cudss));

  PetscCall(PetscNew(&lu));
  lu->reorderAlg     = CUDSS_REORDERING_ALG_DEFAULT;
  lu->pivotType      = CUDSS_PIVOT_AUTO;
  lu->pivotThreshold = 1.0;
  lu->pivotEpsilon   = 0.0;
  lu->useMatching    = 0;
  lu->irNSteps       = 0;
  lu->deterministic  = 0;
  lu->mempool        = PETSC_TRUE;
  B->data            = lu;

  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_cuDSS(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJCUSPARSE, MAT_FACTOR_LU, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJCUSPARSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_cudss));
  PetscFunctionReturn(PETSC_SUCCESS);
}
