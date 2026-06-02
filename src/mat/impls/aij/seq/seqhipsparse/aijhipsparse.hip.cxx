/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the HIPSPARSE library,
  Portions of this code are under:
  Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/dense/seq/dense.h> // MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Internal()
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>
#include <../src/mat/impls/aij/seq/cupm/aijcupm.hpp>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/transform_iterator.h>
#if PETSC_CPP_VERSION >= 14
  #define PETSC_HAVE_THRUST_ASYNC 1
  #include <thrust/async/for_each.h>
#endif
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>

const char *const MatHIPSPARSEStorageFormats[] = {"CSR", "ELL", "HYB", "MatHIPSPARSEStorageFormat", "MAT_HIPSPARSE_", 0};
const char *const MatHIPSPARSESpMVAlgorithms[] = {"MV_ALG_DEFAULT", "COOMV_ALG", "CSRMV_ALG1", "CSRMV_ALG2", "SPMV_ALG_DEFAULT", "SPMV_COO_ALG1", "SPMV_COO_ALG2", "SPMV_CSR_ALG1", "SPMV_CSR_ALG2", "hipsparseSpMVAlg_t", "HIPSPARSE_", 0};
const char *const MatHIPSPARSESpMMAlgorithms[] = {"ALG_DEFAULT", "COO_ALG1", "COO_ALG2", "COO_ALG3", "CSR_ALG1", "COO_ALG4", "CSR_ALG2", "hipsparseSpMMAlg_t", "HIPSPARSE_SPMM_", 0};
//const char *const MatHIPSPARSECsr2CscAlgorithms[] = {"INVALID"/*HIPSPARSE does not have enum 0! We created one*/, "ALG1", "ALG2", "hipsparseCsr2CscAlg_t", "HIPSPARSE_CSR2CSC_", 0};

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, IS, const MatFactorInfo *);
static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(Mat, PetscOptionItems PetscOptionsObject);
static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat, PetscScalar, Mat, MatStructure);
static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat, PetscScalar);
static PetscErrorCode MatDiagonalScale_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultHermitianTranspose_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultAddKernel_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec, PetscBool, PetscBool);

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix **);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct **, MatHIPSPARSEStorageFormat);
static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors **);
static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat);

static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat);
static PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat, PetscBool);

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJHIPSPARSE(Mat, PetscInt, const PetscInt[], PetscScalar[]);
static PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat, PetscCount, PetscInt[], PetscInt[]);
static PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat, const PetscScalar[], InsertMode);

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat, MatType, MatReuse, Mat *);

// cusparseCreateCsr() separates types for row offsets and column indices in prototype, but requires them to have the same type at runtime!
const hipsparseIndexType_t csrRowOffsetsType = PetscDefined(USE_64BIT_INDICES) ? HIPSPARSE_INDEX_64I : HIPSPARSE_INDEX_32I;
const hipsparseIndexType_t csrColIndType     = PetscDefined(USE_64BIT_INDICES) ? HIPSPARSE_INDEX_64I : HIPSPARSE_INDEX_32I;

using Csr2coo        = Petsc::mat::aij::cupm::impl::Csr2coo;
using PetscIntToCInt = Petsc::mat::aij::cupm::impl::PetscIntToCInt;

/*
PetscErrorCode MatHIPSPARSESetStream(Mat A, const hipStream_t stream)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  PetscCheck(hipsparsestruct, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing spptr");
  hipsparsestruct->stream = stream;
  PetscCallHIPSPARSE(hipsparseSetStream(hipsparsestruct->handle, hipsparsestruct->stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatHIPSPARSESetHandle(Mat A, const hipsparseHandle_t handle)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  PetscCheck(hipsparsestruct, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing spptr");
  if (hipsparsestruct->handle != handle) {
    if (hipsparsestruct->handle) PetscCallHIPSPARSE(hipsparseDestroy(hipsparsestruct->handle));
    hipsparsestruct->handle = handle;
  }
  PetscCallHIPSPARSE(hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatHIPSPARSEClearHandle(Mat A)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  PetscBool            flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
  if (!flg || !hipsparsestruct) PetscFunctionReturn(PETSC_SUCCESS);
  if (hipsparsestruct->handle) hipsparsestruct->handle = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
*/

PETSC_INTERN PetscErrorCode MatHIPSPARSESetFormat_SeqAIJHIPSPARSE(Mat A, MatHIPSPARSEFormatOperation op, MatHIPSPARSEStorageFormat format)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT:
    hipsparsestruct->format = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparsestruct->format = format;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported operation %d for MatHIPSPARSEFormatOperation. MAT_HIPSPARSE_MULT and MAT_HIPSPARSE_ALL are currently supported.", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHIPSPARSESetFormat - Sets the storage format of `MATSEQHIPSPARSE` matrices for a particular
  operation. Only the `MatMult()` operation can use different GPU storage formats

  Not Collective

  Input Parameters:
+ A      - Matrix of type `MATSEQAIJHIPSPARSE`
. op     - `MatHIPSPARSEFormatOperation`. `MATSEQAIJHIPSPARSE` matrices support `MAT_HIPSPARSE_MULT` and `MAT_HIPSPARSE_ALL`.
         `MATMPIAIJHIPSPARSE` matrices support `MAT_HIPSPARSE_MULT_DIAG`, `MAT_HIPSPARSE_MULT_OFFDIAG`, and `MAT_HIPSPARSE_ALL`.
- format - `MatHIPSPARSEStorageFormat` (one of `MAT_HIPSPARSE_CSR`, `MAT_HIPSPARSE_ELL`, `MAT_HIPSPARSE_HYB`.)

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATSEQAIJHIPSPARSE`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
@*/
PetscErrorCode MatHIPSPARSESetFormat(Mat A, MatHIPSPARSEFormatOperation op, MatHIPSPARSEStorageFormat format)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscTryMethod(A, "MatHIPSPARSESetFormat_C", (Mat, MatHIPSPARSEFormatOperation, MatHIPSPARSEStorageFormat), (A, op, format));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatHIPSPARSESetUseCPUSolve_SeqAIJHIPSPARSE(Mat A, PetscBool use_cpu)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;

  PetscFunctionBegin;
  hipsparsestruct->use_cpu_solve = use_cpu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHIPSPARSESetUseCPUSolve - Sets use CPU `MatSolve()`.

  Input Parameters:
+ A       - Matrix of type `MATSEQAIJHIPSPARSE`
- use_cpu - set flag for using the built-in CPU `MatSolve()`

  Level: intermediate

  Notes:
  The hipSparse LU solver currently computes the factors with the built-in CPU method
  and moves the factors to the GPU for the solve. We have observed better performance keeping the data on the CPU and computing the solve there.
  This method to specifies if the solve is done on the CPU or GPU (GPU is the default).

.seealso: [](ch_matrices), `Mat`, `MatSolve()`, `MATSEQAIJHIPSPARSE`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
@*/
PetscErrorCode MatHIPSPARSESetUseCPUSolve(Mat A, PetscBool use_cpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscTryMethod(A, "MatHIPSPARSESetUseCPUSolve_C", (Mat, PetscBool), (A, use_cpu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOption_SeqAIJHIPSPARSE(Mat A, MatOption op, PetscBool flg)
{
  PetscFunctionBegin;
  switch (op) {
  case MAT_FORM_EXPLICIT_TRANSPOSE:
    /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
    if (A->form_explicit_transpose && !flg) PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_TRUE));
    A->form_explicit_transpose = flg;
    break;
  default:
    PetscCall(MatSetOption_SeqAIJ(A, op, flg));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(Mat A, PetscOptionItems PetscOptionsObject)
{
  MatHIPSPARSEStorageFormat format;
  PetscBool                 flg;
  Mat_SeqAIJHIPSPARSE      *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SeqAIJHIPSPARSE options");
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_hipsparse_mult_storage_format", "sets storage format of (seq)aijhipsparse gpu matrices for SpMV", "MatHIPSPARSESetFormat", MatHIPSPARSEStorageFormats, (PetscEnum)hipsparsestruct->format, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_MULT, format));
    PetscCall(PetscOptionsEnum("-mat_hipsparse_storage_format", "sets storage format of (seq)aijhipsparse gpu matrices for SpMV and TriSolve", "MatHIPSPARSESetFormat", MatHIPSPARSEStorageFormats, (PetscEnum)hipsparsestruct->format, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_ALL, format));
    PetscCall(PetscOptionsBool("-mat_hipsparse_use_cpu_solve", "Use CPU (I)LU solve", "MatHIPSPARSESetUseCPUSolve", hipsparsestruct->use_cpu_solve, &hipsparsestruct->use_cpu_solve, &flg));
    if (flg) PetscCall(MatHIPSPARSESetUseCPUSolve(A, hipsparsestruct->use_cpu_solve));
    PetscCall(
      PetscOptionsEnum("-mat_hipsparse_spmv_alg", "sets hipSPARSE algorithm used in sparse-mat dense-vector multiplication (SpMV)", "hipsparseSpMVAlg_t", MatHIPSPARSESpMVAlgorithms, (PetscEnum)hipsparsestruct->spmvAlg, (PetscEnum *)&hipsparsestruct->spmvAlg, &flg));
    /* If user did use this option, check its consistency with hipSPARSE, since PetscOptionsEnum() sets enum values based on their position in MatHIPSPARSESpMVAlgorithms[] */
    PetscCheck(!flg || HIPSPARSE_CSRMV_ALG1 == 2, PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSPARSE enum hipsparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");
    PetscCall(
      PetscOptionsEnum("-mat_hipsparse_spmm_alg", "sets hipSPARSE algorithm used in sparse-mat dense-mat multiplication (SpMM)", "hipsparseSpMMAlg_t", MatHIPSPARSESpMMAlgorithms, (PetscEnum)hipsparsestruct->spmmAlg, (PetscEnum *)&hipsparsestruct->spmmAlg, &flg));
    PetscCheck(!flg || HIPSPARSE_SPMM_CSR_ALG1 == 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSPARSE enum hipsparseSpMMAlg_t has been changed but PETSc has not been updated accordingly");
    /*
    PetscCall(PetscOptionsEnum("-mat_hipsparse_csr2csc_alg", "sets hipSPARSE algorithm used in converting CSR matrices to CSC matrices", "hipsparseCsr2CscAlg_t", MatHIPSPARSECsr2CscAlgorithms, (PetscEnum)hipsparsestruct->csr2cscAlg, (PetscEnum*)&hipsparsestruct->csr2cscAlg, &flg));
    PetscCheck(!flg || HIPSPARSE_CSR2CSC_ALG1 == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSPARSE enum hipsparseCsr2CscAlg_t has been changed but PETSc has not been updated accordingly");
    */
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildFactoredMatrix_LU(Mat A)
{
  Mat_SeqAIJ                    *a  = static_cast<Mat_SeqAIJ *>(A->data);
  PetscInt                       m  = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors *fs = static_cast<Mat_SeqAIJHIPSPARSETriFactors *>(A->spptr);
  const PetscInt                *Ai = a->i, *Aj = a->j, *adiag;
  const MatScalar               *Aa = a->a;
  PetscInt                      *Mi, *Mj, Mnz;
  PetscScalar                   *Ma;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, &adiag, NULL));
  if (A->offloadmask == PETSC_OFFLOAD_CPU) { // A's latest factors are on CPU
    if (!fs->csrRowPtr) {                    // Is this the first time we are doing setup? Use csrRowPtr since it is not null even when m=0
      // Re-arrange the (skewed) factored matrix and put the result into M, a regular csr matrix on host
      Mnz = (Ai[m] - Ai[0]) + (adiag[0] - adiag[m]); // Lnz (without the unit diagonal) + Unz (with the non-unit diagonal)
      PetscCall(PetscMalloc1(m + 1, &Mi));
      PetscCall(PetscMalloc1(Mnz, &Mj)); // Mj is temp
      PetscCall(PetscMalloc1(Mnz, &Ma));
      Mi[0] = 0;
      for (PetscInt i = 0; i < m; i++) {
        PetscInt llen = Ai[i + 1] - Ai[i];
        PetscInt ulen = adiag[i] - adiag[i + 1];
        PetscCall(PetscArraycpy(Mj + Mi[i], Aj + Ai[i], llen));                           // entries of L
        Mj[Mi[i] + llen] = i;                                                             // diagonal entry
        PetscCall(PetscArraycpy(Mj + Mi[i] + llen + 1, Aj + adiag[i + 1] + 1, ulen - 1)); // entries of U on the right of the diagonal
        Mi[i + 1] = Mi[i] + llen + ulen;
      }
      // Copy M (L,U) from host to device
      PetscCallHIP(hipMalloc(&fs->csrRowPtr, sizeof(*fs->csrRowPtr) * (m + 1)));
      PetscCallHIP(hipMalloc(&fs->csrColIdx, sizeof(*fs->csrColIdx) * Mnz));
      PetscCallHIP(hipMalloc(&fs->csrVal, sizeof(*fs->csrVal) * Mnz));
      PetscCallHIP(hipMemcpy(fs->csrRowPtr, Mi, sizeof(*fs->csrRowPtr) * (m + 1), hipMemcpyHostToDevice));
      PetscCallHIP(hipMemcpy(fs->csrColIdx, Mj, sizeof(*fs->csrColIdx) * Mnz, hipMemcpyHostToDevice));

      // Create descriptors for L, U. See https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDiagType_t
      // cusparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
      // assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
      // all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
      // assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
      hipsparseFillMode_t fillMode = HIPSPARSE_FILL_MODE_LOWER;
      hipsparseDiagType_t diagType = HIPSPARSE_DIAG_TYPE_UNIT;

      PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_L, m, m, Mnz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

      fillMode = HIPSPARSE_FILL_MODE_UPPER;
      diagType = HIPSPARSE_DIAG_TYPE_NON_UNIT;
      PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_U, m, m, Mnz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

      // Allocate work vectors in SpSv
      PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(*fs->X) * m));
      PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(*fs->Y) * m));

      PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
      PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

      // Query buffer sizes for SpSV and then allocate buffers, temporarily assuming opA = HIPSPARSE_OPERATION_NON_TRANSPOSE
      PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_L));
      PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, &fs->spsvBufferSize_L));
      PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_U));
      PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, &fs->spsvBufferSize_U));
      PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_U, fs->spsvBufferSize_U));
      PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_L, fs->spsvBufferSize_L));

      // Record for reuse
      fs->csrRowPtr_h = Mi;
      fs->csrVal_h    = Ma;
      PetscCall(PetscFree(Mj));
    }
    // Copy the value
    Mi  = fs->csrRowPtr_h;
    Ma  = fs->csrVal_h;
    Mnz = Mi[m];
    for (PetscInt i = 0; i < m; i++) {
      PetscInt llen = Ai[i + 1] - Ai[i];
      PetscInt ulen = adiag[i] - adiag[i + 1];
      PetscCall(PetscArraycpy(Ma + Mi[i], Aa + Ai[i], llen));                           // entries of L
      Ma[Mi[i] + llen] = (MatScalar)1.0 / Aa[adiag[i]];                                 // recover the diagonal entry
      PetscCall(PetscArraycpy(Ma + Mi[i] + llen + 1, Aa + adiag[i + 1] + 1, ulen - 1)); // entries of U on the right of the diagonal
    }
    PetscCallHIP(hipMemcpy(fs->csrVal, Ma, sizeof(*Ma) * Mnz, hipMemcpyHostToDevice));

    {
      // Do hipsparseSpSV_analysis(), which is numeric and requires valid and up-to-date matrix values
      PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

      PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, fs->spsvBuffer_U));
      fs->updatedSpSVAnalysis          = PETSC_TRUE;
      fs->updatedTransposeSpSVAnalysis = PETSC_FALSE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  Mat_SeqAIJ                    *a                   = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  IS                             isrow = a->row, isicol = a->icol;
  PetscBool                      row_identity, col_identity;
  PetscInt                       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");
  PetscCall(MatSeqAIJHIPSPARSEBuildFactoredMatrix_LU(A));

  hipsparseTriFactors->nnz = a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH; // factored matrix is sync'ed to GPU
  /* lower triangular indices */
  PetscCall(ISIdentity(isrow, &row_identity));
  if (!row_identity && !hipsparseTriFactors->rpermIndices) {
    const PetscInt *r;

    PetscCall(ISGetIndices(isrow, &r));
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(r, r + n);
    PetscCall(ISRestoreIndices(isrow, &r));
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
  }

  /* upper triangular indices */
  PetscCall(ISIdentity(isicol, &col_identity));
  if (!col_identity && !hipsparseTriFactors->cpermIndices) {
    const PetscInt *c;

    PetscCall(ISGetIndices(isicol, &c));
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(c, c + n);
    PetscCall(ISRestoreIndices(isicol, &c));
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildFactoredMatrix_Cholesky(Mat A)
{
  Mat_SeqAIJ                    *a  = static_cast<Mat_SeqAIJ *>(A->data);
  PetscInt                       m  = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors *fs = static_cast<Mat_SeqAIJHIPSPARSETriFactors *>(A->spptr);
  const PetscInt                *Ai = a->i, *Aj = a->j, *adiag;
  const MatScalar               *Aa = a->a;
  PetscInt                      *Mj, Mnz;
  PetscScalar                   *Ma, *D;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, &adiag, NULL));
  if (A->offloadmask == PETSC_OFFLOAD_CPU) { // A's latest factors are on CPU
    if (!fs->csrRowPtr) {                    // Is this the first time we are doing setup? Use csrRowPtr since it is not null even m=0
      // Re-arrange the (skewed) factored matrix and put the result into M, a regular csr matrix on host.
      // See comments at MatICCFactorSymbolic_SeqAIJ() on the layout of the factored matrix (U) on host.
      Mnz = Ai[m]; // Unz (with the unit diagonal)
      PetscCall(PetscMalloc1(Mnz, &Ma));
      PetscCall(PetscMalloc1(Mnz, &Mj)); // Mj[] is temp
      PetscCall(PetscMalloc1(m, &D));    // the diagonal
      for (PetscInt i = 0; i < m; i++) {
        PetscInt ulen = Ai[i + 1] - Ai[i];
        Mj[Ai[i]]     = i;                                              // diagonal entry
        PetscCall(PetscArraycpy(Mj + Ai[i] + 1, Aj + Ai[i], ulen - 1)); // entries of U on the right of the diagonal
      }
      // Copy M (U) from host to device
      PetscCallHIP(hipMalloc(&fs->csrRowPtr, sizeof(*fs->csrRowPtr) * (m + 1)));
      PetscCallHIP(hipMalloc(&fs->csrColIdx, sizeof(*fs->csrColIdx) * Mnz));
      PetscCallHIP(hipMalloc(&fs->csrVal, sizeof(*fs->csrVal) * Mnz));
      PetscCallHIP(hipMalloc(&fs->diag, sizeof(*fs->diag) * m));
      PetscCallHIP(hipMemcpy(fs->csrRowPtr, Ai, sizeof(*Ai) * (m + 1), hipMemcpyHostToDevice));
      PetscCallHIP(hipMemcpy(fs->csrColIdx, Mj, sizeof(*Mj) * Mnz, hipMemcpyHostToDevice));

      // Create descriptors for L, U. See https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDiagType_t
      // cusparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
      // assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
      // all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
      // assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
      hipsparseFillMode_t fillMode = HIPSPARSE_FILL_MODE_UPPER;
      hipsparseDiagType_t diagType = HIPSPARSE_DIAG_TYPE_UNIT; // U is unit diagonal

      PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_U, m, m, Mnz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
      PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

      // Allocate work vectors in SpSv
      PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(*fs->X) * m));
      PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(*fs->Y) * m));

      PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
      PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

      // Query buffer sizes for SpSV and then allocate buffers
      PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_U));
      PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, &fs->spsvBufferSize_U));
      PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_U, fs->spsvBufferSize_U));

      PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Ut)); // Ut solve uses the same matrix (spMatDescr_U), but different descr and buffer
      PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Ut, &fs->spsvBufferSize_Ut));
      PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Ut, fs->spsvBufferSize_Ut));

      // Record for reuse
      fs->csrVal_h = Ma;
      fs->diag_h   = D;
      PetscCall(PetscFree(Mj));
    }
    // Copy the value
    Ma  = fs->csrVal_h;
    D   = fs->diag_h;
    Mnz = Ai[m];
    for (PetscInt i = 0; i < m; i++) {
      D[i]      = Aa[adiag[i]];   // actually Aa[adiag[i]] is the inverse of the diagonal
      Ma[Ai[i]] = (MatScalar)1.0; // set the unit diagonal, which is cosmetic since cusparse does not really read it given CUSPARSE_DIAG_TYPE_UNIT
      for (PetscInt k = 0; k < Ai[i + 1] - Ai[i] - 1; k++) Ma[Ai[i] + 1 + k] = -Aa[Ai[i] + k];
    }
    PetscCallHIP(hipMemcpy(fs->csrVal, Ma, sizeof(*Ma) * Mnz, hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(fs->diag, D, sizeof(*D) * m, hipMemcpyHostToDevice));

    {
      // Do hipsparseSpSV_analysis(), which is numeric and requires valid and up-to-date matrix values
      PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, fs->spsvBuffer_U));
      PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));
      fs->updatedSpSVAnalysis = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve Ut D U x = b
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_Cholesky(Mat A, Vec b, Vec x)
{
  Mat_SeqAIJHIPSPARSETriFactors        *fs  = static_cast<Mat_SeqAIJHIPSPARSETriFactors *>(A->spptr);
  Mat_SeqAIJ                           *aij = static_cast<Mat_SeqAIJ *>(A->data);
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  const hipsparseSpSVAlg_t              alg = HIPSPARSE_SPSV_ALG_DEFAULT;
  PetscInt                              m   = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecHIPGetArrayWrite(x, &xarray));
  PetscCall(VecHIPGetArrayRead(b, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  // Reorder b with the row permutation if needed, and wrap the result in fs->X
  if (fs->rpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->end()), thrust::device_pointer_cast(fs->X)));

  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->rpermIndices ? fs->X : (void *)barray));

  // Solve Ut Y = X
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_Y, fs->Y));
#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));
#endif

  // Solve diag(D) Z = Y. Actually just do Y = Y*D since D is already inverted in MatCholeskyFactorNumeric_SeqAIJ().
  // It is basically a vector element-wise multiplication, but cublas does not have it!
  auto multiplies = thrust::multiplies<PetscScalar>();
  PetscCallThrust(thrust::transform(thrust::hip::par.on(PetscDefaultHipStream), thrust::device_pointer_cast(fs->Y), thrust::device_pointer_cast(fs->Y + m), thrust::device_pointer_cast(fs->diag), thrust::device_pointer_cast(fs->Y), multiplies));

  // Solve U X = Y
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->cpermIndices ? fs->X : xarray)); // if need to permute, we need to use the intermediate buffer X

#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_U));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_U, fs->spsvBuffer_U));
#endif

  // Reorder X with the column permutation if needed, and put the result back to x
  if (fs->cpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X), fs->cpermIndices->begin()),
                                 thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X + m), fs->cpermIndices->end()), xGPU));

  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(4.0 * aij->nz - A->rmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  Mat_SeqAIJ                    *a                   = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  IS                             ip                  = a->row;
  PetscBool                      perm_identity;
  PetscInt                       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");

  PetscCall(MatSeqAIJHIPSPARSEBuildFactoredMatrix_Cholesky(A));
  hipsparseTriFactors->nnz = (a->nz - n) * 2 + n;

  A->offloadmask = PETSC_OFFLOAD_BOTH;

  /* lower triangular indices */
  PetscCall(ISIdentity(ip, &perm_identity));
  if (!perm_identity) {
    IS              iip;
    const PetscInt *irip, *rip;

    PetscCall(ISInvertPermutation(ip, PETSC_DECIDE, &iip));
    PetscCall(ISGetIndices(iip, &irip));
    PetscCall(ISGetIndices(ip, &rip));
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(rip, rip + n);
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(irip, irip + n);
    PetscCall(ISRestoreIndices(iip, &irip));
    PetscCall(ISDestroy(&iip));
    PetscCall(ISRestoreIndices(ip, &rip));
    PetscCall(PetscLogCpuToGpu(2. * n * sizeof(PetscInt)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  PetscCall(MatCholeskyFactorNumeric_SeqAIJ(B, A, info));
  B->offloadmask            = PETSC_OFFLOAD_CPU;
  B->ops->solve             = MatSolve_SeqAIJHIPSPARSE_Cholesky;
  B->ops->solvetranspose    = MatSolve_SeqAIJHIPSPARSE_Cholesky; // since symmetric
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  /* get the triangular factors */
  PetscCall(MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEFormExplicitTranspose(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct, *matstructT;
  Mat_SeqAIJ                    *a = (Mat_SeqAIJ *)A->data;
  hipsparseIndexBase_t           indexBase;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  matstruct = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestruct->mat;
  PetscCheck(matstruct, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing mat struct");
  matstructT = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestruct->matTranspose;
  PetscCheck(!A->transupdated || matstructT, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing matTranspose struct");
  if (A->transupdated) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  if (hipsparsestruct->format != MAT_HIPSPARSE_CSR) PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_TRUE));
  if (!hipsparsestruct->matTranspose) { /* create hipsparse matrix */
    matstructT = new Mat_SeqAIJHIPSPARSEMultStruct;
    PetscCallHIPSPARSE(hipsparseCreateMatDescr(&matstructT->descr));
    indexBase = hipsparseGetMatIndexBase(matstruct->descr);
    PetscCallHIPSPARSE(hipsparseSetMatIndexBase(matstructT->descr, indexBase));
    PetscCallHIPSPARSE(hipsparseSetMatType(matstructT->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));

    /* set alpha and beta */
    PetscCallHIP(hipMalloc((void **)&matstructT->alpha_one, sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&matstructT->beta_zero, sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&matstructT->beta_one, sizeof(PetscScalar)));
    PetscCallHIP(hipMemcpy(matstructT->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(matstructT->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(matstructT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));

    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      CsrMatrix *matrixT      = new CsrMatrix;
      matstructT->mat         = matrixT;
      matrixT->num_rows       = A->cmap->n;
      matrixT->num_cols       = A->rmap->n;
      matrixT->num_entries    = a->nz;
      matrixT->row_offsets    = new THRUSTINTARRAY(matrixT->num_rows + 1);
      matrixT->column_indices = new THRUSTINTARRAY(a->nz);
      matrixT->values         = new THRUSTARRAY(a->nz);

      if (!hipsparsestruct->rowoffsets_gpu) hipsparsestruct->rowoffsets_gpu = new THRUSTINTARRAY(A->rmap->n + 1);
      hipsparsestruct->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
      PetscCallHIPSPARSE(hipsparseCreateCsr(&matstructT->matDescr, matrixT->num_rows, matrixT->num_cols, matrixT->num_entries, matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), matrixT->values->data().get(), csrRowOffsetsType, csrColIndType, indexBase, hipsparse_scalartype));
    } else if (hipsparsestruct->format == MAT_HIPSPARSE_ELL || hipsparsestruct->format == MAT_HIPSPARSE_HYB) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MAT_HIPSPARSE_ELL and MAT_HIPSPARSE_HYB are not supported");
    }
  }
  if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) { /* transpose mat struct may be already present, update data */
    CsrMatrix *matrix  = (CsrMatrix *)matstruct->mat;
    CsrMatrix *matrixT = (CsrMatrix *)matstructT->mat;
    PetscCheck(matrix, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrix");
    PetscCheck(matrix->row_offsets, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrix rows");
    PetscCheck(matrix->column_indices, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrix cols");
    PetscCheck(matrix->values, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrix values");
    PetscCheck(matrixT, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrixT");
    PetscCheck(matrixT->row_offsets, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrixT rows");
    PetscCheck(matrixT->column_indices, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrixT cols");
    PetscCheck(matrixT->values, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CsrMatrixT values");
    if (!hipsparsestruct->rowoffsets_gpu) { /* this may be absent when we did not construct the transpose with csr2csc */
      hipsparsestruct->rowoffsets_gpu = new THRUSTINTARRAY(A->rmap->n + 1);
      hipsparsestruct->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
      PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
    }
    if (!hipsparsestruct->csr2csc_i) { // not using hipsparseCsr2cscEx2() because it requires 32-bit indices
      THRUSTINTARRAY row_indices(matrix->num_entries);

      // Transpose the matrix via COO, i.e., by putting the row indices in column_indices[] and the column indices in row_indices[]
      hipsparsestruct->csr2csc_i = new THRUSTINTARRAY(matrix->num_entries); // will store the matrix to matrixT permutation, i.e., entry matrixT[i] is matrix[csr2csc_i[i]]
      PetscCallThrust(thrust::sequence(thrust::device, hipsparsestruct->csr2csc_i->begin(), hipsparsestruct->csr2csc_i->end()));
      PetscCallThrust(thrust::for_each(thrust::device, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(A->rmap->n), Csr2coo(hipsparsestruct->rowoffsets_gpu->data().get(), matrixT->column_indices->data().get())));
      row_indices = *matrix->column_indices;
      // Sort the COO by row then column, and get the permutation csr2csc_i[]
      PetscCallThrust(thrust::sort_by_key(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), matrixT->column_indices->begin())), thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(), matrixT->column_indices->end())),
                                          hipsparsestruct->csr2csc_i->begin()));
      // Finalize matrixT's row_offsets by looking up row_indices[]
      PetscCallThrust(thrust::lower_bound(thrust::device, row_indices.begin(), row_indices.end(), thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(A->cmap->n + 1), matrixT->row_offsets->begin()));
    }
    PetscCallThrust(thrust::gather(thrust::device, hipsparsestruct->csr2csc_i->begin(), hipsparsestruct->csr2csc_i->end(), matrix->values->begin(), matrixT->values->begin()));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));
  /* the compressed row indices is not used for matTranspose */
  matstructT->cprowIndices = NULL;
  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSE *)A->spptr)->matTranspose = matstructT;
  A->transupdated                                 = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_LU(Mat A, Vec b, Vec x)
{
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  Mat_SeqAIJHIPSPARSETriFactors        *fs  = static_cast<Mat_SeqAIJHIPSPARSETriFactors *>(A->spptr);
  const Mat_SeqAIJ                     *aij = static_cast<Mat_SeqAIJ *>(A->data);
  const hipsparseOperation_t            op  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  const hipsparseSpSVAlg_t              alg = HIPSPARSE_SPSV_ALG_DEFAULT;
  PetscInt                              m   = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecHIPGetArrayWrite(x, &xarray));
  PetscCall(VecHIPGetArrayRead(b, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  // Reorder b with the row permutation if needed, and wrap the result in fs->X
  if (fs->rpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->end()), thrust::device_pointer_cast(fs->X)));

  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->rpermIndices ? fs->X : (void *)barray));

  // Solve L Y = X
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_Y, fs->Y));
  // Note that hipsparseSpSV_solve() secretly uses the external buffer used in hipsparseSpSV_analysis()!

#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, op, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_L));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, op, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_L, fs->spsvBuffer_L));
#endif

  // Solve U X = Y
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->cpermIndices ? fs->X : xarray));

#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, op, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_U));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, op, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_U, fs->spsvBuffer_U));
#endif

  // Reorder X with the column permutation if needed, and put the result back to x
  if (fs->cpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X), fs->cpermIndices->begin()),
                                 thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X + m), fs->cpermIndices->end()), xGPU));
  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * aij->nz - m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_LU(Mat A, Vec b, Vec x)
{
  Mat_SeqAIJHIPSPARSETriFactors        *fs  = static_cast<Mat_SeqAIJHIPSPARSETriFactors *>(A->spptr);
  Mat_SeqAIJ                           *aij = static_cast<Mat_SeqAIJ *>(A->data);
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  const hipsparseOperation_t            opA = HIPSPARSE_OPERATION_TRANSPOSE;
  const hipsparseSpSVAlg_t              alg = HIPSPARSE_SPSV_ALG_DEFAULT;
  PetscInt                              m   = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  if (!fs->createdTransposeSpSVDescr) { // Call MatSolveTranspose() for the first time
    PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Lt));
    PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* The matrix is still L. We only do transpose solve with it */
                                                fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Lt, &fs->spsvBufferSize_Lt));

    PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Ut));
    PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut, &fs->spsvBufferSize_Ut));
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Lt, fs->spsvBufferSize_Lt));
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Ut, fs->spsvBufferSize_Ut));
    fs->createdTransposeSpSVDescr = PETSC_TRUE;
  }

  if (!fs->updatedTransposeSpSVAnalysis) {
    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));

    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));
    fs->updatedTransposeSpSVAnalysis = PETSC_TRUE;
  }

  PetscCall(VecHIPGetArrayWrite(x, &xarray));
  PetscCall(VecHIPGetArrayRead(b, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  // Reorder b with the row permutation if needed, and wrap the result in fs->X
  if (fs->rpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU, fs->rpermIndices->end()), thrust::device_pointer_cast(fs->X)));

  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->rpermIndices ? fs->X : (void *)barray));

  // Solve Ut Y = X
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_Y, fs->Y));
#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, alg, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));
#endif

  // Solve Lt X = Y
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, fs->cpermIndices ? fs->X : xarray)); // if need to permute, we need to use the intermediate buffer X

#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_Lt));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, opA, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, alg, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));
#endif

  // Reorder X with the column permutation if needed, and put the result back to x
  if (fs->cpermIndices)
    PetscCallThrust(thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X), fs->cpermIndices->begin()),
                                 thrust::make_permutation_iterator(thrust::device_pointer_cast(fs->X + m), fs->cpermIndices->end()), xGPU));

  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * aij->nz - A->rmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorNumeric_SeqAIJHIPSPARSE_ILU0(Mat fact, Mat A, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs    = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij   = (Mat_SeqAIJ *)fact->data;
  Mat_SeqAIJHIPSPARSE           *Acusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  CsrMatrix                     *Acsr;
  PetscInt                       m, nz;
  PetscBool                      flg;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
  }

  /* Copy A's value to fact */
  m  = fact->rmap->n;
  nz = aij->nz;
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  Acsr = (CsrMatrix *)Acusp->mat->mat;
  PetscCallHIP(hipMemcpyAsync(fs->csrVal, Acsr->values->data().get(), sizeof(PetscScalar) * nz, hipMemcpyDeviceToDevice, PetscDefaultHipStream));

  PetscCall(PetscLogGpuTimeBegin());
  /* Factorize fact inplace */
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                          fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ilu0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    int               numerical_zero;
    hipsparseStatus_t status;
    status = hipsparseXcsrilu02_zeroPivot(fs->handle, fs->ilu0Info_M, &numerical_zero);
    PetscAssert(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Numerical zero pivot detected in csrilu02: A(%d,%d) is zero", numerical_zero, numerical_zero);
  }

  {
    /* hipsparseSpSV_analysis() is numeric, i.e., it requires valid matrix values, therefore, we do it after hipsparseXcsrilu02()
     See discussion at https://github.com/NVIDIA/CUDALibrarySamples/issues/78
    */
    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, fs->spsvBuffer_U));

    fs->updatedSpSVAnalysis = PETSC_TRUE;
    /* L, U values have changed, reset the flag to indicate we need to redo hipsparseSpSV_analysis() for transpose solve */
    fs->updatedTransposeSpSVAnalysis = PETSC_FALSE;
  }

  fact->offloadmask            = PETSC_OFFLOAD_GPU;
  fact->ops->solve             = MatSolve_SeqAIJHIPSPARSE_LU; // spMatDescr_L/U uses 32-bit indices, but hipsparseSpSV_solve() supports both 32 and 64. The info is encoded in hipsparseSpMatDescr_t.
  fact->ops->solvetranspose    = MatSolveTranspose_SeqAIJHIPSPARSE_LU;
  fact->ops->matsolve          = NULL;
  fact->ops->matsolvetranspose = NULL;
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(fs->numericFactFlops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE_ILU0(Mat fact, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs  = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij = (Mat_SeqAIJ *)fact->data;
  PetscInt                       m, nz;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool flg, diagDense;

    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
    PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
    PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, NULL, &diagDense));
    PetscCheck(diagDense, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing a diagonal entry");
  }

  /* Free the old stale stuff */
  PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&fs));

  /* Copy over A's meta data to fact. Note that we also allocated fact's i,j,a on host,
     but they will not be used. Allocate them just for easy debugging.
   */
  PetscCall(MatDuplicateNoCreate_SeqAIJ(fact, A, MAT_DO_NOT_COPY_VALUES, PETSC_TRUE /*malloc*/));

  fact->offloadmask            = PETSC_OFFLOAD_BOTH;
  fact->factortype             = MAT_FACTOR_ILU;
  fact->info.factor_mallocs    = 0;
  fact->info.fill_ratio_given  = info->fill;
  fact->info.fill_ratio_needed = 1.0;

  aij->row = NULL;
  aij->col = NULL;

  /* ====================================================================== */
  /* Copy A's i, j to fact and also allocate the value array of fact.       */
  /* We'll do in-place factorization on fact                                */
  /* ====================================================================== */
  const PetscInt *Ai, *Aj;

  m  = fact->rmap->n;
  nz = aij->nz;

  PetscCallHIP(hipMalloc((void **)&fs->csrRowPtr32, sizeof(*fs->csrRowPtr32) * (m + 1)));
  PetscCallHIP(hipMalloc((void **)&fs->csrColIdx32, sizeof(*fs->csrColIdx32) * nz));
  PetscCallHIP(hipMalloc((void **)&fs->csrVal, sizeof(*fs->csrVal) * nz));
  PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &Ai, &Aj)); // Ai is uncompressed

  PetscCheck(nz <= INT_MAX && m <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "nnz %" PetscInt_FMT " and rows %" PetscInt_FMT " overflow C int", nz, m);
  PetscCallThrust(thrust::transform(thrust::hip::par.on(PetscDefaultHipStream), Ai, Ai + m + 1, fs->csrRowPtr32, PetscIntToCInt()));
  PetscCallThrust(thrust::transform(thrust::hip::par.on(PetscDefaultHipStream), Aj, Aj + nz, fs->csrColIdx32, PetscIntToCInt()));

  /* ====================================================================== */
  /* Create descriptors for M, L, U                                         */
  /* ====================================================================== */
  hipsparseFillMode_t fillMode;
  hipsparseDiagType_t diagType;

  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&fs->matDescr_M));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(fs->matDescr_M, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(fs->matDescr_M, HIPSPARSE_MATRIX_TYPE_GENERAL));

  /* https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDiagType_t
    cusparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
    assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
    all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
    assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
  */
  fillMode = HIPSPARSE_FILL_MODE_LOWER;
  diagType = HIPSPARSE_DIAG_TYPE_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_L, m, m, nz, fs->csrRowPtr32, fs->csrColIdx32, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  fillMode = HIPSPARSE_FILL_MODE_UPPER;
  diagType = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_U, m, m, nz, fs->csrRowPtr32, fs->csrColIdx32, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  /* ========================================================================= */
  /* Query buffer sizes for csrilu0, SpSV and allocate buffers                 */
  /* ========================================================================= */
  PetscCallHIPSPARSE(hipsparseCreateCsrilu02Info(&fs->ilu0Info_M));
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02_bufferSize(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                                     fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ilu0Info_M, &fs->factBufferSize_M));

  PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(PetscScalar) * m));
  PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(PetscScalar) * m));

  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_L));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, &fs->spsvBufferSize_L));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_U));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, &fs->spsvBufferSize_U));

  /* From my experiment with the example at https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/bicgstab,
     and discussion at https://github.com/NVIDIA/CUDALibrarySamples/issues/77,
     spsvBuffer_L/U can not be shared (i.e., the same) for our case, but factBuffer_M can share with either of spsvBuffer_L/U.
     To save memory, we make factBuffer_M share with the bigger of spsvBuffer_L/U.
   */
  if (fs->spsvBufferSize_L > fs->spsvBufferSize_U) {
    PetscCallHIP(hipMalloc((void **)&fs->factBuffer_M, PetscMax(fs->spsvBufferSize_L, (size_t)fs->factBufferSize_M)));
    fs->spsvBuffer_L = fs->factBuffer_M;
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_U, fs->spsvBufferSize_U));
  } else {
    PetscCallHIP(hipMalloc((void **)&fs->factBuffer_M, PetscMax(fs->spsvBufferSize_U, (size_t)fs->factBufferSize_M)));
    fs->spsvBuffer_U = fs->factBuffer_M;
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_L, fs->spsvBufferSize_L));
  }

  /* ========================================================================== */
  /* Perform analysis of ilu0 on M, SpSv on L and U                             */
  /* The lower(upper) triangular part of M has the same sparsity pattern as L(U)*/
  /* ========================================================================== */
  int               structural_zero;
  hipsparseStatus_t status;

  fs->policy_M = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02_analysis(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                                   fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ilu0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    /* hipsparseXcsrilu02_zeroPivot() is a blocking call. It calls hipDeviceSynchronize() to make sure all previous kernels are done. */
    status = hipsparseXcsrilu02_zeroPivot(fs->handle, fs->ilu0Info_M, &structural_zero);
    PetscCheck(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Structural zero pivot detected in csrilu02: A(%d,%d) is missing", structural_zero, structural_zero);
  }

  /* Estimate FLOPs of the numeric factorization */
  {
    Mat_SeqAIJ     *Aseq = (Mat_SeqAIJ *)A->data;
    PetscInt       *Ai, nzRow, nzLeft;
    const PetscInt *adiag;
    PetscLogDouble  flops = 0.0;

    PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, &adiag, NULL));
    Ai = Aseq->i;
    for (PetscInt i = 0; i < m; i++) {
      if (Ai[i] < adiag[i] && adiag[i] < Ai[i + 1]) { /* There are nonzeros left to the diagonal of row i */
        nzRow  = Ai[i + 1] - Ai[i];
        nzLeft = adiag[i] - Ai[i];
        /* We want to eliminate nonzeros left to the diagonal one by one. Assume each time, nonzeros right
          and include the eliminated one will be updated, which incurs a multiplication and an addition.
        */
        nzLeft = (nzRow - 1) / 2;
        flops += nzLeft * (2.0 * nzRow - nzLeft + 1);
      }
    }
    fs->numericFactFlops = flops;
  }
  fact->ops->lufactornumeric = MatILUFactorNumeric_SeqAIJHIPSPARSE_ILU0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_ICC0(Mat fact, Vec b, Vec x)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs  = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij = (Mat_SeqAIJ *)fact->data;
  const PetscScalar             *barray;
  PetscScalar                   *xarray;

  PetscFunctionBegin;
  PetscCall(VecHIPGetArrayWrite(x, &xarray));
  PetscCall(VecHIPGetArrayRead(b, &barray));
  PetscCall(PetscLogGpuTimeBegin());

  /* Solve L*y = b */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, (void *)barray));
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_Y, fs->Y));
#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* L Y = X */
                                         fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* L Y = X */
                                         fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));
#endif
  /* Solve Lt*x = y */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, xarray));
#if PETSC_PKG_HIP_VERSION_EQ(5, 6, 0) || PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* Lt X = Y */
                                         fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt));
#else
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* Lt X = Y */
                                         fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));
#endif
  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));

  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * aij->nz - fact->rmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatICCFactorNumeric_SeqAIJHIPSPARSE_ICC0(Mat fact, Mat A, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs    = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij   = (Mat_SeqAIJ *)fact->data;
  Mat_SeqAIJHIPSPARSE           *Acusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  CsrMatrix                     *Acsr;
  PetscInt                       m, nz;
  PetscBool                      flg;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
  }

  /* Copy A's value to fact */
  m  = fact->rmap->n;
  nz = aij->nz;
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  Acsr = (CsrMatrix *)Acusp->mat->mat;
  PetscCallHIP(hipMemcpyAsync(fs->csrVal, Acsr->values->data().get(), sizeof(PetscScalar) * nz, hipMemcpyDeviceToDevice, PetscDefaultHipStream));

  /* Factorize fact inplace */
  /* https://docs.nvidia.com/cuda/cusparse/index.html#csric02_solve
     csric02() only takes the lower triangular part of matrix A to perform factorization.
     The matrix type must be CUSPARSE_MATRIX_TYPE_GENERAL, the fill mode and diagonal type are ignored,
     and the strictly upper triangular part is ignored and never touched. It does not matter if A is Hermitian or not.
     In other words, from the point of view of csric02() A is Hermitian and only the lower triangular part is provided.
   */
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ic0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    int               numerical_zero;
    hipsparseStatus_t status;
    status = hipsparseXcsric02_zeroPivot(fs->handle, fs->ic0Info_M, &numerical_zero);
    PetscAssert(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Numerical zero pivot detected in csric02: A(%d,%d) is zero", numerical_zero, numerical_zero);
  }

  {
    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

    /* Note that cusparse reports this error if we use double and CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    ** On entry to cusparseSpSV_analysis(): conjugate transpose (opA) is not supported for matA data type, current -> CUDA_R_64F
  */
    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));
    fs->updatedSpSVAnalysis = PETSC_TRUE;
  }

  fact->offloadmask            = PETSC_OFFLOAD_GPU;
  fact->ops->solve             = MatSolve_SeqAIJHIPSPARSE_ICC0;
  fact->ops->solvetranspose    = MatSolve_SeqAIJHIPSPARSE_ICC0;
  fact->ops->matsolve          = NULL;
  fact->ops->matsolvetranspose = NULL;
  PetscCall(PetscLogGpuFlops(fs->numericFactFlops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE_ICC0(Mat fact, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs  = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij = (Mat_SeqAIJ *)fact->data;
  PetscInt                       m, nz;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool flg, diagDense;

    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
    PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
    PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, NULL, &diagDense));
    PetscCheck(diagDense, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entries");
  }

  /* Free the old stale stuff */
  PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&fs));

  /* Copy over A's meta data to fact. Note that we also allocated fact's i,j,a on host,
     but they will not be used. Allocate them just for easy debugging.
   */
  PetscCall(MatDuplicateNoCreate_SeqAIJ(fact, A, MAT_DO_NOT_COPY_VALUES, PETSC_TRUE /*malloc*/));

  fact->offloadmask            = PETSC_OFFLOAD_BOTH;
  fact->factortype             = MAT_FACTOR_ICC;
  fact->info.factor_mallocs    = 0;
  fact->info.fill_ratio_given  = info->fill;
  fact->info.fill_ratio_needed = 1.0;

  aij->row = NULL;
  aij->col = NULL;

  /* ====================================================================== */
  /* Copy A's i, j to fact and also allocate the value array of fact.       */
  /* We'll do in-place factorization on fact                                */
  /* ====================================================================== */
  const PetscInt *Ai, *Aj;

  m  = fact->rmap->n;
  nz = aij->nz;

  PetscCallHIP(hipMalloc((void **)&fs->csrRowPtr32, sizeof(*fs->csrRowPtr32) * (m + 1)));
  PetscCallHIP(hipMalloc((void **)&fs->csrColIdx32, sizeof(*fs->csrColIdx32) * nz));
  PetscCallHIP(hipMalloc((void **)&fs->csrVal, sizeof(PetscScalar) * nz));
  PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &Ai, &Aj)); // Ai is uncompressed

  PetscCheck(nz <= INT_MAX && m <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "nnz %" PetscInt_FMT " and rows %" PetscInt_FMT " overflow C int", nz, m);
  PetscCallThrust(thrust::transform(thrust::hip::par.on(PetscDefaultHipStream), Ai, Ai + m + 1, fs->csrRowPtr32, PetscIntToCInt()));
  PetscCallThrust(thrust::transform(thrust::hip::par.on(PetscDefaultHipStream), Aj, Aj + nz, fs->csrColIdx32, PetscIntToCInt()));

  /* ====================================================================== */
  /* Create mat descriptors for M, L                                        */
  /* ====================================================================== */
  hipsparseFillMode_t fillMode;
  hipsparseDiagType_t diagType;

  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&fs->matDescr_M));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(fs->matDescr_M, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(fs->matDescr_M, HIPSPARSE_MATRIX_TYPE_GENERAL));

  /* https://docs.nvidia.com/cuda/cusparse/index.html#cusparseDiagType_t
    cusparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
    assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
    all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
    assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
  */
  fillMode = HIPSPARSE_FILL_MODE_LOWER;
  diagType = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_L, m, m, nz, fs->csrRowPtr32, fs->csrColIdx32, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  /* ========================================================================= */
  /* Query buffer sizes for csric0, SpSV of L and Lt, and allocate buffers     */
  /* ========================================================================= */
  PetscCallHIPSPARSE(hipsparseCreateCsric02Info(&fs->ic0Info_M));
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02_bufferSize(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ic0Info_M, &fs->factBufferSize_M));

  PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(PetscScalar) * m));
  PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(PetscScalar) * m));

  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_L));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, &fs->spsvBufferSize_L));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Lt));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, &fs->spsvBufferSize_Lt));

  /* To save device memory, we make the factorization buffer share with one of the solver buffer.
     See also comments in MatILUFactorSymbolic_SeqAIJCUSPARSE_ILU0().
   */
  if (fs->spsvBufferSize_L > fs->spsvBufferSize_Lt) {
    PetscCallHIP(hipMalloc((void **)&fs->factBuffer_M, PetscMax(fs->spsvBufferSize_L, (size_t)fs->factBufferSize_M)));
    fs->spsvBuffer_L = fs->factBuffer_M;
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Lt, fs->spsvBufferSize_Lt));
  } else {
    PetscCallHIP(hipMalloc((void **)&fs->factBuffer_M, PetscMax(fs->spsvBufferSize_Lt, (size_t)fs->factBufferSize_M)));
    fs->spsvBuffer_Lt = fs->factBuffer_M;
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_L, fs->spsvBufferSize_L));
  }

  /* ========================================================================== */
  /* Perform analysis of ic0 on M                                               */
  /* The lower triangular part of M has the same sparsity pattern as L          */
  /* ========================================================================== */
  int               structural_zero;
  hipsparseStatus_t status;

  fs->policy_M = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02_analysis(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr32, fs->csrColIdx32, fs->ic0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    /* hipsparseXcsric02_zeroPivot() is a blocking call. It calls cudaDeviceSynchronize() to make sure all previous kernels are done. */
    status = hipsparseXcsric02_zeroPivot(fs->handle, fs->ic0Info_M, &structural_zero);
    PetscCheck(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Structural zero pivot detected in csric02: A(%d,%d) is missing", structural_zero, structural_zero);
  }

  /* Estimate FLOPs of the numeric factorization */
  {
    Mat_SeqAIJ    *Aseq = (Mat_SeqAIJ *)A->data;
    PetscInt      *Ai, nzRow, nzLeft;
    PetscLogDouble flops = 0.0;

    Ai = Aseq->i;
    for (PetscInt i = 0; i < m; i++) {
      nzRow = Ai[i + 1] - Ai[i];
      if (nzRow > 1) {
        /* We want to eliminate nonzeros left to the diagonal one by one. Assume each time, nonzeros right
          and include the eliminated one will be updated, which incurs a multiplication and an addition.
        */
        nzLeft = (nzRow - 1) / 2;
        flops += nzLeft * (2.0 * nzRow - nzLeft + 1);
      }
    }
    fs->numericFactFlops = flops;
  }
  fact->ops->choleskyfactornumeric = MatICCFactorNumeric_SeqAIJHIPSPARSE_ICC0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat B, Mat A, const MatFactorInfo *info)
{
  // use_cpu_solve is a field in Mat_SeqAIJHIPSPARSE. B, a factored matrix, uses Mat_SeqAIJHIPSPARSETriFactors.
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = static_cast<Mat_SeqAIJHIPSPARSE *>(A->spptr);

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  PetscCall(MatLUFactorNumeric_SeqAIJ(B, A, info));
  B->offloadmask = PETSC_OFFLOAD_CPU;

  if (!hipsparsestruct->use_cpu_solve) {
    B->ops->solve          = MatSolve_SeqAIJHIPSPARSE_LU;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_LU;
  }
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;

  /* get the triangular factors */
  if (!hipsparsestruct->use_cpu_solve) PetscCall(MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
  PetscCall(MatLUFactorSymbolic_SeqAIJ(B, A, isrow, iscol, info));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
  PetscBool row_identity = PETSC_FALSE, col_identity = PETSC_FALSE;
  if (!info->factoronhost) {
    PetscCall(ISIdentity(isrow, &row_identity));
    PetscCall(ISIdentity(iscol, &col_identity));
  }
  if (!info->levels && row_identity && col_identity) PetscCall(MatILUFactorSymbolic_SeqAIJHIPSPARSE_ILU0(B, A, isrow, iscol, info));
  else {
    PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
    PetscCall(MatILUFactorSymbolic_SeqAIJ(B, A, isrow, iscol, info));
    B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
  PetscBool perm_identity = PETSC_FALSE;
  if (!info->factoronhost) PetscCall(ISIdentity(perm, &perm_identity));
  if (!info->levels && perm_identity) PetscCall(MatICCFactorSymbolic_SeqAIJHIPSPARSE_ICC0(B, A, perm, info));
  else {
    PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
    PetscCall(MatICCFactorSymbolic_SeqAIJ(B, A, perm, info));
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
  PetscCall(MatCholeskyFactorSymbolic_SeqAIJ(B, A, perm, info));
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_hipsparse(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHIPSPARSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERHIPSPARSE = "hipsparse" - A matrix type providing triangular solvers for sequential matrices
  on a single GPU of type, `MATSEQAIJHIPSPARSE`. Currently supported
  algorithms are ILU(k) and ICC(k). Typically, deeper factorizations (larger k) results in poorer
  performance in the triangular solves. Full LU, and Cholesky decompositions can be solved through the
  HipSPARSE triangular solve algorithm. However, the performance can be quite poor and thus these
  algorithms are not recommended. This class does NOT support direct solver operations.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATSEQAIJHIPSPARSE`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCreateSeqAIJHIPSPARSE()`, `MATAIJHIPSPARSE`, `MatCreateAIJHIPSPARSE()`, `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat A, MatFactorType ftype, Mat *B)
{
  PetscInt n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  (*B)->factortype = ftype; // factortype makes MatSetType() allocate spptr of type Mat_SeqAIJHIPSPARSETriFactors
  PetscCall(MatSetType(*B, MATSEQAIJHIPSPARSE));

  if (A->boundtocpu && A->bindingpropagates) PetscCall(MatBindToCPU(*B, PETSC_TRUE));
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    PetscCall(MatSetBlockSizesFromMats(*B, A, A));
    if (!A->boundtocpu) {
      (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJHIPSPARSE;
      (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJHIPSPARSE;
    } else {
      (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJ;
      (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJ;
    }
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILUDT]));
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    if (!A->boundtocpu) {
      (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJHIPSPARSE;
      (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE;
    } else {
      (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJ;
      (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ;
    }
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported for HIPSPARSE Matrix Types");

  PetscCall(MatSeqAIJSetPreallocation(*B, MAT_SKIP_ALLOCATION, NULL));
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscObjectComposeFunction((PetscObject)*B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_hipsparse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat A)
{
  Mat_SeqAIJ                    *a    = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSE           *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactors *fs   = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    PetscCall(PetscLogEventBegin(MAT_HIPSPARSECopyFromGPU, A, 0, 0, 0));
    if (A->factortype == MAT_FACTOR_NONE) {
      CsrMatrix *matrix = (CsrMatrix *)cusp->mat->mat;
      PetscCallHIP(hipMemcpy(a->a, matrix->values->data().get(), a->nz * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    } else if (fs->csrVal) {
      /* We have a factorized matrix on device and are able to copy it to host */
      PetscCallHIP(hipMemcpy(a->a, fs->csrVal, a->nz * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for copying this type of factorized matrix from device to host");
    PetscCall(PetscLogGpuToCpu(a->nz * sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(MAT_HIPSPARSECopyFromGPU, A, 0, 0, 0));
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Policy struct for MatSeqAIJCUSPARSE_CUPM shared template (HIP specialisation) */
struct MatSeqAIJHIPSPARSE_Policy {
  typedef Mat_SeqAIJHIPSPARSE           mat_struct_type;
  typedef Mat_SeqAIJHIPSPARSEMultStruct mult_struct_type;

  static int storage_format_csr() { return (int)MAT_HIPSPARSE_CSR; }
  static int storage_format_ell() { return (int)MAT_HIPSPARSE_ELL; }
  static int storage_format_hyb() { return (int)MAT_HIPSPARSE_HYB; }

  static PetscErrorCode CopyToGPU(Mat A) { return MatSeqAIJHIPSPARSECopyToGPU(A); }
  static PetscErrorCode CopyFromGPU(Mat A) { return MatSeqAIJHIPSPARSECopyFromGPU(A); }
  static PetscErrorCode InvalidateTranspose(Mat A, PetscBool d) { return MatSeqAIJHIPSPARSEInvalidateTranspose(A, d); }
  static PetscErrorCode ConvertFromSeqAIJ(Mat B, MatType t, MatReuse r, Mat *C) { return MatConvert_SeqAIJ_SeqAIJHIPSPARSE(B, t, r, C); }
  static const char    *mat_type_name;

  static PetscErrorCode Destroy(Mat A) { return MatSeqAIJHIPSPARSE_Destroy(A); }
  static PetscErrorCode TriFactorsDestroy(void **spptr) { return MatSeqAIJHIPSPARSETriFactors_Destroy((Mat_SeqAIJHIPSPARSETriFactors **)spptr); }
  static const char    *set_format_c;
  static const char    *set_use_cpu_solve_c;
  static const char    *product_seqdense_device_c;
  static const char    *product_seqdense_c;
  static const char    *product_self_c;
  static const char    *seq_convert_hypre_c;

  static PetscErrorCode VecGetArrayRead(Vec v, const PetscScalar **a) { return VecHIPGetArrayRead(v, a); }
  static PetscErrorCode VecRestoreArrayRead(Vec v, const PetscScalar **a) { return VecHIPRestoreArrayRead(v, a); }
  static PetscErrorCode VecGetArrayWrite(Vec v, PetscScalar **a) { return VecHIPGetArrayWrite(v, a); }
  static PetscErrorCode VecRestoreArrayWrite(Vec v, PetscScalar **a) { return VecHIPRestoreArrayWrite(v, a); }
};
const char *MatSeqAIJHIPSPARSE_Policy::mat_type_name             = MATSEQAIJHIPSPARSE;
const char *MatSeqAIJHIPSPARSE_Policy::set_format_c              = "MatHIPSPARSESetFormat_C";
const char *MatSeqAIJHIPSPARSE_Policy::set_use_cpu_solve_c       = "MatHIPSPARSESetUseCPUSolve_C";
const char *MatSeqAIJHIPSPARSE_Policy::product_seqdense_device_c = "MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C";
const char *MatSeqAIJHIPSPARSE_Policy::product_seqdense_c        = "MatProductSetFromOptions_seqaijhipsparse_seqdense_C";
const char *MatSeqAIJHIPSPARSE_Policy::product_self_c            = "MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C";
const char *MatSeqAIJHIPSPARSE_Policy::seq_convert_hypre_c       = "MatConvert_seqaijhipsparse_hypre_C";

using MatSeqAIJHIPSPARSE_CUPM_t = Petsc::mat::aij::cupm::impl::MatSeqAIJCUSPARSE_CUPM<Petsc::device::cupm::DeviceType::HIP, MatSeqAIJHIPSPARSE_Policy>;

static PetscErrorCode MatSeqAIJGetArray_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJGetArray(A, array);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJRestoreArray(A, array);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJHIPSPARSE(Mat A, const PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJGetArrayRead(A, array);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJHIPSPARSE(Mat A, const PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJRestoreArrayRead(A, array);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJGetArrayWrite(A, array);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::SeqAIJRestoreArrayWrite(A, array);
}

static PetscErrorCode MatSeqAIJGetCSRAndMemType_SeqAIJHIPSPARSE(Mat A, const PetscInt **i, const PetscInt **j, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqAIJHIPSPARSE *cusp;
  CsrMatrix           *matrix;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  cusp = static_cast<Mat_SeqAIJHIPSPARSE *>(A->spptr);
  PetscCheck(cusp != NULL, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "cusp is NULL");
  matrix = (CsrMatrix *)cusp->mat->mat;

  if (i) *i = matrix->row_offsets->data().get();
  if (j) *j = matrix->column_indices->data().get();
  if (a) *a = matrix->values->data().get();
  if (mtype) *mtype = PETSC_MEMTYPE_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct       = hipsparsestruct->mat;
  Mat_SeqAIJ                    *a               = (Mat_SeqAIJ *)A->data;
  PetscInt                       m               = A->rmap->n, *ii, *ridx, tmp;
  PetscBool                      both            = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCheck(!A->boundtocpu, PETSC_COMM_SELF, PETSC_ERR_GPU, "Cannot copy to GPU");
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (A->nonzerostate == hipsparsestruct->nonzerostate && hipsparsestruct->format == MAT_HIPSPARSE_CSR) { /* Copy values only */
      CsrMatrix *matrix;
      matrix = (CsrMatrix *)hipsparsestruct->mat->mat;

      PetscCheck(!a->nz || a->a, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CSR values");
      PetscCall(PetscLogEventBegin(MAT_HIPSPARSECopyToGPU, A, 0, 0, 0));
      matrix->values->assign(a->a, a->a + a->nz);
      PetscCallHIP(WaitForHIP());
      PetscCall(PetscLogCpuToGpu(a->nz * sizeof(PetscScalar)));
      PetscCall(PetscLogEventEnd(MAT_HIPSPARSECopyToGPU, A, 0, 0, 0));
      PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_FALSE));
    } else {
      PetscInt nnz;
      PetscCall(PetscLogEventBegin(MAT_HIPSPARSECopyToGPU, A, 0, 0, 0));
      PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&hipsparsestruct->mat, hipsparsestruct->format));
      PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_TRUE));
      delete hipsparsestruct->workVector;
      delete hipsparsestruct->rowoffsets_gpu;
      hipsparsestruct->workVector     = NULL;
      hipsparsestruct->rowoffsets_gpu = NULL;
      try {
        if (a->compressedrow.use) {
          m    = a->compressedrow.nrows;
          ii   = a->compressedrow.i;
          ridx = a->compressedrow.rindex;
        } else {
          m    = A->rmap->n;
          ii   = a->i;
          ridx = NULL;
        }
        PetscCheck(ii, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CSR row data");
        if (!a->a) {
          nnz  = ii[m];
          both = PETSC_FALSE;
        } else nnz = a->nz;
        PetscCheck(!nnz || a->j, PETSC_COMM_SELF, PETSC_ERR_GPU, "Missing CSR column data");

        /* create cusparse matrix */
        hipsparsestruct->nrows = m;
        matstruct              = new Mat_SeqAIJHIPSPARSEMultStruct;
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&matstruct->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(matstruct->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(matstruct->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));

        PetscCallHIP(hipMalloc((void **)&matstruct->alpha_one, sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&matstruct->beta_zero, sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&matstruct->beta_one, sizeof(PetscScalar)));
        PetscCallHIP(hipMemcpy(matstruct->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(matstruct->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(matstruct->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIPSPARSE(hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE));

        /* Build a hybrid/ellpack matrix if this option is chosen for the storage */
        if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
          /* set the matrix */
          CsrMatrix *mat   = new CsrMatrix;
          mat->num_rows    = m;
          mat->num_cols    = A->cmap->n;
          mat->num_entries = nnz;
          PetscCallCXX(mat->row_offsets = new THRUSTINTARRAY(m + 1));
          mat->row_offsets->assign(ii, ii + m + 1);
          PetscCallCXX(mat->column_indices = new THRUSTINTARRAY(nnz));
          mat->column_indices->assign(a->j, a->j + nnz);

          PetscCallCXX(mat->values = new THRUSTARRAY(nnz));
          if (a->a) mat->values->assign(a->a, a->a + nnz);

          /* assign the pointer */
          matstruct->mat = mat;
          if (mat->num_rows) { /* cusparse errors on empty matrices! */
            PetscCallHIPSPARSE(hipsparseCreateCsr(&matstruct->matDescr, mat->num_rows, mat->num_cols, mat->num_entries, mat->row_offsets->data().get(), mat->column_indices->data().get(), mat->values->data().get(), csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
          }
        } else if (hipsparsestruct->format == MAT_HIPSPARSE_ELL || hipsparsestruct->format == MAT_HIPSPARSE_HYB) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MAT_HIPSPARSE_ELL and MAT_HIPSPARSE_HYB are not supported");
        }

        /* assign the compressed row indices */
        if (a->compressedrow.use) {
          PetscCallCXX(hipsparsestruct->workVector = new THRUSTARRAY(m));
          PetscCallCXX(matstruct->cprowIndices = new THRUSTINTARRAY(m));
          matstruct->cprowIndices->assign(ridx, ridx + m);
          tmp = m;
        } else {
          hipsparsestruct->workVector = NULL;
          matstruct->cprowIndices     = NULL;
          tmp                         = 0;
        }
        PetscCall(PetscLogCpuToGpu(((m + 1) + (a->nz)) * sizeof(int) + tmp * sizeof(PetscInt) + (3 + (a->nz)) * sizeof(PetscScalar)));

        /* assign the pointer */
        hipsparsestruct->mat = matstruct;
      } catch (char *ex) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HIPSPARSE error: %s", ex);
      }
      PetscCallHIP(WaitForHIP());
      PetscCall(PetscLogEventEnd(MAT_HIPSPARSECopyToGPU, A, 0, 0, 0));
      hipsparsestruct->nonzerostate = A->nonzerostate;
    }
    if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct VecHIPPlusEquals {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

struct VecHIPEquals {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

struct VecHIPEqualsReverse {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t);
  }
};

struct MatProductCtx_MatMatHipsparse {
  PetscBool              cisdense;
  PetscScalar           *Bt;
  Mat                    X;
  PetscBool              reusesym; /* Hipsparse does not have split symbolic and numeric phases for sparse matmat operations */
  PetscLogDouble         flops;
  CsrMatrix             *Bcsr;
  hipsparseSpMatDescr_t  matSpBDescr;
  PetscBool              initialized; /* C = alpha op(A) op(B) + beta C */
  hipsparseDnMatDescr_t  matBDescr;
  hipsparseDnMatDescr_t  matCDescr;
  PetscInt               Blda, Clda; /* Record leading dimensions of B and C here to detect changes*/
  size_t                 mmBufferSize;
  void                  *mmBuffer, *mmBuffer2; /* SpGEMM WorkEstimation buffer */
  hipsparseSpGEMMDescr_t spgemmDesc;
};

static PetscErrorCode MatProductCtxDestroy_MatMatHipsparse(PetscCtxRt data)
{
  MatProductCtx_MatMatHipsparse *mmdata = *(MatProductCtx_MatMatHipsparse **)data;

  PetscFunctionBegin;
  PetscCallHIP(hipFree(mmdata->Bt));
  delete mmdata->Bcsr;
  if (mmdata->matSpBDescr) PetscCallHIPSPARSE(hipsparseDestroySpMat(mmdata->matSpBDescr));
  if (mmdata->matBDescr) PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matBDescr));
  if (mmdata->matCDescr) PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matCDescr));
  if (mmdata->spgemmDesc) PetscCallHIPSPARSE(hipsparseSpGEMM_destroyDescr(mmdata->spgemmDesc));
  PetscCallHIP(hipFree(mmdata->mmBuffer));
  PetscCallHIP(hipFree(mmdata->mmBuffer2));
  PetscCall(MatDestroy(&mmdata->X));
  PetscCall(PetscFree(*(void **)data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_SeqAIJHIPSPARSE_SeqDENSEHIP(Mat C)
{
  Mat_Product                   *product = C->product;
  Mat                            A, B;
  PetscInt                       m, n, blda, clda;
  PetscBool                      flg, biship;
  Mat_SeqAIJHIPSPARSE           *cusp;
  hipsparseOperation_t           opA;
  const PetscScalar             *barray;
  PetscScalar                   *carray;
  MatProductCtx_MatMatHipsparse *mmdata;
  Mat_SeqAIJHIPSPARSEMultStruct *mat;
  CsrMatrix                     *csrmat;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data empty");
  mmdata = (MatProductCtx_MatMatHipsparse *)product->data;
  A      = product->A;
  B      = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Not for type %s", ((PetscObject)A)->type_name);
  /* currently CopyToGpu does not copy if the matrix is bound to CPU
     Instead of silently accepting the wrong answer, I prefer to raise the error */
  PetscCheck(!A->boundtocpu, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_PtAP:
    mat = cusp->mat;
    opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    if (!A->form_explicit_transpose) {
      mat = cusp->mat;
      opA = HIPSPARSE_OPERATION_TRANSPOSE;
    } else {
      PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(A));
      mat = cusp->matTranspose;
      opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    }
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
  case MATPRODUCT_RARt:
    mat = cusp->mat;
    opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->rmap->n;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Unsupported product type %s", MatProductTypes[product->type]);
  }
  PetscCheck(mat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csrmat = (CsrMatrix *)mat->mat;
  /* if the user passed a CPU matrix, copy the data to the GPU */
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEHIP, &biship));
  if (!biship) PetscCall(MatConvert(B, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &B));
  PetscCall(MatDenseGetArrayReadAndMemType(B, &barray, nullptr));
  PetscCall(MatDenseGetLDA(B, &blda));
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    PetscCall(MatDenseGetArrayWriteAndMemType(mmdata->X, &carray, nullptr));
    PetscCall(MatDenseGetLDA(mmdata->X, &clda));
  } else {
    PetscCall(MatDenseGetArrayWriteAndMemType(C, &carray, nullptr));
    PetscCall(MatDenseGetLDA(C, &clda));
  }

  PetscCall(PetscLogGpuTimeBegin());
  hipsparseOperation_t opB = (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  /* (re)allocate mmBuffer if not initialized or LDAs are different */
  if (!mmdata->initialized || mmdata->Blda != blda || mmdata->Clda != clda) {
    size_t mmBufferSize;
    if (mmdata->initialized && mmdata->Blda != blda) {
      PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matBDescr));
      mmdata->matBDescr = NULL;
    }
    if (!mmdata->matBDescr) {
      PetscCallHIPSPARSE(hipsparseCreateDnMat(&mmdata->matBDescr, B->rmap->n, B->cmap->n, blda, (void *)barray, hipsparse_scalartype, HIPSPARSE_ORDER_COL));
      mmdata->Blda = blda;
    }
    if (mmdata->initialized && mmdata->Clda != clda) {
      PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matCDescr));
      mmdata->matCDescr = NULL;
    }
    if (!mmdata->matCDescr) { /* matCDescr is for C or mmdata->X */
      PetscCallHIPSPARSE(hipsparseCreateDnMat(&mmdata->matCDescr, m, n, clda, (void *)carray, hipsparse_scalartype, HIPSPARSE_ORDER_COL));
      mmdata->Clda = clda;
    }
    if (!mat->matDescr) {
      PetscCallHIPSPARSE(hipsparseCreateCsr(&mat->matDescr, csrmat->num_rows, csrmat->num_cols, csrmat->num_entries, csrmat->row_offsets->data().get(), csrmat->column_indices->data().get(), csrmat->values->data().get(), csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
    }
    PetscCallHIPSPARSE(hipsparseSpMM_bufferSize(cusp->handle, opA, opB, mat->alpha_one, mat->matDescr, mmdata->matBDescr, mat->beta_zero, mmdata->matCDescr, hipsparse_scalartype, cusp->spmmAlg, &mmBufferSize));
    if ((mmdata->mmBuffer && mmdata->mmBufferSize < mmBufferSize) || !mmdata->mmBuffer) {
      PetscCallHIP(hipFree(mmdata->mmBuffer));
      PetscCallHIP(hipMalloc(&mmdata->mmBuffer, mmBufferSize));
      mmdata->mmBufferSize = mmBufferSize;
    }
    mmdata->initialized = PETSC_TRUE;
  } else {
    /* to be safe, always update pointers of the mats */
    PetscCallHIPSPARSE(hipsparseSpMatSetValues(mat->matDescr, csrmat->values->data().get()));
    PetscCallHIPSPARSE(hipsparseDnMatSetValues(mmdata->matBDescr, (void *)barray));
    PetscCallHIPSPARSE(hipsparseDnMatSetValues(mmdata->matCDescr, (void *)carray));
  }

  /* do hipsparseSpMM, which supports transpose on B */
  PetscCallHIPSPARSE(hipsparseSpMM(cusp->handle, opA, opB, mat->alpha_one, mat->matDescr, mmdata->matBDescr, mat->beta_zero, mmdata->matCDescr, hipsparse_scalartype, cusp->spmmAlg, mmdata->mmBuffer));

  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(n * 2.0 * csrmat->num_entries));
  PetscCall(MatDenseRestoreArrayReadAndMemType(B, &barray));
  if (product->type == MATPRODUCT_RARt) {
    PetscCall(MatDenseRestoreArrayWriteAndMemType(mmdata->X, &carray));
    PetscCall(MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Internal(B, mmdata->X, C, PETSC_FALSE, PETSC_FALSE));
  } else if (product->type == MATPRODUCT_PtAP) {
    PetscCall(MatDenseRestoreArrayWriteAndMemType(mmdata->X, &carray));
    PetscCall(MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Internal(B, mmdata->X, C, PETSC_TRUE, PETSC_FALSE));
  } else PetscCall(MatDenseRestoreArrayWriteAndMemType(C, &carray));
  if (mmdata->cisdense) PetscCall(MatConvert(C, MATSEQDENSE, MAT_INPLACE_MATRIX, &C));
  if (!biship) PetscCall(MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_SeqAIJHIPSPARSE_SeqDENSEHIP(Mat C)
{
  Mat_Product                   *product = C->product;
  Mat                            A, B;
  PetscInt                       m, n;
  PetscBool                      cisdense, flg;
  MatProductCtx_MatMatHipsparse *mmdata;
  Mat_SeqAIJHIPSPARSE           *cusp;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data not empty");
  A = product->A;
  B = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for type %s", ((PetscObject)A)->type_name);
  cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  PetscCheck(cusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
  switch (product->type) {
  case MATPRODUCT_AB:
    m = A->rmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    break;
  case MATPRODUCT_PtAP:
    m = B->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_RARt:
    m = B->rmap->n;
    n = B->rmap->n;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Unsupported product type %s", MatProductTypes[product->type]);
  }
  PetscCall(MatSetSizes(C, m, n, m, n));
  /* if C is of type MATSEQDENSE (CPU), perform the operation on the GPU and then copy on the CPU */
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATSEQDENSE, &cisdense));
  PetscCall(MatSetType(C, MATSEQDENSEHIP));

  /* product data */
  PetscCall(PetscNew(&mmdata));
  mmdata->cisdense = cisdense;
  /* for these products we need intermediate storage */
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)C), &mmdata->X));
    PetscCall(MatSetType(mmdata->X, MATSEQDENSEHIP));
    /* do not preallocate, since the first call to MatDenseHIPGetArray will preallocate on the GPU for us */
    if (product->type == MATPRODUCT_RARt) PetscCall(MatSetSizes(mmdata->X, A->rmap->n, B->rmap->n, A->rmap->n, B->rmap->n));
    else PetscCall(MatSetSizes(mmdata->X, A->rmap->n, B->cmap->n, A->rmap->n, B->cmap->n));
  }
  C->product->data       = mmdata;
  C->product->destroy    = MatProductCtxDestroy_MatMatHipsparse;
  C->ops->productnumeric = MatProductNumeric_SeqAIJHIPSPARSE_SeqDENSEHIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE(Mat C)
{
  Mat_Product                   *product = C->product;
  Mat                            A, B;
  Mat_SeqAIJHIPSPARSE           *Acusp, *Bcusp, *Ccusp;
  Mat_SeqAIJ                    *c = (Mat_SeqAIJ *)C->data;
  Mat_SeqAIJHIPSPARSEMultStruct *Amat, *Bmat, *Cmat;
  CsrMatrix                     *Acsr, *Bcsr, *Ccsr;
  PetscBool                      flg;
  MatProductType                 ptype;
  MatProductCtx_MatMatHipsparse *mmdata;
  hipsparseSpMatDescr_t          BmatSpDescr;
  hipsparseOperation_t           opA = HIPSPARSE_OPERATION_NON_TRANSPOSE, opB = HIPSPARSE_OPERATION_NON_TRANSPOSE; /* hipSPARSE spgemm doesn't support transpose yet */

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data empty");
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for C of type %s", ((PetscObject)C)->type_name);
  mmdata = (MatProductCtx_MatMatHipsparse *)C->product->data;
  A      = product->A;
  B      = product->B;
  if (mmdata->reusesym) { /* this happens when api_user is true, meaning that the matrix values have been already computed in the MatProductSymbolic phase */
    mmdata->reusesym = PETSC_FALSE;
    Ccusp            = (Mat_SeqAIJHIPSPARSE *)C->spptr;
    PetscCheck(Ccusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
    Cmat = Ccusp->mat;
    PetscCheck(Cmat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing C mult struct for product type %s", MatProductTypes[C->product->type]);
    Ccsr = (CsrMatrix *)Cmat->mat;
    PetscCheck(Ccsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing C CSR struct");
    goto finalize;
  }
  if (!c->nz) goto finalize;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for type %s", ((PetscObject)A)->type_name);
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for B of type %s", ((PetscObject)B)->type_name);
  PetscCheck(!A->boundtocpu, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONG, "Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  PetscCheck(!B->boundtocpu, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONG, "Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  Acusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Bcusp = (Mat_SeqAIJHIPSPARSE *)B->spptr;
  Ccusp = (Mat_SeqAIJHIPSPARSE *)C->spptr;
  PetscCheck(Acusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
  PetscCheck(Bcusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
  PetscCheck(Ccusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(B));

  ptype = product->type;
  if (A->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_AtB) {
    ptype = MATPRODUCT_AB;
    PetscCheck(product->symbolic_used_the_fact_A_is_symmetric, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Symbolic should have been built using the fact that A is symmetric");
  }
  if (B->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_ABt) {
    ptype = MATPRODUCT_AB;
    PetscCheck(product->symbolic_used_the_fact_B_is_symmetric, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Symbolic should have been built using the fact that B is symmetric");
  }
  switch (ptype) {
  case MATPRODUCT_AB:
    Amat = Acusp->mat;
    Bmat = Bcusp->mat;
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(A));
    Amat = Acusp->matTranspose;
    Bmat = Bcusp->mat;
    break;
  case MATPRODUCT_ABt:
    Amat = Acusp->mat;
    PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(B));
    Bmat = Bcusp->matTranspose;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Unsupported product type %s", MatProductTypes[product->type]);
  }
  Cmat = Ccusp->mat;
  PetscCheck(Amat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing A mult struct for product type %s", MatProductTypes[ptype]);
  PetscCheck(Bmat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing B mult struct for product type %s", MatProductTypes[ptype]);
  PetscCheck(Cmat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing C mult struct for product type %s", MatProductTypes[ptype]);
  Acsr = (CsrMatrix *)Amat->mat;
  Bcsr = mmdata->Bcsr ? mmdata->Bcsr : (CsrMatrix *)Bmat->mat; /* B may be in compressed row storage */
  Ccsr = (CsrMatrix *)Cmat->mat;
  PetscCheck(Acsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing A CSR struct");
  PetscCheck(Bcsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing B CSR struct");
  PetscCheck(Ccsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing C CSR struct");
  PetscCall(PetscLogGpuTimeBegin());
  BmatSpDescr = mmdata->Bcsr ? mmdata->matSpBDescr : Bmat->matDescr; /* B may be in compressed row storage */
  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer));
  PetscCallHIPSPARSE(hipsparseSpGEMM_copy(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogGpuTimeEnd());
  C->offloadmask = PETSC_OFFLOAD_GPU;
finalize:
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  PetscCall(PetscInfo(C, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded, %" PetscInt_FMT " used\n", C->rmap->n, C->cmap->n, c->nz));
  PetscCall(PetscInfo(C, "Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(C, "Maximum nonzeros in any row is %" PetscInt_FMT "\n", c->rmax));
  c->reallocs = 0;
  C->info.mallocs += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE(Mat C)
{
  Mat_Product                   *product = C->product;
  Mat                            A, B;
  Mat_SeqAIJHIPSPARSE           *Acusp, *Bcusp, *Ccusp;
  Mat_SeqAIJ                    *a, *b, *c;
  Mat_SeqAIJHIPSPARSEMultStruct *Amat, *Bmat, *Cmat;
  CsrMatrix                     *Acsr, *Bcsr, *Ccsr;
  PetscInt                       i, j, m, n, k;
  PetscBool                      flg;
  MatProductType                 ptype;
  MatProductCtx_MatMatHipsparse *mmdata;
  PetscLogDouble                 flops;
  PetscBool                      biscompressed, ciscompressed;
  int64_t                        C_num_rows1, C_num_cols1, C_nnz1;
  hipsparseSpMatDescr_t          BmatSpDescr;
  hipsparseOperation_t           opA = HIPSPARSE_OPERATION_NON_TRANSPOSE, opB = HIPSPARSE_OPERATION_NON_TRANSPOSE; /* HIPSPARSE spgemm doesn't support transpose yet */
  size_t                         bufSize2;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data not empty");
  A = product->A;
  B = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for type %s", ((PetscObject)A)->type_name);
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for B of type %s", ((PetscObject)B)->type_name);
  a = (Mat_SeqAIJ *)A->data;
  b = (Mat_SeqAIJ *)B->data;
  /* product data */
  PetscCall(PetscNew(&mmdata));
  C->product->data    = mmdata;
  C->product->destroy = MatProductCtxDestroy_MatMatHipsparse;

  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(B));
  Acusp = (Mat_SeqAIJHIPSPARSE *)A->spptr; /* Access spptr after MatSeqAIJHIPSPARSECopyToGPU, not before */
  Bcusp = (Mat_SeqAIJHIPSPARSE *)B->spptr;
  PetscCheck(Acusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");
  PetscCheck(Bcusp->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Only for MAT_HIPSPARSE_CSR format");

  ptype = product->type;
  if (A->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_AtB) {
    ptype                                          = MATPRODUCT_AB;
    product->symbolic_used_the_fact_A_is_symmetric = PETSC_TRUE;
  }
  if (B->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_ABt) {
    ptype                                          = MATPRODUCT_AB;
    product->symbolic_used_the_fact_B_is_symmetric = PETSC_TRUE;
  }
  biscompressed = PETSC_FALSE;
  ciscompressed = PETSC_FALSE;
  switch (ptype) {
  case MATPRODUCT_AB:
    m    = A->rmap->n;
    n    = B->cmap->n;
    k    = A->cmap->n;
    Amat = Acusp->mat;
    Bmat = Bcusp->mat;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    k = A->rmap->n;
    PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(A));
    Amat = Acusp->matTranspose;
    Bmat = Bcusp->mat;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    k = A->cmap->n;
    PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(B));
    Amat = Acusp->mat;
    Bmat = Bcusp->matTranspose;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Unsupported product type %s", MatProductTypes[product->type]);
  }

  /* create hipsparse matrix */
  PetscCall(MatSetSizes(C, m, n, m, n));
  PetscCall(MatSetType(C, MATSEQAIJHIPSPARSE));
  c     = (Mat_SeqAIJ *)C->data;
  Ccusp = (Mat_SeqAIJHIPSPARSE *)C->spptr;
  Cmat  = new Mat_SeqAIJHIPSPARSEMultStruct;
  Ccsr  = new CsrMatrix;

  c->compressedrow.use = ciscompressed;
  if (c->compressedrow.use) { /* if a is in compressed row, than c will be in compressed row format */
    c->compressedrow.nrows = a->compressedrow.nrows;
    PetscCall(PetscMalloc2(c->compressedrow.nrows + 1, &c->compressedrow.i, c->compressedrow.nrows, &c->compressedrow.rindex));
    PetscCall(PetscArraycpy(c->compressedrow.rindex, a->compressedrow.rindex, c->compressedrow.nrows));
    Ccusp->workVector  = new THRUSTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices = new THRUSTINTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices->assign(c->compressedrow.rindex, c->compressedrow.rindex + c->compressedrow.nrows);
  } else {
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Ccusp->workVector       = NULL;
    Cmat->cprowIndices      = NULL;
  }
  Ccusp->nrows      = ciscompressed ? c->compressedrow.nrows : m;
  Ccusp->mat        = Cmat;
  Ccusp->mat->mat   = Ccsr;
  Ccsr->num_rows    = Ccusp->nrows;
  Ccsr->num_cols    = n;
  Ccsr->row_offsets = new THRUSTINTARRAY(Ccusp->nrows + 1);
  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&Cmat->descr));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(Cmat->descr, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(Cmat->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  PetscCallHIP(hipMalloc((void **)&Cmat->alpha_one, sizeof(PetscScalar)));
  PetscCallHIP(hipMalloc((void **)&Cmat->beta_zero, sizeof(PetscScalar)));
  PetscCallHIP(hipMalloc((void **)&Cmat->beta_one, sizeof(PetscScalar)));
  PetscCallHIP(hipMemcpy(Cmat->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
  PetscCallHIP(hipMemcpy(Cmat->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
  PetscCallHIP(hipMemcpy(Cmat->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
  if (!Ccsr->num_rows || !Ccsr->num_cols || !a->nz || !b->nz) { /* hipsparse raise errors in different calls when matrices have zero rows/columns! */
    PetscCallThrust(thrust::fill(thrust::device, Ccsr->row_offsets->begin(), Ccsr->row_offsets->end(), 0));
    c->nz                = 0;
    Ccsr->column_indices = new THRUSTINTARRAY(c->nz);
    Ccsr->values         = new THRUSTARRAY(c->nz);
    goto finalizesym;
  }

  PetscCheck(Amat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing A mult struct for product type %s", MatProductTypes[ptype]);
  PetscCheck(Bmat, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing B mult struct for product type %s", MatProductTypes[ptype]);
  Acsr = (CsrMatrix *)Amat->mat;
  if (!biscompressed) {
    Bcsr        = (CsrMatrix *)Bmat->mat;
    BmatSpDescr = Bmat->matDescr;
  } else { /* we need to use row offsets for the full matrix */
    CsrMatrix *cBcsr     = (CsrMatrix *)Bmat->mat;
    Bcsr                 = new CsrMatrix;
    Bcsr->num_rows       = B->rmap->n;
    Bcsr->num_cols       = cBcsr->num_cols;
    Bcsr->num_entries    = cBcsr->num_entries;
    Bcsr->column_indices = cBcsr->column_indices;
    Bcsr->values         = cBcsr->values;
    if (!Bcusp->rowoffsets_gpu) {
      Bcusp->rowoffsets_gpu = new THRUSTINTARRAY(B->rmap->n + 1);
      Bcusp->rowoffsets_gpu->assign(b->i, b->i + B->rmap->n + 1);
      PetscCall(PetscLogCpuToGpu((B->rmap->n + 1) * sizeof(PetscInt)));
    }
    Bcsr->row_offsets = Bcusp->rowoffsets_gpu;
    mmdata->Bcsr      = Bcsr;
    if (Bcsr->num_rows && Bcsr->num_cols) {
      PetscCallHIPSPARSE(hipsparseCreateCsr(&mmdata->matSpBDescr, Bcsr->num_rows, Bcsr->num_cols, Bcsr->num_entries, Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(), Bcsr->values->data().get(), csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
    }
    BmatSpDescr = mmdata->matSpBDescr;
  }
  PetscCheck(Acsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing A CSR struct");
  PetscCheck(Bcsr, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Missing B CSR struct");
  /* precompute flops count */
  if (ptype == MATPRODUCT_AB) {
    for (i = 0, flops = 0; i < A->rmap->n; i++) {
      const PetscInt st = a->i[i];
      const PetscInt en = a->i[i + 1];
      for (j = st; j < en; j++) {
        const PetscInt brow = a->j[j];
        flops += 2. * (b->i[brow + 1] - b->i[brow]);
      }
    }
  } else if (ptype == MATPRODUCT_AtB) {
    for (i = 0, flops = 0; i < A->rmap->n; i++) {
      const PetscInt anzi = a->i[i + 1] - a->i[i];
      const PetscInt bnzi = b->i[i + 1] - b->i[i];
      flops += (2. * anzi) * bnzi;
    }
  } else flops = 0.; /* TODO */

  mmdata->flops = flops;
  PetscCall(PetscLogGpuTimeBegin());

  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  // cuda-12.2 requires non-null csrRowOffsets
  PetscCallHIPSPARSE(hipsparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, 0, Ccsr->row_offsets->data().get(), NULL, NULL, csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpGEMM_createDescr(&mmdata->spgemmDesc));
  // Note that cusparseSpGEMMreuse is deprecated in CUDA 13.2.1

  /* ask bufferSize bytes for external memory */
  PetscCallHIPSPARSE(hipsparseSpGEMM_workEstimation(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufSize2, NULL));
  PetscCallHIP(hipMalloc((void **)&mmdata->mmBuffer2, bufSize2));
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  PetscCallHIPSPARSE(hipsparseSpGEMM_workEstimation(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufSize2, mmdata->mmBuffer2));
  /* ask bufferSize again bytes for external memory */
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, NULL));
  /* The HIPSPARSE documentation is not clear, nor the API
     We need both buffers to perform the operations properly!
     mmdata->mmBuffer2 does not appear anywhere in the compute/copy API
     it only appears for the workEstimation stuff, but it seems it is needed in compute, so probably the address
     is stored in the descriptor! What a messy API... */
  PetscCallHIP(hipMalloc((void **)&mmdata->mmBuffer, mmdata->mmBufferSize));
  /* compute the intermediate product of A * B */
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer));
  /* get matrix C non-zero entries C_nnz1 */
  PetscCallHIPSPARSE(hipsparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1));
  PetscCall(PetscIntCast(C_nnz1, &c->nz));
  PetscCall(PetscInfo(C, "Buffer sizes for type %s, result %" PetscInt_FMT " x %" PetscInt_FMT " (k %" PetscInt_FMT ", nzA %" PetscInt_FMT ", nzB %" PetscInt_FMT ", nzC %" PetscInt_FMT ") are: %ldKB %ldKB\n", MatProductTypes[ptype], m, n, k, a->nz, b->nz, c->nz, bufSize2 / 1024,
                      mmdata->mmBufferSize / 1024));
  Ccsr->column_indices = new THRUSTINTARRAY(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
  // hipSparse errors with null pointers even with nz = 0
  if (c->nz) PetscCallHIPSPARSE(hipsparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get()));
  PetscCallHIPSPARSE(hipsparseSpGEMM_copy(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCall(PetscLogGpuTimeEnd());
finalizesym:
  c->free_a = PETSC_TRUE;
  PetscCall(PetscShmgetAllocateArray(c->nz, sizeof(PetscInt), (void **)&c->j));
  PetscCall(PetscShmgetAllocateArray(m + 1, sizeof(PetscInt), (void **)&c->i));
  c->free_ij = PETSC_TRUE;

  PetscInt *d_i = c->i;
  if (ciscompressed) d_i = c->compressedrow.i;
  PetscCallHIP(hipMemcpy(d_i, Ccsr->row_offsets->data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
  PetscCallHIP(hipMemcpy(c->j, Ccsr->column_indices->data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
  if (ciscompressed) { /* need to expand host row offsets */
    PetscInt r = 0;
    c->i[0]    = 0;
    for (k = 0; k < c->compressedrow.nrows; k++) {
      const PetscInt next = c->compressedrow.rindex[k];
      const PetscInt old  = c->compressedrow.i[k];
      for (; r < next; r++) c->i[r + 1] = old;
    }
    for (; r < m; r++) c->i[r + 1] = c->compressedrow.i[c->compressedrow.nrows];
  }
  PetscCall(PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size()) * sizeof(PetscInt)));
  PetscCall(PetscMalloc1(m, &c->ilen));
  PetscCall(PetscMalloc1(m, &c->imax));
  c->maxnz         = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax          = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k + 1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax, nn);
  }
  PetscCall(PetscMalloc1(c->nz, &c->a));
  Ccsr->num_entries = c->nz;

  C->nonzerostate++;
  PetscCall(PetscLayoutSetUp(C->rmap));
  PetscCall(PetscLayoutSetUp(C->cmap));
  Ccusp->nonzerostate = C->nonzerostate;
  C->offloadmask      = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated     = PETSC_TRUE;
  C->assembled        = PETSC_FALSE;
  C->was_assembled    = PETSC_FALSE;
  if (product->api_user && A->offloadmask == PETSC_OFFLOAD_BOTH && B->offloadmask == PETSC_OFFLOAD_BOTH) { /* flag the matrix C values as computed, so that the numeric phase will only call MatAssembly */
    mmdata->reusesym = PETSC_TRUE;
    C->offloadmask   = PETSC_OFFLOAD_GPU;
  }
  C->ops->productnumeric = MatProductNumeric_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* handles sparse or dense B */
static PetscErrorCode MatProductSetFromOptions_SeqAIJHIPSPARSE(Mat mat)
{
  Mat_Product *product = mat->product;
  PetscBool    isdense = PETSC_FALSE, Biscusp = PETSC_FALSE, Ciscusp = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)product->B, MATSEQDENSE, &isdense));
  if (!product->A->boundtocpu && !product->B->boundtocpu) PetscCall(PetscObjectTypeCompare((PetscObject)product->B, MATSEQAIJHIPSPARSE, &Biscusp));
  if (product->type == MATPRODUCT_ABC) {
    Ciscusp = PETSC_FALSE;
    if (!product->C->boundtocpu) PetscCall(PetscObjectTypeCompare((PetscObject)product->C, MATSEQAIJHIPSPARSE, &Ciscusp));
  }
  if (Biscusp && Ciscusp) { /* we can always select the CPU backend */
    PetscBool usecpu = PETSC_FALSE;
    switch (product->type) {
    case MATPRODUCT_AB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatMatMult", "Mat");
        PetscCall(PetscOptionsBool("-matmatmult_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_AtB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatTransposeMatMult", "Mat");
        PetscCall(PetscOptionsBool("-mattransposematmult_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AtB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_PtAP:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatPtAP", "Mat");
        PetscCall(PetscOptionsBool("-matptap_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_PtAP", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_RARt:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatRARt", "Mat");
        PetscCall(PetscOptionsBool("-matrart_backend_cpu", "Use CPU code", "MatRARt", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_RARt", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatRARt", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_ABC:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatMatMatMult", "Mat");
        PetscCall(PetscOptionsBool("-matmatmatmult_backend_cpu", "Use CPU code", "MatMatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_ABC", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatMatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    default:
      break;
    }
    if (usecpu) Biscusp = Ciscusp = PETSC_FALSE;
  }
  /* dispatch */
  if (isdense) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
      if (product->A->boundtocpu) PetscCall(MatProductSetFromOptions_SeqAIJ_SeqDense(mat));
      else mat->ops->productsymbolic = MatProductSymbolic_SeqAIJHIPSPARSE_SeqDENSEHIP;
      break;
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else if (Biscusp && Ciscusp) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else PetscCall(MatProductSetFromOptions_SeqAIJ(mat)); /* fallback for AIJ */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, NULL, yy, PETSC_FALSE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, yy, zz, PETSC_FALSE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, NULL, yy, PETSC_TRUE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, yy, zz, PETSC_TRUE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, NULL, yy, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

__global__ static void ScatterAdd(PetscInt n, PetscInt *idx, const PetscScalar *x, PetscScalar *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[idx[i]] += x[i];
}

/* z = op(A) x + y. If trans & !herm, op = ^T; if trans & herm, op = ^H; if !trans, op = no-op */
static PetscErrorCode MatMultAddKernel_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy, Vec zz, PetscBool trans, PetscBool herm)
{
  Mat_SeqAIJ                    *a               = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct;
  PetscScalar                   *xarray, *zarray, *dptr, *beta, *xptr;
  hipsparseOperation_t           opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  PetscBool                      compressed;
  PetscInt                       nx, ny;

  PetscFunctionBegin;
  PetscCheck(!herm || trans, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Hermitian and not transpose not supported");
  if (!a->nz) {
    if (yy) PetscCall(VecSeq_HIP::Copy(yy, zz));
    else PetscCall(VecSeq_HIP::Set(zz, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  if (!trans) {
    matstruct = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestruct->mat;
    PetscCheck(matstruct, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "SeqAIJHIPSPARSE does not have a 'mat' (need to fix)");
  } else {
    if (herm || !A->form_explicit_transpose) {
      opA       = herm ? HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE : HIPSPARSE_OPERATION_TRANSPOSE;
      matstruct = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestruct->mat;
    } else {
      if (!hipsparsestruct->matTranspose) PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(A));
      matstruct = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestruct->matTranspose;
    }
  }
  /* Does the matrix use compressed rows (i.e., drop zero rows)? */
  compressed = matstruct->cprowIndices ? PETSC_TRUE : PETSC_FALSE;
  try {
    PetscCall(VecHIPGetArrayRead(xx, (const PetscScalar **)&xarray));
    if (yy == zz) PetscCall(VecHIPGetArray(zz, &zarray)); /* read & write zz, so need to get up-to-date zarray on GPU */
    else PetscCall(VecHIPGetArrayWrite(zz, &zarray));     /* write zz, so no need to init zarray on GPU */

    PetscCall(PetscLogGpuTimeBegin());
    if (opA == HIPSPARSE_OPERATION_NON_TRANSPOSE) {
      /* z = A x + beta y.
         If A is compressed (with less rows), then Ax is shorter than the full z, so we need a work vector to store Ax.
         When A is non-compressed, and z = y, we can set beta=1 to compute y = Ax + y in one call.
      */
      xptr = xarray;
      dptr = compressed ? hipsparsestruct->workVector->data().get() : zarray;
      beta = (yy == zz && !compressed) ? matstruct->beta_one : matstruct->beta_zero;
      /* Get length of x, y for y=Ax. ny might be shorter than the work vector's allocated length, since the work vector is
          allocated to accommodate different uses. So we get the length info directly from mat.
       */
      if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
        CsrMatrix *mat = (CsrMatrix *)matstruct->mat;
        nx             = mat->num_cols;
        ny             = mat->num_rows;
      }
    } else {
      /* z = A^T x + beta y
         If A is compressed, then we need a work vector as the shorter version of x to compute A^T x.
         Note A^Tx is of full length, so we set beta to 1.0 if y exists.
       */
      xptr = compressed ? hipsparsestruct->workVector->data().get() : xarray;
      dptr = zarray;
      beta = yy ? matstruct->beta_one : matstruct->beta_zero;
      if (compressed) { /* Scatter x to work vector */
        thrust::device_ptr<PetscScalar> xarr = thrust::device_pointer_cast(xarray);
        thrust::for_each(
#if PetscDefined(HAVE_THRUST_ASYNC)
          thrust::hip::par.on(PetscDefaultHipStream),
#endif
          thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))),
          thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(), VecHIPEqualsReverse());
      }
      if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
        CsrMatrix *mat = (CsrMatrix *)matstruct->mat;
        nx             = mat->num_rows;
        ny             = mat->num_cols;
      }
    }
    /* csr_spmv does y = alpha op(A) x + beta y */
    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
#if PETSC_PKG_HIP_VERSION_GE(5, 1, 0) && !(PETSC_PKG_HIP_VERSION_GT(6, 4, 3) && PETSC_PKG_HIP_VERSION_LE(7, 2, 0))
      PetscCheck(opA >= 0 && opA <= 2, PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSPARSE API on hipsparseOperation_t has changed and PETSc has not been updated accordingly");
      if (!matstruct->hipSpMV[opA].initialized) { /* built on demand */
        PetscCallHIPSPARSE(hipsparseCreateDnVec(&matstruct->hipSpMV[opA].vecXDescr, nx, xptr, hipsparse_scalartype));
        PetscCallHIPSPARSE(hipsparseCreateDnVec(&matstruct->hipSpMV[opA].vecYDescr, ny, dptr, hipsparse_scalartype));
        PetscCallHIPSPARSE(hipsparseSpMV_bufferSize(hipsparsestruct->handle, opA, matstruct->alpha_one, matstruct->matDescr, matstruct->hipSpMV[opA].vecXDescr, beta, matstruct->hipSpMV[opA].vecYDescr, hipsparse_scalartype, hipsparsestruct->spmvAlg,
                                                    &matstruct->hipSpMV[opA].spmvBufferSize));
        PetscCallHIP(hipMalloc(&matstruct->hipSpMV[opA].spmvBuffer, matstruct->hipSpMV[opA].spmvBufferSize));
        matstruct->hipSpMV[opA].initialized = PETSC_TRUE;
      } else {
        /* x, y's value pointers might change between calls, but their shape is kept, so we just update pointers */
        PetscCallHIPSPARSE(hipsparseDnVecSetValues(matstruct->hipSpMV[opA].vecXDescr, xptr));
        PetscCallHIPSPARSE(hipsparseDnVecSetValues(matstruct->hipSpMV[opA].vecYDescr, dptr));
      }
      PetscCallHIPSPARSE(hipsparseSpMV(hipsparsestruct->handle, opA, matstruct->alpha_one, matstruct->matDescr, /* built in MatSeqAIJHIPSPARSECopyToGPU() or MatSeqAIJHIPSPARSEFormExplicitTranspose() */
                                       matstruct->hipSpMV[opA].vecXDescr, beta, matstruct->hipSpMV[opA].vecYDescr, hipsparse_scalartype, hipsparsestruct->spmvAlg, matstruct->hipSpMV[opA].spmvBuffer));
#else
      CsrMatrix *mat = (CsrMatrix *)matstruct->mat;
      nx             = mat->num_rows; /* nx,ny are set before the #if block, set them again to avoid set-but-not-used warning */
      ny             = mat->num_cols;
      PetscCallHIPSPARSE(hipsparse_csr_spmv(hipsparsestruct->handle, opA, nx, ny, mat->num_entries, matstruct->alpha_one, matstruct->descr, mat->values->data().get(), mat->row_offsets->data().get(), mat->column_indices->data().get(), xptr, beta, dptr));
#endif
    } else {
      if (hipsparsestruct->nrows) {
        hipsparseHybMat_t hybMat = (hipsparseHybMat_t)matstruct->mat;
        PetscCallHIPSPARSE(hipsparse_hyb_spmv(hipsparsestruct->handle, opA, matstruct->alpha_one, matstruct->descr, hybMat, xptr, beta, dptr));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());

    if (opA == HIPSPARSE_OPERATION_NON_TRANSPOSE) {
      if (yy) {                                     /* MatMultAdd: zz = A*xx + yy */
        if (compressed) {                           /* A is compressed. We first copy yy to zz, then ScatterAdd the work vector to zz */
          PetscCall(VecSeq_HIP::Copy(yy, zz));      /* zz = yy */
        } else if (zz != yy) {                      /* A is not compressed. zz already contains A*xx, and we just need to add yy */
          PetscCall(VecSeq_HIP::AXPY(zz, 1.0, yy)); /* zz += yy */
        }
      } else if (compressed) { /* MatMult: zz = A*xx. A is compressed, so we zero zz first, then ScatterAdd the work vector to zz */
        PetscCall(VecSeq_HIP::Set(zz, 0));
      }

      /* ScatterAdd the result from work vector into the full vector when A is compressed */
      if (compressed) {
        PetscCall(PetscLogGpuTimeBegin());
        /* I wanted to make this for_each asynchronous but failed. thrust::async::for_each() returns an event (internally registered)
           and in the destructor of the scope, it will call hipStreamSynchronize() on this stream. One has to store all events to
           prevent that. So I just add a ScatterAdd kernel.
         */
#if 0
        thrust::device_ptr<PetscScalar> zptr = thrust::device_pointer_cast(zarray);
        thrust::async::for_each(thrust::hip::par.on(hipsparsestruct->stream),
                         thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecHIPPlusEquals());
#else
        PetscInt n = matstruct->cprowIndices->size();
        hipLaunchKernelGGL(ScatterAdd, dim3((n + 255) / 256), dim3(256), 0, PetscDefaultHipStream, n, matstruct->cprowIndices->data().get(), hipsparsestruct->workVector->data().get(), zarray);
#endif
        PetscCall(PetscLogGpuTimeEnd());
      }
    } else {
      if (yy && yy != zz) PetscCall(VecSeq_HIP::AXPY(zz, 1.0, yy)); /* zz += yy */
    }
    PetscCall(VecHIPRestoreArrayRead(xx, (const PetscScalar **)&xarray));
    if (yy == zz) PetscCall(VecHIPRestoreArray(zz, &zarray));
    else PetscCall(VecHIPRestoreArrayWrite(zz, &zarray));
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HIPSPARSE error: %s", ex);
  }
  if (yy) PetscCall(PetscLogGpuFlops(2.0 * a->nz));
  else PetscCall(PetscLogGpuFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJHIPSPARSE(A, xx, yy, zz, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJHIPSPARSE(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::AssemblyEnd(A, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSeqAIJHIPSPARSE - Creates a sparse matrix in `MATAIJHIPSPARSE` (compressed row) format.
  This matrix will ultimately pushed down to AMD GPUs and use the HIPSPARSE library for calculations.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. m    - number of rows
. n    - number of columns
. nz   - number of nonzeros per row (same for all rows), ignored if `nnz` is set
- nnz  - array containing the number of nonzeros in the various rows (possibly different for each row) or `NULL`

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  `MatXXXXSetPreallocation()` paradgm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation`]

  The AIJ format (compressed row storage), is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  Specify the preallocated storage with either `nz` or `nnz` (not both).
  Set `nz` = `PETSC_DEFAULT` and `nnz` = `NULL` for PETSc to control dynamic memory
  allocation.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MATSEQAIJHIPSPARSE`, `MATAIJHIPSPARSE`
@*/
PetscErrorCode MatCreateSeqAIJHIPSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A)
{
  return MatSeqAIJHIPSPARSE_CUPM_t::CreateSeqAIJ(comm, m, n, nz, nnz, A);
}

static PetscErrorCode MatDestroy_SeqAIJHIPSPARSE(Mat A)
{
  return MatSeqAIJHIPSPARSE_CUPM_t::Destroy(A);
}

static PetscErrorCode MatDuplicate_SeqAIJHIPSPARSE(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::Duplicate(A, cpvalues, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_SeqAIJ          *x = (Mat_SeqAIJ *)X->data, *y = (Mat_SeqAIJ *)Y->data;
  Mat_SeqAIJHIPSPARSE *cy;
  Mat_SeqAIJHIPSPARSE *cx;
  CsrMatrix           *csry, *csrx;

  PetscFunctionBegin;
  cy = (Mat_SeqAIJHIPSPARSE *)Y->spptr;
  cx = (Mat_SeqAIJHIPSPARSE *)X->spptr;
  if (X->ops->axpy != Y->ops->axpy) {
    PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(Y, PETSC_FALSE));
    PetscCall(MatAXPY_SeqAIJ(Y, a, X, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* if we are here, it means both matrices are bound to GPU */
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(Y));
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(X));
  PetscCheck(cy->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)Y), PETSC_ERR_GPU, "only MAT_HIPSPARSE_CSR supported");
  PetscCheck(cx->format == MAT_HIPSPARSE_CSR, PetscObjectComm((PetscObject)X), PETSC_ERR_GPU, "only MAT_HIPSPARSE_CSR supported");
  csry = (CsrMatrix *)cy->mat->mat;
  csrx = (CsrMatrix *)cx->mat->mat;
  /* see if we can turn this into a cublas axpy */
  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz && !x->compressedrow.use && !y->compressedrow.use) {
    bool eq = thrust::equal(thrust::device, csry->row_offsets->begin(), csry->row_offsets->end(), csrx->row_offsets->begin());
    if (eq) eq = thrust::equal(thrust::device, csry->column_indices->begin(), csry->column_indices->end(), csrx->column_indices->begin());
    if (eq) str = SAME_NONZERO_PATTERN;
  }
  /* spgeam is buggy with one column */
  if (Y->cmap->n == 1 && str != SAME_NONZERO_PATTERN) str = DIFFERENT_NONZERO_PATTERN;

#if !defined(PETSC_USE_64BIT_INDICES) // hipsparseScsrgeam2 etc. do not support 64bit indices
  if (str == SUBSET_NONZERO_PATTERN) {
    PetscScalar       *ay, b = 1.0;
    const PetscScalar *ax;
    size_t             bufferSize;
    void              *buffer;

    PetscCall(MatSeqAIJHIPSPARSEGetArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSEGetArray(Y, &ay));
    PetscCallHIPSPARSE(hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_HOST));
    PetscCallHIPSPARSE(hipsparse_csr_spgeam_bufferSize(cy->handle, Y->rmap->n, Y->cmap->n, &a, cx->mat->descr, x->nz, ax, csrx->row_offsets->data().get(), csrx->column_indices->data().get(), &b, cy->mat->descr, y->nz, ay, csry->row_offsets->data().get(),
                                                       csry->column_indices->data().get(), cy->mat->descr, ay, csry->row_offsets->data().get(), csry->column_indices->data().get(), &bufferSize));
    PetscCallHIP(hipMalloc(&buffer, bufferSize));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallHIPSPARSE(hipsparse_csr_spgeam(cy->handle, Y->rmap->n, Y->cmap->n, &a, cx->mat->descr, x->nz, ax, csrx->row_offsets->data().get(), csrx->column_indices->data().get(), &b, cy->mat->descr, y->nz, ay, csry->row_offsets->data().get(),
                                            csry->column_indices->data().get(), cy->mat->descr, ay, csry->row_offsets->data().get(), csry->column_indices->data().get(), buffer));
    PetscCall(PetscLogGpuFlops(x->nz + y->nz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCallHIP(hipFree(buffer));

    PetscCallHIPSPARSE(hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_DEVICE));
    PetscCall(MatSeqAIJHIPSPARSERestoreArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(Y, &ay));
  } else
#endif
    if (str == SAME_NONZERO_PATTERN) {
    PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::AXPY_SameNZ(Y, a, X));
  } else {
    PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(Y, PETSC_FALSE));
    PetscCall(MatAXPY_SeqAIJ(Y, a, X, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat Y, PetscScalar a)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::Scale(Y, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_SeqAIJHIPSPARSE(Mat A, Vec diag)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::GetDiagonal(A, diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalScale_SeqAIJHIPSPARSE(Mat A, Vec ll, Vec rr)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::DiagonalScale(A, ll, rr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqAIJHIPSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::ZeroEntries(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetCurrentMemType_SeqAIJHIPSPARSE(Mat A, PetscMemType *m)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::GetCurrentMemType(A, m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatBindToCPU_SeqAIJHIPSPARSE(Mat A, PetscBool flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) {
    A->boundtocpu = flg;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (flg) {
    PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));

    A->ops->scale                     = MatScale_SeqAIJ;
    A->ops->getdiagonal               = MatGetDiagonal_SeqAIJ;
    A->ops->diagonalscale             = MatDiagonalScale_SeqAIJ;
    A->ops->axpy                      = MatAXPY_SeqAIJ;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJ;
    A->ops->mult                      = MatMult_SeqAIJ;
    A->ops->multadd                   = MatMultAdd_SeqAIJ;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJ;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJ;
    A->ops->multhermitiantranspose    = NULL;
    A->ops->multhermitiantransposeadd = NULL;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJ;
    A->ops->getcurrentmemtype         = NULL;
    PetscCall(PetscMemzero(a->ops, sizeof(Mat_SeqAIJOps)));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdense_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C", NULL));
  } else {
    A->ops->scale                     = MatScale_SeqAIJHIPSPARSE;
    A->ops->getdiagonal               = MatGetDiagonal_SeqAIJHIPSPARSE;
    A->ops->diagonalscale             = MatDiagonalScale_SeqAIJHIPSPARSE;
    A->ops->axpy                      = MatAXPY_SeqAIJHIPSPARSE;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJHIPSPARSE;
    A->ops->mult                      = MatMult_SeqAIJHIPSPARSE;
    A->ops->multadd                   = MatMultAdd_SeqAIJHIPSPARSE;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJHIPSPARSE;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJHIPSPARSE;
    A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJHIPSPARSE;
    A->ops->getcurrentmemtype         = MatGetCurrentMemType_SeqAIJHIPSPARSE;
    a->ops->getarray                  = MatSeqAIJGetArray_SeqAIJHIPSPARSE;
    a->ops->restorearray              = MatSeqAIJRestoreArray_SeqAIJHIPSPARSE;
    a->ops->getarrayread              = MatSeqAIJGetArrayRead_SeqAIJHIPSPARSE;
    a->ops->restorearrayread          = MatSeqAIJRestoreArrayRead_SeqAIJHIPSPARSE;
    a->ops->getarraywrite             = MatSeqAIJGetArrayWrite_SeqAIJHIPSPARSE;
    a->ops->restorearraywrite         = MatSeqAIJRestoreArrayWrite_SeqAIJHIPSPARSE;
    a->ops->getcsrandmemtype          = MatSeqAIJGetCSRAndMemType_SeqAIJHIPSPARSE;

    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", MatSeqAIJCopySubArray_SeqAIJHIPSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C", MatProductSetFromOptions_SeqAIJHIPSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdense_C", MatProductSetFromOptions_SeqAIJHIPSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_SeqAIJHIPSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_SeqAIJHIPSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C", MatProductSetFromOptions_SeqAIJHIPSPARSE));
  }
  A->boundtocpu = flg;
  a->inode.use  = (flg && a->inode.size_csr) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP)); /* first use of HIPSPARSE may be via MatConvert */
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(A, *newmat, SAME_NONZERO_PATTERN));
  }
  B = *newmat;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECHIP, &B->defaultvectype));
  if (reuse != MAT_REUSE_MATRIX && !B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      Mat_SeqAIJHIPSPARSE *spptr;
      PetscCall(PetscNew(&spptr));
      PetscCallHIPSPARSE(hipsparseCreate(&spptr->handle));
      PetscCallHIPSPARSE(hipsparseSetStream(spptr->handle, PetscDefaultHipStream));
      spptr->format  = MAT_HIPSPARSE_CSR;
      spptr->spmvAlg = HIPSPARSE_SPMV_CSR_ALG1;
      spptr->spmmAlg = HIPSPARSE_SPMM_CSR_ALG1; /* default, only support column-major dense matrix B */
      //spptr->csr2cscAlg = HIPSPARSE_CSR2CSC_ALG1;

      B->spptr = spptr;
    } else {
      Mat_SeqAIJHIPSPARSETriFactors *spptr;

      PetscCall(PetscNew(&spptr));
      PetscCallHIPSPARSE(hipsparseCreate(&spptr->handle));
      PetscCallHIPSPARSE(hipsparseSetStream(spptr->handle, PetscDefaultHipStream));
      B->spptr = spptr;
    }
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  B->ops->assemblyend       = MatAssemblyEnd_SeqAIJHIPSPARSE;
  B->ops->destroy           = MatDestroy_SeqAIJHIPSPARSE;
  B->ops->setoption         = MatSetOption_SeqAIJHIPSPARSE;
  B->ops->setfromoptions    = MatSetFromOptions_SeqAIJHIPSPARSE;
  B->ops->bindtocpu         = MatBindToCPU_SeqAIJHIPSPARSE;
  B->ops->duplicate         = MatDuplicate_SeqAIJHIPSPARSE;
  B->ops->getcurrentmemtype = MatGetCurrentMemType_SeqAIJHIPSPARSE;

  PetscCall(MatBindToCPU_SeqAIJHIPSPARSE(B, PETSC_FALSE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatHIPSPARSESetFormat_C", MatHIPSPARSESetFormat_SeqAIJHIPSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqaijhipsparse_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatHIPSPARSESetUseCPUSolve_C", MatHIPSPARSESetUseCPUSolve_SeqAIJHIPSPARSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJHIPSPARSE(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqAIJ(B));
  PetscCall(MatConvert_SeqAIJ_SeqAIJHIPSPARSE(B, MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQAIJHIPSPARSE - MATAIJHIPSPARSE = "(seq)aijhipsparse" - A matrix type to be used for sparse matrices on AMD GPUs

   A matrix type whose data resides on AMD GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format.
   All matrix calculations are performed on AMD/NVIDIA GPUs using the HIPSPARSE library.

   Options Database Keys:
+  -mat_type aijhipsparse - sets the matrix type to `MATSEQAIJHIPSPARSE`
.  -mat_hipsparse_storage_format csr - sets the storage format of matrices (for `MatMult()` and factors in `MatSolve()`).
                                       Other options include ell (ellpack) or hyb (hybrid).
. -mat_hipsparse_mult_storage_format csr - sets the storage format of matrices (for `MatMult()`). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_use_cpu_solve - Do `MatSolve()` on the CPU

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatCreateSeqAIJHIPSPARSE()`, `MATAIJHIPSPARSE`, `MatCreateAIJHIPSPARSE()`, `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
M*/

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_HIPSPARSE(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_LU, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_ILU, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_ICC, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat mat)
{
  Mat_SeqAIJHIPSPARSE *cusp = static_cast<Mat_SeqAIJHIPSPARSE *>(mat->spptr);

  PetscFunctionBegin;
  if (cusp) {
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&cusp->mat, cusp->format));
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&cusp->matTranspose, cusp->format));
    delete cusp->workVector;
    delete cusp->rowoffsets_gpu;
    delete cusp->csr2csc_i;
    delete cusp->coords;
    if (cusp->handle) PetscCallHIPSPARSE(hipsparseDestroy(cusp->handle));
    PetscCall(PetscFree(mat->spptr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix **mat)
{
  PetscFunctionBegin;
  if (*mat) {
    delete (*mat)->values;
    delete (*mat)->column_indices;
    delete (*mat)->row_offsets;
    delete *mat;
    *mat = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct **matstruct, MatHIPSPARSEStorageFormat format)
{
  CsrMatrix *mat;

  PetscFunctionBegin;
  if (*matstruct) {
    if ((*matstruct)->mat) {
      if (format == MAT_HIPSPARSE_ELL || format == MAT_HIPSPARSE_HYB) {
        hipsparseHybMat_t hybMat = (hipsparseHybMat_t)(*matstruct)->mat;
        PetscCallHIPSPARSE(hipsparseDestroyHybMat(hybMat));
      } else {
        mat = (CsrMatrix *)(*matstruct)->mat;
        PetscCall(CsrMatrix_Destroy(&mat));
      }
    }
    if ((*matstruct)->descr) PetscCallHIPSPARSE(hipsparseDestroyMatDescr((*matstruct)->descr));
    delete (*matstruct)->cprowIndices;
    PetscCallHIP(hipFree((*matstruct)->alpha_one));
    PetscCallHIP(hipFree((*matstruct)->beta_zero));
    PetscCallHIP(hipFree((*matstruct)->beta_one));

    Mat_SeqAIJHIPSPARSEMultStruct *mdata = *matstruct;
    if (mdata->matDescr) PetscCallHIPSPARSE(hipsparseDestroySpMat(mdata->matDescr));
    for (int i = 0; i < 3; i++) {
      if (mdata->hipSpMV[i].initialized) {
        PetscCallHIP(hipFree(mdata->hipSpMV[i].spmvBuffer));
        PetscCallHIPSPARSE(hipsparseDestroyDnVec(mdata->hipSpMV[i].vecXDescr));
        PetscCallHIPSPARSE(hipsparseDestroyDnVec(mdata->hipSpMV[i].vecYDescr));
      }
    }
    delete *matstruct;
    *matstruct = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Reset(Mat_SeqAIJHIPSPARSETriFactors_p *trifactors)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs = *trifactors;

  PetscFunctionBegin;
  if (fs) {
    delete fs->rpermIndices;
    delete fs->cpermIndices;
    fs->rpermIndices  = NULL;
    fs->cpermIndices  = NULL;
    fs->init_dev_prop = PETSC_FALSE;
    PetscCallHIP(hipFree(fs->csrRowPtr));
    PetscCallHIP(hipFree(fs->csrColIdx));
    PetscCallHIP(hipFree(fs->csrRowPtr32));
    PetscCallHIP(hipFree(fs->csrColIdx32));
    PetscCallHIP(hipFree(fs->csrVal));
    PetscCallHIP(hipFree(fs->diag));
    PetscCallHIP(hipFree(fs->X));
    PetscCallHIP(hipFree(fs->Y));
    // PetscCallHIP(hipFree(fs->factBuffer_M)); /* No needed since factBuffer_M shares with one of spsvBuffer_L/U */
    PetscCallHIP(hipFree(fs->spsvBuffer_L));
    PetscCallHIP(hipFree(fs->spsvBuffer_U));
    PetscCallHIP(hipFree(fs->spsvBuffer_Lt));
    PetscCallHIP(hipFree(fs->spsvBuffer_Ut));
    PetscCallHIPSPARSE(hipsparseDestroyMatDescr(fs->matDescr_M));
    if (fs->spMatDescr_L) PetscCallHIPSPARSE(hipsparseDestroySpMat(fs->spMatDescr_L));
    if (fs->spMatDescr_U) PetscCallHIPSPARSE(hipsparseDestroySpMat(fs->spMatDescr_U));
    PetscCallHIPSPARSE(hipsparseSpSV_destroyDescr(fs->spsvDescr_L));
    PetscCallHIPSPARSE(hipsparseSpSV_destroyDescr(fs->spsvDescr_Lt));
    PetscCallHIPSPARSE(hipsparseSpSV_destroyDescr(fs->spsvDescr_U));
    PetscCallHIPSPARSE(hipsparseSpSV_destroyDescr(fs->spsvDescr_Ut));
    if (fs->dnVecDescr_X) PetscCallHIPSPARSE(hipsparseDestroyDnVec(fs->dnVecDescr_X));
    if (fs->dnVecDescr_Y) PetscCallHIPSPARSE(hipsparseDestroyDnVec(fs->dnVecDescr_Y));
    PetscCallHIPSPARSE(hipsparseDestroyCsrilu02Info(fs->ilu0Info_M));
    PetscCallHIPSPARSE(hipsparseDestroyCsric02Info(fs->ic0Info_M));
    PetscCall(PetscFree(fs->csrRowPtr_h));
    PetscCall(PetscFree(fs->csrVal_h));
    PetscCall(PetscFree(fs->diag_h));
    fs->createdTransposeSpSVDescr    = PETSC_FALSE;
    fs->updatedTransposeSpSVAnalysis = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors **trifactors)
{
  PetscFunctionBegin;
  if (*trifactors) {
    PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(trifactors));
    PetscCallHIPSPARSE(hipsparseDestroy((*trifactors)->handle));
    PetscCall(PetscFree(*trifactors));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat A, PetscBool destroy)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  if (!cusp) PetscFunctionReturn(PETSC_SUCCESS);
  if (destroy) {
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&cusp->matTranspose, cusp->format));
    delete cusp->csr2csc_i;
    cusp->csr2csc_i = NULL;
  }
  A->transupdated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::SetPreallocationCOO(mat, coo_n, coo_i, coo_j));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::SetValuesCOO(A, v, imode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJHIPSPARSEGetIJ - returns the device row storage `i` and `j` indices for `MATSEQAIJHIPSPARSE` matrices.

  Not Collective

  Input Parameters:
+ A          - the matrix
- compressed - `PETSC_TRUE` or `PETSC_FALSE` indicating the matrix data structure should be always returned in compressed form

  Output Parameters:
+ i - the CSR row pointers
- j - the CSR column indices

  Level: developer

  Note:
  When compressed is true, the CSR structure does not contain empty rows

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSERestoreIJ()`, `MatSeqAIJHIPSPARSEGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetIJ(Mat A, PetscBool compressed, const PetscInt *i[], const PetscInt *j[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::GetIJ(A, compressed, i, j));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJHIPSPARSERestoreIJ - restore the device row storage `i` and `j` indices obtained with `MatSeqAIJHIPSPARSEGetIJ()`

  Not Collective

  Input Parameters:
+ A          - the matrix
. compressed - `PETSC_TRUE` or `PETSC_FALSE` indicating the matrix data structure should be always returned in compressed form
. i          - the CSR row pointers
- j          - the CSR column indices

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetIJ()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreIJ(Mat A, PetscBool compressed, const PetscInt *i[], const PetscInt *j[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::RestoreIJ(A, compressed, i, j));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatSeqAIJHIPSPARSEGetArrayRead - gives read-only access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

  Not Collective

  Input Parameter:
. A - a `MATSEQAIJHIPSPARSE` matrix

  Output Parameter:
. a - pointer to the device data

  Level: developer

  Note:
  May trigger host-device copies if the up-to-date matrix data is on host

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`, `MatSeqAIJHIPSPARSEGetArrayWrite()`, `MatSeqAIJHIPSPARSERestoreArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArrayRead(Mat A, const PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::GetArrayRead(A, a);
}

/*@C
  MatSeqAIJHIPSPARSERestoreArrayRead - restore the read-only access array obtained from `MatSeqAIJHIPSPARSEGetArrayRead()`

  Not Collective

  Input Parameters:
+ A - a `MATSEQAIJHIPSPARSE` matrix
- a - pointer to the device data

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayRead(Mat A, const PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::RestoreArrayRead(A, a);
}

/*@C
  MatSeqAIJHIPSPARSEGetArray - gives read-write access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

  Not Collective

  Input Parameter:
. A - a `MATSEQAIJHIPSPARSE` matrix

  Output Parameter:
. a - pointer to the device data

  Level: developer

  Note:
  May trigger host-device copies if up-to-date matrix data is on host

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayRead()`, `MatSeqAIJHIPSPARSEGetArrayWrite()`, `MatSeqAIJHIPSPARSERestoreArray()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArray(Mat A, PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::GetArray(A, a);
}
/*@C
  MatSeqAIJHIPSPARSERestoreArray - restore the read-write access array obtained from `MatSeqAIJHIPSPARSEGetArray()`

  Not Collective

  Input Parameters:
+ A - a `MATSEQAIJHIPSPARSE` matrix
- a - pointer to the device data

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArray(Mat A, PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::RestoreArray(A, a);
}

/*@C
  MatSeqAIJHIPSPARSEGetArrayWrite - gives write access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

  Not Collective

  Input Parameter:
. A - a `MATSEQAIJHIPSPARSE` matrix

  Output Parameter:
. a - pointer to the device data

  Level: developer

  Note:
  Does not trigger host-device copies and flags data validity on the GPU

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`, `MatSeqAIJHIPSPARSEGetArrayRead()`, `MatSeqAIJHIPSPARSERestoreArrayWrite()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArrayWrite(Mat A, PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::GetArrayWrite(A, a);
}

/*@C
  MatSeqAIJHIPSPARSERestoreArrayWrite - restore the write-only access array obtained from `MatSeqAIJHIPSPARSEGetArrayWrite()`

  Not Collective

  Input Parameters:
+ A - a `MATSEQAIJHIPSPARSE` matrix
- a - pointer to the device data

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayWrite()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayWrite(Mat A, PetscScalar *a[])
{
  return MatSeqAIJHIPSPARSE_CUPM_t::RestoreArrayWrite(A, a);
}

struct IJCompare4 {
  __host__ __device__ inline bool operator()(const thrust::tuple<int, int, PetscScalar, int> &t1, const thrust::tuple<int, int, PetscScalar, int> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct Shift {
  int _shift;

  Shift(int shift) : _shift(shift) { }
  __host__ __device__ inline int operator()(const int &c) { return c + _shift; }
};

/* merges two SeqAIJHIPSPARSE matrices A, B by concatenating their rows. [A';B']' operation in MATLAB notation */
PetscErrorCode MatSeqAIJHIPSPARSEMergeMats(Mat A, Mat B, MatReuse reuse, Mat *C)
{
  Mat_SeqAIJ                    *a = (Mat_SeqAIJ *)A->data, *b = (Mat_SeqAIJ *)B->data, *c;
  Mat_SeqAIJHIPSPARSE           *Acusp = (Mat_SeqAIJHIPSPARSE *)A->spptr, *Bcusp = (Mat_SeqAIJHIPSPARSE *)B->spptr, *Ccusp;
  Mat_SeqAIJHIPSPARSEMultStruct *Cmat;
  CsrMatrix                     *Acsr, *Bcsr, *Ccsr;
  PetscInt                       Annz, Bnnz;
  PetscInt                       i, m, n, zero = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscAssertPointer(C, 4);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCheckTypeName(B, MATSEQAIJHIPSPARSE);
  PetscCheck(A->rmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT, A->rmap->n, B->rmap->n);
  PetscCheck(reuse != MAT_INPLACE_MATRIX, PETSC_COMM_SELF, PETSC_ERR_SUP, "MAT_INPLACE_MATRIX not supported");
  PetscCheck(Acusp->format != MAT_HIPSPARSE_ELL && Acusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCheck(Bcusp->format != MAT_HIPSPARSE_ELL && Bcusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  if (reuse == MAT_INITIAL_MATRIX) {
    m = A->rmap->n;
    n = A->cmap->n + B->cmap->n;
    PetscCall(MatCreate(PETSC_COMM_SELF, C));
    PetscCall(MatSetSizes(*C, m, n, m, n));
    PetscCall(MatSetType(*C, MATSEQAIJHIPSPARSE));
    c                       = (Mat_SeqAIJ *)(*C)->data;
    Ccusp                   = (Mat_SeqAIJHIPSPARSE *)(*C)->spptr;
    Cmat                    = new Mat_SeqAIJHIPSPARSEMultStruct;
    Ccsr                    = new CsrMatrix;
    Cmat->cprowIndices      = NULL;
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Ccusp->workVector       = NULL;
    Ccusp->nrows            = m;
    Ccusp->mat              = Cmat;
    Ccusp->mat->mat         = Ccsr;
    Ccsr->num_rows          = m;
    Ccsr->num_cols          = n;
    PetscCallHIPSPARSE(hipsparseCreateMatDescr(&Cmat->descr));
    PetscCallHIPSPARSE(hipsparseSetMatIndexBase(Cmat->descr, HIPSPARSE_INDEX_BASE_ZERO));
    PetscCallHIPSPARSE(hipsparseSetMatType(Cmat->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
    PetscCallHIP(hipMalloc((void **)&Cmat->alpha_one, sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&Cmat->beta_zero, sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&Cmat->beta_one, sizeof(PetscScalar)));
    PetscCallHIP(hipMemcpy(Cmat->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(Cmat->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(Cmat->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
    PetscCall(MatSeqAIJHIPSPARSECopyToGPU(B));
    PetscCheck(Acusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
    PetscCheck(Bcusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");

    Acsr                 = (CsrMatrix *)Acusp->mat->mat;
    Bcsr                 = (CsrMatrix *)Bcusp->mat->mat;
    Annz                 = (PetscInt)Acsr->column_indices->size();
    Bnnz                 = (PetscInt)Bcsr->column_indices->size();
    c->nz                = Annz + Bnnz;
    Ccsr->row_offsets    = new THRUSTINTARRAY(m + 1);
    Ccsr->column_indices = new THRUSTINTARRAY(c->nz);
    Ccsr->values         = new THRUSTARRAY(c->nz);
    Ccsr->num_entries    = c->nz;
    Ccusp->coords        = new THRUSTINTARRAY(c->nz);
    if (c->nz) {
      auto            Acoo = new THRUSTINTARRAY(Annz); // initialized with zeros
      auto            Bcoo = new THRUSTINTARRAY(Bnnz);
      auto            Ccoo = new THRUSTINTARRAY(c->nz);
      THRUSTINTARRAY *Aroff, *Broff;

      if (a->compressedrow.use) { /* need full row offset */
        if (!Acusp->rowoffsets_gpu) {
          Acusp->rowoffsets_gpu = new THRUSTINTARRAY(A->rmap->n + 1);
          Acusp->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
        }
        Aroff = Acusp->rowoffsets_gpu;
      } else Aroff = Acsr->row_offsets;
      if (b->compressedrow.use) { /* need full row offset */
        if (!Bcusp->rowoffsets_gpu) {
          Bcusp->rowoffsets_gpu = new THRUSTINTARRAY(B->rmap->n + 1);
          Bcusp->rowoffsets_gpu->assign(b->i, b->i + B->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((B->rmap->n + 1) * sizeof(PetscInt)));
        }
        Broff = Bcusp->rowoffsets_gpu;
      } else Broff = Bcsr->row_offsets;
      PetscCall(PetscLogGpuTimeBegin());
      // Implement cusparseXcsr2coo() with Thrust, as the former doesn't support 64-bit indices.
      PetscCallThrust(thrust::for_each(thrust::device, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(m), Csr2coo(Aroff->data().get(), Acoo->data().get())));
      PetscCallThrust(thrust::for_each(thrust::device, thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(m), Csr2coo(Broff->data().get(), Bcoo->data().get())));

      /* Issues when using bool with large matrices on SUMMIT 10.2.89 */
      auto Aperm = thrust::make_constant_iterator(1);
      auto Bperm = thrust::make_constant_iterator(0);
      auto Bcib  = thrust::make_transform_iterator(Bcsr->column_indices->begin(), Shift(A->cmap->n));
      auto Bcie  = thrust::make_transform_iterator(Bcsr->column_indices->end(), Shift(A->cmap->n));
      auto wPerm = new THRUSTINTARRAY(Annz + Bnnz);
      auto Azb   = thrust::make_zip_iterator(thrust::make_tuple(Acoo->begin(), Acsr->column_indices->begin(), Acsr->values->begin(), Aperm));
      auto Aze   = thrust::make_zip_iterator(thrust::make_tuple(Acoo->end(), Acsr->column_indices->end(), Acsr->values->end(), Aperm));
      auto Bzb   = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->begin(), Bcib, Bcsr->values->begin(), Bperm)); // Use B column indices shifted by A->cmap->n
      auto Bze   = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->end(), Bcie, Bcsr->values->end(), Bperm));
      auto Czb   = thrust::make_zip_iterator(thrust::make_tuple(Ccoo->begin(), Ccsr->column_indices->begin(), Ccsr->values->begin(), wPerm->begin()));
      auto p1    = Ccusp->coords->begin();
      auto p2    = Ccusp->coords->begin();
      thrust::advance(p2, Annz);
      PetscCallThrust(thrust::merge(thrust::device, Azb, Aze, Bzb, Bze, Czb, IJCompare4())); // put nonzeros in A and B to C in sorted order (by row and then by column)
      auto cci  = thrust::make_counting_iterator(zero);
      auto cce  = thrust::make_counting_iterator(c->nz);
      auto pred = [](const int &x) { return x; };
      PetscCallThrust(thrust::copy_if(thrust::device, cci, cce, wPerm->begin(), p1, pred));
      PetscCallThrust(thrust::remove_copy_if(thrust::device, cci, cce, wPerm->begin(), p2, pred));
      // Implement a simplified hipsparseXcoo2csr() with Thrust (assuming the row indices are already sorted), as the former doesn't support 64-bit indices.
      PetscCallThrust(thrust::lower_bound(thrust::device, Ccoo->begin(), Ccoo->end(), thrust::counting_iterator<PetscInt>(0), thrust::counting_iterator<PetscInt>(m + 1), Ccsr->row_offsets->begin()));
      PetscCall(PetscLogGpuTimeEnd());
      delete wPerm;
      delete Acoo;
      delete Bcoo;
      delete Ccoo;
      PetscCallHIPSPARSE(hipsparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, Ccsr->num_entries, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get(), csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
      if (A->form_explicit_transpose && B->form_explicit_transpose) { /* if A and B have the transpose, generate C transpose too */
        PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(A));
        PetscCall(MatSeqAIJHIPSPARSEFormExplicitTranspose(B));
        PetscBool                      AT = Acusp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bcusp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        Mat_SeqAIJHIPSPARSEMultStruct *CmatT = new Mat_SeqAIJHIPSPARSEMultStruct;
        CsrMatrix                     *CcsrT = new CsrMatrix;
        CsrMatrix                     *AcsrT = AT ? (CsrMatrix *)Acusp->matTranspose->mat : NULL;
        CsrMatrix                     *BcsrT = BT ? (CsrMatrix *)Bcusp->matTranspose->mat : NULL;

        (*C)->form_explicit_transpose = PETSC_TRUE;
        (*C)->transupdated            = PETSC_TRUE;
        Ccusp->rowoffsets_gpu         = NULL;
        CmatT->cprowIndices           = NULL;
        CmatT->mat                    = CcsrT;
        CcsrT->num_rows               = n;
        CcsrT->num_cols               = m;
        CcsrT->num_entries            = c->nz;

        CcsrT->row_offsets    = new THRUSTINTARRAY(n + 1);
        CcsrT->column_indices = new THRUSTINTARRAY(c->nz);
        CcsrT->values         = new THRUSTARRAY(c->nz);

        PetscCall(PetscLogGpuTimeBegin());
        auto rT = CcsrT->row_offsets->begin();
        if (AT) {
          rT = thrust::copy(AcsrT->row_offsets->begin(), AcsrT->row_offsets->end(), rT);
          thrust::advance(rT, -1);
        }
        if (BT) {
          auto titb = thrust::make_transform_iterator(BcsrT->row_offsets->begin(), Shift(a->nz));
          auto tite = thrust::make_transform_iterator(BcsrT->row_offsets->end(), Shift(a->nz));
          thrust::copy(titb, tite, rT);
        }
        auto cT = CcsrT->column_indices->begin();
        if (AT) cT = thrust::copy(AcsrT->column_indices->begin(), AcsrT->column_indices->end(), cT);
        if (BT) thrust::copy(BcsrT->column_indices->begin(), BcsrT->column_indices->end(), cT);
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(), AcsrT->values->end(), vT);
        if (BT) thrust::copy(BcsrT->values->begin(), BcsrT->values->end(), vT);
        PetscCall(PetscLogGpuTimeEnd());

        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&CmatT->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(CmatT->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(CmatT->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallHIP(hipMalloc((void **)&CmatT->alpha_one, sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&CmatT->beta_zero, sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&CmatT->beta_one, sizeof(PetscScalar)));
        PetscCallHIP(hipMemcpy(CmatT->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(CmatT->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(CmatT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIPSPARSE(hipsparseCreateCsr(&CmatT->matDescr, CcsrT->num_rows, CcsrT->num_cols, CcsrT->num_entries, CcsrT->row_offsets->data().get(), CcsrT->column_indices->data().get(), CcsrT->values->data().get(), csrRowOffsetsType, csrColIndType, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
        Ccusp->matTranspose = CmatT;
      }
    }

    c->free_a = PETSC_TRUE;
    PetscCall(PetscShmgetAllocateArray(c->nz, sizeof(PetscInt), (void **)&c->j));
    PetscCall(PetscShmgetAllocateArray(m + 1, sizeof(PetscInt), (void **)&c->i));
    c->free_ij = PETSC_TRUE;
    PetscCallHIP(hipMemcpy(c->i, Ccsr->row_offsets->data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    PetscCallHIP(hipMemcpy(c->j, Ccsr->column_indices->data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    PetscCall(PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size()) * sizeof(PetscInt)));
    PetscCall(PetscMalloc1(m, &c->ilen));
    PetscCall(PetscMalloc1(m, &c->imax));
    c->maxnz         = c->nz;
    c->nonzerorowcnt = 0;
    c->rmax          = 0;
    for (i = 0; i < m; i++) {
      const PetscInt nn = c->i[i + 1] - c->i[i];
      c->ilen[i] = c->imax[i] = nn;
      c->nonzerorowcnt += (PetscInt)!!nn;
      c->rmax = PetscMax(c->rmax, nn);
    }
    PetscCall(PetscMalloc1(c->nz, &c->a));
    (*C)->nonzerostate++;
    PetscCall(PetscLayoutSetUp((*C)->rmap));
    PetscCall(PetscLayoutSetUp((*C)->cmap));
    Ccusp->nonzerostate = (*C)->nonzerostate;
    (*C)->preallocated  = PETSC_TRUE;
  } else {
    PetscCheck((*C)->rmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT, (*C)->rmap->n, B->rmap->n);
    c = (Mat_SeqAIJ *)(*C)->data;
    if (c->nz) {
      Ccusp = (Mat_SeqAIJHIPSPARSE *)(*C)->spptr;
      PetscCheck(Ccusp->coords, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing coords");
      PetscCheck(Ccusp->format != MAT_HIPSPARSE_ELL && Ccusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
      PetscCheck(Ccusp->nonzerostate == (*C)->nonzerostate, PETSC_COMM_SELF, PETSC_ERR_COR, "Wrong nonzerostate");
      PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
      PetscCall(MatSeqAIJHIPSPARSECopyToGPU(B));
      PetscCheck(Acusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
      PetscCheck(Bcusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
      Acsr = (CsrMatrix *)Acusp->mat->mat;
      Bcsr = (CsrMatrix *)Bcusp->mat->mat;
      Ccsr = (CsrMatrix *)Ccusp->mat->mat;
      PetscCheck(Acsr->num_entries == (PetscInt)Acsr->values->size(), PETSC_COMM_SELF, PETSC_ERR_COR, "A nnz %" PetscInt_FMT " != %" PetscInt_FMT, Acsr->num_entries, (PetscInt)Acsr->values->size());
      PetscCheck(Bcsr->num_entries == (PetscInt)Bcsr->values->size(), PETSC_COMM_SELF, PETSC_ERR_COR, "B nnz %" PetscInt_FMT " != %" PetscInt_FMT, Bcsr->num_entries, (PetscInt)Bcsr->values->size());
      PetscCheck(Ccsr->num_entries == (PetscInt)Ccsr->values->size(), PETSC_COMM_SELF, PETSC_ERR_COR, "C nnz %" PetscInt_FMT " != %" PetscInt_FMT, Ccsr->num_entries, (PetscInt)Ccsr->values->size());
      PetscCheck(Ccsr->num_entries == Acsr->num_entries + Bcsr->num_entries, PETSC_COMM_SELF, PETSC_ERR_COR, "C nnz %" PetscInt_FMT " != %" PetscInt_FMT " + %" PetscInt_FMT, Ccsr->num_entries, Acsr->num_entries, Bcsr->num_entries);
      PetscCheck(Ccusp->coords->size() == Ccsr->values->size(), PETSC_COMM_SELF, PETSC_ERR_COR, "permSize %" PetscInt_FMT " != %" PetscInt_FMT, (PetscInt)Ccusp->coords->size(), (PetscInt)Ccsr->values->size());
      auto pmid = Ccusp->coords->begin();
      thrust::advance(pmid, Acsr->num_entries);
      PetscCall(PetscLogGpuTimeBegin());
      auto zibait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->begin(), thrust::make_permutation_iterator(Ccsr->values->begin(), Ccusp->coords->begin())));
      auto zieait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->end(), thrust::make_permutation_iterator(Ccsr->values->begin(), pmid)));
      thrust::for_each(zibait, zieait, VecHIPEquals());
      auto zibbit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->begin(), thrust::make_permutation_iterator(Ccsr->values->begin(), pmid)));
      auto ziebit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->end(), thrust::make_permutation_iterator(Ccsr->values->begin(), Ccusp->coords->end())));
      thrust::for_each(zibbit, ziebit, VecHIPEquals());
      PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(*C, PETSC_FALSE));
      if (A->form_explicit_transpose && B->form_explicit_transpose && (*C)->form_explicit_transpose) {
        PetscCheck(Ccusp->matTranspose, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing transpose Mat_SeqAIJHIPSPARSEMultStruct");
        PetscBool  AT = Acusp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bcusp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        CsrMatrix *AcsrT = AT ? (CsrMatrix *)Acusp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix *)Bcusp->matTranspose->mat : NULL;
        CsrMatrix *CcsrT = (CsrMatrix *)Ccusp->matTranspose->mat;
        auto       vT    = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(), AcsrT->values->end(), vT);
        if (BT) thrust::copy(BcsrT->values->begin(), BcsrT->values->end(), vT);
        (*C)->transupdated = PETSC_TRUE;
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)*C));
  (*C)->assembled     = PETSC_TRUE;
  (*C)->was_assembled = PETSC_FALSE;
  (*C)->offloadmask   = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJHIPSPARSE(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSE_CUPM_t::CopySubArray(A, n, idx, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
