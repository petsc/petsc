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

const char *const MatHIPSPARSEStorageFormats[] = {"CSR", "ELL", "HYB", "MatHIPSPARSEStorageFormat", "MAT_HIPSPARSE_", 0};
const char *const MatHIPSPARSESpMVAlgorithms[] = {"MV_ALG_DEFAULT", "COOMV_ALG", "CSRMV_ALG1", "CSRMV_ALG2", "SPMV_ALG_DEFAULT", "SPMV_COO_ALG1", "SPMV_COO_ALG2", "SPMV_CSR_ALG1", "SPMV_CSR_ALG2", "hipsparseSpMVAlg_t", "HIPSPARSE_", 0};
const char *const MatHIPSPARSESpMMAlgorithms[] = {"ALG_DEFAULT", "COO_ALG1", "COO_ALG2", "COO_ALG3", "CSR_ALG1", "COO_ALG4", "CSR_ALG2", "hipsparseSpMMAlg_t", "HIPSPARSE_SPMM_", 0};
//const char *const MatHIPSPARSECsr2CscAlgorithms[] = {"INVALID"/*HIPSPARSE does not have enum 0! We created one*/, "ALG1", "ALG2", "hipsparseCsr2CscAlg_t", "HIPSPARSE_CSR2CSC_", 0};

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, IS, const MatFactorInfo *);
static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat, Mat, IS, IS, const MatFactorInfo *);
static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat, Vec, Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat, Vec, Vec);
static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(Mat, PetscOptionItems *PetscOptionsObject);
static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat, PetscScalar, Mat, MatStructure);
static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat, PetscScalar);
static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultHermitianTranspose_SeqAIJHIPSPARSE(Mat, Vec, Vec);
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec);
static PetscErrorCode MatMultAddKernel_SeqAIJHIPSPARSE(Mat, Vec, Vec, Vec, PetscBool, PetscBool);
static PetscErrorCode CsrMatrix_Destroy(CsrMatrix **);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct **);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct **, MatHIPSPARSEStorageFormat);
static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors **);
static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE **);
static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat);
static PetscErrorCode MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(Mat);
static PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat, PetscBool);
static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJHIPSPARSE(Mat, PetscInt, const PetscInt[], PetscScalar[]);
static PetscErrorCode MatBindToCPU_SeqAIJHIPSPARSE(Mat, PetscBool);
static PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat, PetscCount, PetscInt[], PetscInt[]);
static PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat, const PetscScalar[], InsertMode);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat, MatType, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse_band(Mat, MatFactorType, Mat *);

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
+  A - Matrix of type `MATSEQAIJHIPSPARSE`
.  op - `MatHIPSPARSEFormatOperation`. `MATSEQAIJHIPSPARSE` matrices support `MAT_HIPSPARSE_MULT` and `MAT_HIPSPARSE_ALL`.
         `MATMPIAIJHIPSPARSE` matrices support `MAT_HIPSPARSE_MULT_DIAG`, `MAT_HIPSPARSE_MULT_OFFDIAG`, and `MAT_HIPSPARSE_ALL`.
-  format - `MatHIPSPARSEStorageFormat` (one of `MAT_HIPSPARSE_CSR`, `MAT_HIPSPARSE_ELL`, `MAT_HIPSPARSE_HYB`.)

   Level: intermediate

.seealso: [](chapter_matrices), `Mat`, `Mat`, `MATSEQAIJHIPSPARSE`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
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
+  A - Matrix of type `MATSEQAIJHIPSPARSE`
-  use_cpu - set flag for using the built-in CPU `MatSolve()`

   Level: intermediate

   Notes:
   The hipSparse LU solver currently computes the factors with the built-in CPU method
   and moves the factors to the GPU for the solve. We have observed better performance keeping the data on the CPU and computing the solve there.
   This method to specifies if the solve is done on the CPU or GPU (GPU is the default).

.seealso: [](chapter_matrices), `Mat`, `MatSolve()`, `MATSEQAIJHIPSPARSE`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
@*/
PetscErrorCode MatHIPSPARSESetUseCPUSolve(Mat A, PetscBool use_cpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscTryMethod(A, "MatHIPSPARSESetUseCPUSolve_C", (Mat, PetscBool), (A, use_cpu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetOption_SeqAIJHIPSPARSE(Mat A, MatOption op, PetscBool flg)
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

static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscBool            row_identity, col_identity;
  Mat_SeqAIJ          *b     = (Mat_SeqAIJ *)B->data;
  IS                   isrow = b->row, iscol = b->col;
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  PetscCall(MatLUFactorNumeric_SeqAIJ(B, A, info));
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  PetscCall(ISIdentity(isrow, &row_identity));
  PetscCall(ISIdentity(iscol, &col_identity));
  if (!hipsparsestruct->use_cpu_solve) {
    if (row_identity && col_identity) {
      B->ops->solve          = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
      B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    } else {
      B->ops->solve          = MatSolve_SeqAIJHIPSPARSE;
      B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE;
    }
  }
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;

  /* get the triangular factors */
  if (!hipsparsestruct->use_cpu_solve) { PetscCall(MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(B)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(Mat A, PetscOptionItems *PetscOptionsObject)
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

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                         *a                   = (Mat_SeqAIJ *)A->data;
  PetscInt                            n                   = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtr;
  const PetscInt                     *ai = a->i, *aj = a->j, *vi;
  const MatScalar                    *aa = a->a, *v;
  PetscInt                           *AiLo, *AjLo;
  PetscInt                            i, nz, nzLower, offset, rowOffset;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower = n + ai[n] - ai[1];
      if (!loTriFactor) {
        PetscScalar *AALo;
        PetscCallHIP(hipHostMalloc((void **)&AALo, nzLower * sizeof(PetscScalar)));

        /* Allocate Space for the lower triangular matrix */
        PetscCallHIP(hipHostMalloc((void **)&AiLo, (n + 1) * sizeof(PetscInt)));
        PetscCallHIP(hipHostMalloc((void **)&AjLo, nzLower * sizeof(PetscInt)));

        /* Fill the lower triangular matrix */
        AiLo[0]   = (PetscInt)0;
        AiLo[n]   = nzLower;
        AjLo[0]   = (PetscInt)0;
        AALo[0]   = (MatScalar)1.0;
        v         = aa;
        vi        = aj;
        offset    = 1;
        rowOffset = 1;
        for (i = 1; i < n; i++) {
          nz = ai[i + 1] - ai[i];
          /* additional 1 for the term on the diagonal */
          AiLo[i] = rowOffset;
          rowOffset += nz + 1;

          PetscCall(PetscArraycpy(&(AjLo[offset]), vi, nz));
          PetscCall(PetscArraycpy(&(AALo[offset]), v, nz));
          offset += nz;
          AjLo[offset] = (PetscInt)i;
          AALo[offset] = (MatScalar)1.0;
          offset += 1;
          v += nz;
          vi += nz;
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&loTriFactor));
        loTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
        /* Create the matrix description */
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&loTriFactor->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallHIPSPARSE(hipsparseSetMatFillMode(loTriFactor->descr, HIPSPARSE_FILL_MODE_LOWER));
        PetscCallHIPSPARSE(hipsparseSetMatDiagType(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT));

        /* set the operation */
        loTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat                 = new CsrMatrix;
        loTriFactor->csrMat->num_rows       = n;
        loTriFactor->csrMat->num_cols       = n;
        loTriFactor->csrMat->num_entries    = nzLower;
        loTriFactor->csrMat->row_offsets    = new THRUSTINTARRAY32(n + 1);
        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzLower);
        loTriFactor->csrMat->values         = new THRUSTARRAY(nzLower);

        loTriFactor->csrMat->row_offsets->assign(AiLo, AiLo + n + 1);
        loTriFactor->csrMat->column_indices->assign(AjLo, AjLo + nzLower);
        loTriFactor->csrMat->values->assign(AALo, AALo + nzLower);

        /* Create the solve analysis information */
        PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
        PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&loTriFactor->solveInfo));
        PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                                    loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, &loTriFactor->solveBufferSize));
        PetscCallHIP(hipMalloc(&loTriFactor->solveBuffer, loTriFactor->solveBufferSize));

        /* perform the solve analysis */
        PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                                    loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, loTriFactor->solvePolicy, loTriFactor->solveBuffer));

        PetscCallHIP(WaitForHIP());
        PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->loTriFactorPtr = loTriFactor;
        loTriFactor->AA_h                                           = AALo;
        PetscCallHIP(hipHostFree(AiLo));
        PetscCallHIP(hipHostFree(AjLo));
        PetscCall(PetscLogCpuToGpu((n + 1 + nzLower) * sizeof(int) + nzLower * sizeof(PetscScalar)));
      } else { /* update values only */
        if (!loTriFactor->AA_h) PetscCallHIP(hipHostMalloc((void **)&loTriFactor->AA_h, nzLower * sizeof(PetscScalar)));
        /* Fill the lower triangular matrix */
        loTriFactor->AA_h[0] = 1.0;
        v                    = aa;
        vi                   = aj;
        offset               = 1;
        for (i = 1; i < n; i++) {
          nz = ai[i + 1] - ai[i];
          PetscCall(PetscArraycpy(&(loTriFactor->AA_h[offset]), v, nz));
          offset += nz;
          loTriFactor->AA_h[offset] = 1.0;
          offset += 1;
          v += nz;
        }
        loTriFactor->csrMat->values->assign(loTriFactor->AA_h, loTriFactor->AA_h + nzLower);
        PetscCall(PetscLogCpuToGpu(nzLower * sizeof(PetscScalar)));
      }
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ                         *a                   = (Mat_SeqAIJ *)A->data;
  PetscInt                            n                   = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtr;
  const PetscInt                     *aj = a->j, *adiag = a->diag, *vi;
  const MatScalar                    *aa = a->a, *v;
  PetscInt                           *AiUp, *AjUp;
  PetscInt                            i, nz, nzUpper, offset;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0] - adiag[n];
      if (!upTriFactor) {
        PetscScalar *AAUp;
        PetscCallHIP(hipHostMalloc((void **)&AAUp, nzUpper * sizeof(PetscScalar)));

        /* Allocate Space for the upper triangular matrix */
        PetscCallHIP(hipHostMalloc((void **)&AiUp, (n + 1) * sizeof(PetscInt)));
        PetscCallHIP(hipHostMalloc((void **)&AjUp, nzUpper * sizeof(PetscInt)));

        /* Fill the upper triangular matrix */
        AiUp[0] = (PetscInt)0;
        AiUp[n] = nzUpper;
        offset  = nzUpper;
        for (i = n - 1; i >= 0; i--) {
          v  = aa + adiag[i + 1] + 1;
          vi = aj + adiag[i + 1] + 1;
          nz = adiag[i] - adiag[i + 1] - 1; /* number of elements NOT on the diagonal */
          offset -= (nz + 1);               /* decrement the offset */

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt)i;
          AAUp[offset] = (MatScalar)1. / v[nz];
          AiUp[i]      = AiUp[i + 1] - (nz + 1);

          PetscCall(PetscArraycpy(&(AjUp[offset + 1]), vi, nz));
          PetscCall(PetscArraycpy(&(AAUp[offset + 1]), v, nz));
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&upTriFactor));
        upTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&upTriFactor->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallHIPSPARSE(hipsparseSetMatFillMode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER));
        PetscCallHIPSPARSE(hipsparseSetMatDiagType(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT));

        /* set the operation */
        upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        upTriFactor->csrMat                 = new CsrMatrix;
        upTriFactor->csrMat->num_rows       = n;
        upTriFactor->csrMat->num_cols       = n;
        upTriFactor->csrMat->num_entries    = nzUpper;
        upTriFactor->csrMat->row_offsets    = new THRUSTINTARRAY32(n + 1);
        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzUpper);
        upTriFactor->csrMat->values         = new THRUSTARRAY(nzUpper);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp + n + 1);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp + nzUpper);
        upTriFactor->csrMat->values->assign(AAUp, AAUp + nzUpper);

        /* Create the solve analysis information */
        PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
        PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&upTriFactor->solveInfo));
        PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                                    upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, &upTriFactor->solveBufferSize));
        PetscCallHIP(hipMalloc(&upTriFactor->solveBuffer, upTriFactor->solveBufferSize));

        /* perform the solve analysis */
        PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                                    upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, upTriFactor->solvePolicy, upTriFactor->solveBuffer));

        PetscCallHIP(WaitForHIP());
        PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->upTriFactorPtr = upTriFactor;
        upTriFactor->AA_h                                           = AAUp;
        PetscCallHIP(hipHostFree(AiUp));
        PetscCallHIP(hipHostFree(AjUp));
        PetscCall(PetscLogCpuToGpu((n + 1 + nzUpper) * sizeof(int) + nzUpper * sizeof(PetscScalar)));
      } else {
        if (!upTriFactor->AA_h) PetscCallHIP(hipHostMalloc((void **)&upTriFactor->AA_h, nzUpper * sizeof(PetscScalar)));
        /* Fill the upper triangular matrix */
        offset = nzUpper;
        for (i = n - 1; i >= 0; i--) {
          v  = aa + adiag[i + 1] + 1;
          nz = adiag[i] - adiag[i + 1] - 1; /* number of elements NOT on the diagonal */
          offset -= (nz + 1);               /* decrement the offset */

          /* first, set the diagonal elements */
          upTriFactor->AA_h[offset] = 1. / v[nz];
          PetscCall(PetscArraycpy(&(upTriFactor->AA_h[offset + 1]), v, nz));
        }
        upTriFactor->csrMat->values->assign(upTriFactor->AA_h, upTriFactor->AA_h + nzUpper);
        PetscCall(PetscLogCpuToGpu(nzUpper * sizeof(PetscScalar)));
      }
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  PetscBool                      row_identity, col_identity;
  Mat_SeqAIJ                    *a                   = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  IS                             isrow = a->row, iscol = a->icol;
  PetscInt                       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");
  PetscCall(MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(A));
  PetscCall(MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(A));

  if (!hipsparseTriFactors->workVector) hipsparseTriFactors->workVector = new THRUSTARRAY(n);
  hipsparseTriFactors->nnz = a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH;
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
  PetscCall(ISIdentity(iscol, &col_identity));
  if (!col_identity && !hipsparseTriFactors->cpermIndices) {
    const PetscInt *c;

    PetscCall(ISGetIndices(iscol, &c));
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(c, c + n);
    PetscCall(ISRestoreIndices(iscol, &c));
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                         *a                   = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtr;
  PetscInt                           *AiUp, *AjUp;
  PetscScalar                        *AAUp;
  PetscScalar                        *AALo;
  PetscInt                            nzUpper = a->nz, n = A->rmap->n, i, offset, nz, j;
  Mat_SeqSBAIJ                       *b  = (Mat_SeqSBAIJ *)A->data;
  const PetscInt                     *ai = b->i, *aj = b->j, *vj;
  const MatScalar                    *aa = b->a, *v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      PetscCallHIP(hipHostMalloc((void **)&AAUp, nzUpper * sizeof(PetscScalar)));
      PetscCallHIP(hipHostMalloc((void **)&AALo, nzUpper * sizeof(PetscScalar)));
      if (!upTriFactor && !loTriFactor) {
        /* Allocate Space for the upper triangular matrix */
        PetscCallHIP(hipHostMalloc((void **)&AiUp, (n + 1) * sizeof(PetscInt)));
        PetscCallHIP(hipHostMalloc((void **)&AjUp, nzUpper * sizeof(PetscInt)));

        /* Fill the upper triangular matrix */
        AiUp[0] = (PetscInt)0;
        AiUp[n] = nzUpper;
        offset  = 0;
        for (i = 0; i < n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          vj = aj + ai[i];
          nz = ai[i + 1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt)i;
          AAUp[offset] = (MatScalar)1.0 / v[nz];
          AiUp[i]      = offset;
          AALo[offset] = (MatScalar)1.0 / v[nz];

          offset += 1;
          if (nz > 0) {
            PetscCall(PetscArraycpy(&(AjUp[offset]), vj, nz));
            PetscCall(PetscArraycpy(&(AAUp[offset]), v, nz));
            for (j = offset; j < offset + nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j] / v[nz];
            }
            offset += nz;
          }
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&upTriFactor));
        upTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&upTriFactor->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallHIPSPARSE(hipsparseSetMatFillMode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER));
        PetscCallHIPSPARSE(hipsparseSetMatDiagType(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT));

        /* set the matrix */
        upTriFactor->csrMat                 = new CsrMatrix;
        upTriFactor->csrMat->num_rows       = A->rmap->n;
        upTriFactor->csrMat->num_cols       = A->cmap->n;
        upTriFactor->csrMat->num_entries    = a->nz;
        upTriFactor->csrMat->row_offsets    = new THRUSTINTARRAY32(A->rmap->n + 1);
        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        upTriFactor->csrMat->values         = new THRUSTARRAY(a->nz);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp + A->rmap->n + 1);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp + a->nz);
        upTriFactor->csrMat->values->assign(AAUp, AAUp + a->nz);

        /* set the operation */
        upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* Create the solve analysis information */
        PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
        PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&upTriFactor->solveInfo));
        PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                                    upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, &upTriFactor->solveBufferSize));
        PetscCallHIP(hipMalloc(&upTriFactor->solveBuffer, upTriFactor->solveBufferSize));

        /* perform the solve analysis */
        PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                                    upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, upTriFactor->solvePolicy, upTriFactor->solveBuffer));

        PetscCallHIP(WaitForHIP());
        PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->upTriFactorPtr = upTriFactor;

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&loTriFactor));
        loTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&loTriFactor->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallHIPSPARSE(hipsparseSetMatFillMode(loTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER));
        PetscCallHIPSPARSE(hipsparseSetMatDiagType(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT));

        /* set the operation */
        loTriFactor->solveOp = HIPSPARSE_OPERATION_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat                 = new CsrMatrix;
        loTriFactor->csrMat->num_rows       = A->rmap->n;
        loTriFactor->csrMat->num_cols       = A->cmap->n;
        loTriFactor->csrMat->num_entries    = a->nz;
        loTriFactor->csrMat->row_offsets    = new THRUSTINTARRAY32(A->rmap->n + 1);
        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        loTriFactor->csrMat->values         = new THRUSTARRAY(a->nz);
        loTriFactor->csrMat->row_offsets->assign(AiUp, AiUp + A->rmap->n + 1);
        loTriFactor->csrMat->column_indices->assign(AjUp, AjUp + a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo + a->nz);

        /* Create the solve analysis information */
        PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
        PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&loTriFactor->solveInfo));
        PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                                    loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, &loTriFactor->solveBufferSize));
        PetscCallHIP(hipMalloc(&loTriFactor->solveBuffer, loTriFactor->solveBufferSize));

        /* perform the solve analysis */
        PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                                    loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, loTriFactor->solvePolicy, loTriFactor->solveBuffer));

        PetscCallHIP(WaitForHIP());
        PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->loTriFactorPtr = loTriFactor;

        PetscCall(PetscLogCpuToGpu(2 * (((A->rmap->n + 1) + (a->nz)) * sizeof(int) + (a->nz) * sizeof(PetscScalar))));
        PetscCallHIP(hipHostFree(AiUp));
        PetscCallHIP(hipHostFree(AjUp));
      } else {
        /* Fill the upper triangular matrix */
        offset = 0;
        for (i = 0; i < n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          nz = ai[i + 1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AAUp[offset] = 1.0 / v[nz];
          AALo[offset] = 1.0 / v[nz];

          offset += 1;
          if (nz > 0) {
            PetscCall(PetscArraycpy(&(AAUp[offset]), v, nz));
            for (j = offset; j < offset + nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j] / v[nz];
            }
            offset += nz;
          }
        }
        PetscCheck(upTriFactor, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");
        PetscCheck(loTriFactor, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");
        upTriFactor->csrMat->values->assign(AAUp, AAUp + a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo + a->nz);
        PetscCall(PetscLogCpuToGpu(2 * (a->nz) * sizeof(PetscScalar)));
      }
      PetscCallHIP(hipHostFree(AAUp));
      PetscCallHIP(hipHostFree(AALo));
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  PetscBool                      perm_identity;
  Mat_SeqAIJ                    *a                   = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  IS                             ip                  = a->row;
  PetscInt                       n                   = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");
  PetscCall(MatSeqAIJHIPSPARSEBuildICCTriMatrices(A));
  if (!hipsparseTriFactors->workVector) hipsparseTriFactors->workVector = new THRUSTARRAY(n);
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
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(rip, rip + n);
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
  PetscBool   perm_identity;
  Mat_SeqAIJ *b  = (Mat_SeqAIJ *)B->data;
  IS          ip = b->row;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  PetscCall(MatCholeskyFactorNumeric_SeqAIJ(B, A, info));
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  PetscCall(ISIdentity(ip, &perm_identity));
  if (perm_identity) {
    B->ops->solve             = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->solvetranspose    = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->matsolve          = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve             = MatSolve_SeqAIJHIPSPARSE;
    B->ops->solvetranspose    = MatSolveTranspose_SeqAIJHIPSPARSE;
    B->ops->matsolve          = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  PetscCall(MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT;
  hipsparseIndexBase_t                indexBase;
  hipsparseMatrixType_t               matrixType;
  hipsparseFillMode_t                 fillMode;
  hipsparseDiagType_t                 diagType;

  PetscFunctionBegin;
  /* allocate space for the transpose of the lower triangular factor */
  PetscCall(PetscNew(&loTriFactorT));
  loTriFactorT->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the lower triangular factor */
  matrixType = hipsparseGetMatType(loTriFactor->descr);
  indexBase  = hipsparseGetMatIndexBase(loTriFactor->descr);
  fillMode   = hipsparseGetMatFillMode(loTriFactor->descr) == HIPSPARSE_FILL_MODE_UPPER ? HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType   = hipsparseGetMatDiagType(loTriFactor->descr);

  /* Create the matrix description */
  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&loTriFactorT->descr));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(loTriFactorT->descr, indexBase));
  PetscCallHIPSPARSE(hipsparseSetMatType(loTriFactorT->descr, matrixType));
  PetscCallHIPSPARSE(hipsparseSetMatFillMode(loTriFactorT->descr, fillMode));
  PetscCallHIPSPARSE(hipsparseSetMatDiagType(loTriFactorT->descr, diagType));

  /* set the operation */
  loTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the lower triangular factor*/
  loTriFactorT->csrMat                 = new CsrMatrix;
  loTriFactorT->csrMat->num_rows       = loTriFactor->csrMat->num_cols;
  loTriFactorT->csrMat->num_cols       = loTriFactor->csrMat->num_rows;
  loTriFactorT->csrMat->num_entries    = loTriFactor->csrMat->num_entries;
  loTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_rows + 1);
  loTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_entries);
  loTriFactorT->csrMat->values         = new THRUSTARRAY(loTriFactorT->csrMat->num_entries);

  /* compute the transpose of the lower triangular factor, i.e. the CSC */
  /* Csr2cscEx2 is not implemented in ROCm-5.2.0 and is planned for implementation in hipsparse with future releases of ROCm
#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PetscCallHIPSPARSE(hipsparseCsr2cscEx2_bufferSize(hipsparseTriFactors->handle, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries, loTriFactor->csrMat->values->data().get(),
                                                  loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                                  loTriFactorT->csrMat->column_indices->data().get(), hipsparse_scalartype, HIPSPARSE_ACTION_NUMERIC, indexBase, HIPSPARSE_CSR2CSC_ALG1, &loTriFactor->csr2cscBufferSize));
  PetscCallHIP(hipMalloc(&loTriFactor->csr2cscBuffer, loTriFactor->csr2cscBufferSize));
#endif
*/
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));

  PetscCallHIPSPARSE(hipsparse_csr2csc(hipsparseTriFactors->handle, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries, loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(), loTriFactorT->csrMat->values->data().get(),
#if 0 /* when Csr2cscEx2 is implemented in hipSparse PETSC_PKG_HIP_VERSION_GE(5, 2, 0)*/
                          loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(),
                          hipsparse_scalartype, HIPSPARSE_ACTION_NUMERIC, indexBase, HIPSPARSE_CSR2CSC_ALG1, loTriFactor->csr2cscBuffer));
#else
                                       loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->csrMat->row_offsets->data().get(), HIPSPARSE_ACTION_NUMERIC, indexBase));
#endif

  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));

  /* Create the solve analysis information */
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
  PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&loTriFactorT->solveInfo));
  PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, loTriFactorT->solveOp, loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr, loTriFactorT->csrMat->values->data().get(),
                                              loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo, &loTriFactorT->solveBufferSize));
  PetscCallHIP(hipMalloc(&loTriFactorT->solveBuffer, loTriFactorT->solveBufferSize));

  /* perform the solve analysis */
  PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, loTriFactorT->solveOp, loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr, loTriFactorT->csrMat->values->data().get(),
                                              loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo, loTriFactorT->solvePolicy, loTriFactorT->solveBuffer));

  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->loTriFactorPtrTranspose = loTriFactorT;

  /*********************************************/
  /* Now the Transpose of the Upper Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the upper triangular factor */
  PetscCall(PetscNew(&upTriFactorT));
  upTriFactorT->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the upper triangular factor */
  matrixType = hipsparseGetMatType(upTriFactor->descr);
  indexBase  = hipsparseGetMatIndexBase(upTriFactor->descr);
  fillMode   = hipsparseGetMatFillMode(upTriFactor->descr) == HIPSPARSE_FILL_MODE_UPPER ? HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType   = hipsparseGetMatDiagType(upTriFactor->descr);

  /* Create the matrix description */
  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&upTriFactorT->descr));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(upTriFactorT->descr, indexBase));
  PetscCallHIPSPARSE(hipsparseSetMatType(upTriFactorT->descr, matrixType));
  PetscCallHIPSPARSE(hipsparseSetMatFillMode(upTriFactorT->descr, fillMode));
  PetscCallHIPSPARSE(hipsparseSetMatDiagType(upTriFactorT->descr, diagType));

  /* set the operation */
  upTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the upper triangular factor*/
  upTriFactorT->csrMat                 = new CsrMatrix;
  upTriFactorT->csrMat->num_rows       = upTriFactor->csrMat->num_cols;
  upTriFactorT->csrMat->num_cols       = upTriFactor->csrMat->num_rows;
  upTriFactorT->csrMat->num_entries    = upTriFactor->csrMat->num_entries;
  upTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_rows + 1);
  upTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_entries);
  upTriFactorT->csrMat->values         = new THRUSTARRAY(upTriFactorT->csrMat->num_entries);

  /* compute the transpose of the upper triangular factor, i.e. the CSC */
  /* Csr2cscEx2 is not implemented in ROCm-5.2.0 and is planned for implementation in hipsparse with future releases of ROCm
#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PetscCallHIPSPARSE(hipsparseCsr2cscEx2_bufferSize(hipsparseTriFactors->handle, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries, upTriFactor->csrMat->values->data().get(),
                                                  upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                                  upTriFactorT->csrMat->column_indices->data().get(), hipsparse_scalartype, HIPSPARSE_ACTION_NUMERIC, indexBase, HIPSPARSE_CSR2CSC_ALG1, &upTriFactor->csr2cscBufferSize));
  PetscCallHIP(hipMalloc(&upTriFactor->csr2cscBuffer, upTriFactor->csr2cscBufferSize));
#endif
*/
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));
  PetscCallHIPSPARSE(hipsparse_csr2csc(hipsparseTriFactors->handle, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries, upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                       upTriFactor->csrMat->column_indices->data().get(), upTriFactorT->csrMat->values->data().get(),
#if 0 /* when Csr2cscEx2 is implemented in hipSparse PETSC_PKG_HIP_VERSION_GE(5, 2, 0)*/
                          upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(),
                          hipsparse_scalartype, HIPSPARSE_ACTION_NUMERIC, indexBase, HIPSPARSE_CSR2CSC_ALG1, upTriFactor->csr2cscBuffer));
#else
                                       upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->csrMat->row_offsets->data().get(), HIPSPARSE_ACTION_NUMERIC, indexBase));
#endif

  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose, A, 0, 0, 0));

  /* Create the solve analysis information */
  PetscCall(PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));
  PetscCallHIPSPARSE(hipsparseCreateCsrsvInfo(&upTriFactorT->solveInfo));
  PetscCallHIPSPARSE(hipsparseXcsrsv_buffsize(hipsparseTriFactors->handle, upTriFactorT->solveOp, upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr, upTriFactorT->csrMat->values->data().get(),
                                              upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo, &upTriFactorT->solveBufferSize));
  PetscCallHIP(hipMalloc(&upTriFactorT->solveBuffer, upTriFactorT->solveBufferSize));

  /* perform the solve analysis */
  PetscCallHIPSPARSE(hipsparseXcsrsv_analysis(hipsparseTriFactors->handle, upTriFactorT->solveOp, upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr, upTriFactorT->csrMat->values->data().get(),
                                              upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo, upTriFactorT->solvePolicy, upTriFactorT->solveBuffer));

  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis, A, 0, 0, 0));

  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSETriFactors *)A->spptr)->upTriFactorPtrTranspose = upTriFactorT;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct PetscScalarToPetscInt {
  __host__ __device__ PetscInt operator()(PetscScalar s) { return (PetscInt)PetscRealPart(s); }
};

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
    PetscCallHIP(hipMalloc((void **)&(matstructT->alpha_one), sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&(matstructT->beta_zero), sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&(matstructT->beta_one), sizeof(PetscScalar)));
    PetscCallHIP(hipMemcpy(matstructT->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(matstructT->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
    PetscCallHIP(hipMemcpy(matstructT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));

    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      CsrMatrix *matrixT      = new CsrMatrix;
      matstructT->mat         = matrixT;
      matrixT->num_rows       = A->cmap->n;
      matrixT->num_cols       = A->rmap->n;
      matrixT->num_entries    = a->nz;
      matrixT->row_offsets    = new THRUSTINTARRAY32(matrixT->num_rows + 1);
      matrixT->column_indices = new THRUSTINTARRAY32(a->nz);
      matrixT->values         = new THRUSTARRAY(a->nz);

      if (!hipsparsestruct->rowoffsets_gpu) hipsparsestruct->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n + 1);
      hipsparsestruct->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);

      PetscCallHIPSPARSE(hipsparseCreateCsr(&matstructT->matDescr, matrixT->num_rows, matrixT->num_cols, matrixT->num_entries, matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), matrixT->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, /* row offset, col idx type due to THRUSTINTARRAY32 */
                                            indexBase, hipsparse_scalartype));
    } else if (hipsparsestruct->format == MAT_HIPSPARSE_ELL || hipsparsestruct->format == MAT_HIPSPARSE_HYB) {
      CsrMatrix *temp  = new CsrMatrix;
      CsrMatrix *tempT = new CsrMatrix;
      /* First convert HYB to CSR */
      temp->num_rows       = A->rmap->n;
      temp->num_cols       = A->cmap->n;
      temp->num_entries    = a->nz;
      temp->row_offsets    = new THRUSTINTARRAY32(A->rmap->n + 1);
      temp->column_indices = new THRUSTINTARRAY32(a->nz);
      temp->values         = new THRUSTARRAY(a->nz);

      PetscCallHIPSPARSE(hipsparse_hyb2csr(hipsparsestruct->handle, matstruct->descr, (hipsparseHybMat_t)matstruct->mat, temp->values->data().get(), temp->row_offsets->data().get(), temp->column_indices->data().get()));

      /* Next, convert CSR to CSC (i.e. the matrix transpose) */
      tempT->num_rows       = A->rmap->n;
      tempT->num_cols       = A->cmap->n;
      tempT->num_entries    = a->nz;
      tempT->row_offsets    = new THRUSTINTARRAY32(A->rmap->n + 1);
      tempT->column_indices = new THRUSTINTARRAY32(a->nz);
      tempT->values         = new THRUSTARRAY(a->nz);

      PetscCallHIPSPARSE(hipsparse_csr2csc(hipsparsestruct->handle, temp->num_rows, temp->num_cols, temp->num_entries, temp->values->data().get(), temp->row_offsets->data().get(), temp->column_indices->data().get(), tempT->values->data().get(),
                                           tempT->column_indices->data().get(), tempT->row_offsets->data().get(), HIPSPARSE_ACTION_NUMERIC, indexBase));

      /* Last, convert CSC to HYB */
      hipsparseHybMat_t hybMat;
      PetscCallHIPSPARSE(hipsparseCreateHybMat(&hybMat));
      hipsparseHybPartition_t partition = hipsparsestruct->format == MAT_HIPSPARSE_ELL ? HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
      PetscCallHIPSPARSE(hipsparse_csr2hyb(hipsparsestruct->handle, A->rmap->n, A->cmap->n, matstructT->descr, tempT->values->data().get(), tempT->row_offsets->data().get(), tempT->column_indices->data().get(), hybMat, 0, partition));

      /* assign the pointer */
      matstructT->mat = hybMat;
      A->transupdated = PETSC_TRUE;
      /* delete temporaries */
      if (tempT) {
        if (tempT->values) delete (THRUSTARRAY *)tempT->values;
        if (tempT->column_indices) delete (THRUSTINTARRAY32 *)tempT->column_indices;
        if (tempT->row_offsets) delete (THRUSTINTARRAY32 *)tempT->row_offsets;
        delete (CsrMatrix *)tempT;
      }
      if (temp) {
        if (temp->values) delete (THRUSTARRAY *)temp->values;
        if (temp->column_indices) delete (THRUSTINTARRAY32 *)temp->column_indices;
        if (temp->row_offsets) delete (THRUSTINTARRAY32 *)temp->row_offsets;
        delete (CsrMatrix *)temp;
      }
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
      hipsparsestruct->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n + 1);
      hipsparsestruct->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
      PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
    }
    if (!hipsparsestruct->csr2csc_i) {
      THRUSTARRAY csr2csc_a(matrix->num_entries);
      PetscCallThrust(thrust::sequence(thrust::device, csr2csc_a.begin(), csr2csc_a.end(), 0.0));

      indexBase = hipsparseGetMatIndexBase(matstruct->descr);
      if (matrix->num_entries) {
        /* This routine is known to give errors with CUDA-11, but works fine with CUDA-10
           Need to verify this for ROCm.
        */
        PetscCallHIPSPARSE(hipsparse_csr2csc(hipsparsestruct->handle, A->rmap->n, A->cmap->n, matrix->num_entries, csr2csc_a.data().get(), hipsparsestruct->rowoffsets_gpu->data().get(), matrix->column_indices->data().get(), matrixT->values->data().get(),
                                             matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(), HIPSPARSE_ACTION_NUMERIC, indexBase));
      } else {
        matrixT->row_offsets->assign(matrixT->row_offsets->size(), indexBase);
      }

      hipsparsestruct->csr2csc_i = new THRUSTINTARRAY(matrix->num_entries);
      PetscCallThrust(thrust::transform(thrust::device, matrixT->values->begin(), matrixT->values->end(), hipsparsestruct->csr2csc_i->begin(), PetscScalarToPetscInt()));
    }
    PetscCallThrust(
      thrust::copy(thrust::device, thrust::make_permutation_iterator(matrix->values->begin(), hipsparsestruct->csr2csc_i->begin()), thrust::make_permutation_iterator(matrix->values->begin(), hipsparsestruct->csr2csc_i->end()), matrixT->values->begin()));
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

/* Why do we need to analyze the transposed matrix again? Can't we just use op(A) = HIPSPARSE_OPERATION_TRANSPOSE in MatSolve_SeqAIJHIPSPARSE? */
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat A, Vec bb, Vec xx)
{
  PetscInt                              n = xx->map->n;
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  Mat_SeqAIJHIPSPARSETriFactors        *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct   *loTriFactorT        = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct   *upTriFactorT        = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                          *tempGPU             = (THRUSTARRAY *)hipsparseTriFactors->workVector;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    PetscCall(MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A));
    loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  PetscCall(VecHIPGetArrayWrite(xx, &xarray));
  PetscCall(VecHIPGetArrayRead(bb, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  PetscCall(PetscLogGpuTimeBegin());
  /* First, reorder with the row permutation */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU + n, hipsparseTriFactors->rpermIndices->end()), xGPU);

  /* First, solve U */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp, upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, upTriFactorT->descr, upTriFactorT->csrMat->values->data().get(),
                                           upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo, xarray, tempGPU->data().get(), upTriFactorT->solvePolicy, upTriFactorT->solveBuffer));

  /* Then, solve L */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp, loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, loTriFactorT->descr, loTriFactorT->csrMat->values->data().get(),
                                           loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo, tempGPU->data().get(), xarray, loTriFactorT->solvePolicy, loTriFactorT->solveBuffer));

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(xGPU, hipsparseTriFactors->cpermIndices->begin()), thrust::make_permutation_iterator(xGPU + n, hipsparseTriFactors->cpermIndices->end()), tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  PetscCall(VecHIPRestoreArrayRead(bb, &barray));
  PetscCall(VecHIPRestoreArrayWrite(xx, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * hipsparseTriFactors->nnz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  const PetscScalar                  *barray;
  PetscScalar                        *xarray;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT        = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT        = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                        *tempGPU             = (THRUSTARRAY *)hipsparseTriFactors->workVector;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    PetscCall(MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A));
    loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  PetscCall(VecHIPGetArrayWrite(xx, &xarray));
  PetscCall(VecHIPGetArrayRead(bb, &barray));

  PetscCall(PetscLogGpuTimeBegin());
  /* First, solve U */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp, upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, upTriFactorT->descr, upTriFactorT->csrMat->values->data().get(),
                                           upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo, barray, tempGPU->data().get(), upTriFactorT->solvePolicy, upTriFactorT->solveBuffer));

  /* Then, solve L */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp, loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, loTriFactorT->descr, loTriFactorT->csrMat->values->data().get(),
                                           loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo, tempGPU->data().get(), xarray, loTriFactorT->solvePolicy, loTriFactorT->solveBuffer));

  /* restore */
  PetscCall(VecHIPRestoreArrayRead(bb, &barray));
  PetscCall(VecHIPRestoreArrayWrite(xx, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * hipsparseTriFactors->nnz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE(Mat A, Vec bb, Vec xx)
{
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  Mat_SeqAIJHIPSPARSETriFactors        *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct   *loTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct   *upTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                          *tempGPU             = (THRUSTARRAY *)hipsparseTriFactors->workVector;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  PetscCall(VecHIPGetArrayWrite(xx, &xarray));
  PetscCall(VecHIPGetArrayRead(bb, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  PetscCall(PetscLogGpuTimeBegin());
  /* First, reorder with the row permutation */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->end()), tempGPU->begin());

  /* Next, solve L */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                           loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, tempGPU->data().get(), xarray, loTriFactor->solvePolicy, loTriFactor->solveBuffer));

  /* Then, solve U */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                           upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, xarray, tempGPU->data().get(), upTriFactor->solvePolicy, upTriFactor->solveBuffer));

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->begin()), thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->end()), xGPU);

  PetscCall(VecHIPRestoreArrayRead(bb, &barray));
  PetscCall(VecHIPRestoreArrayWrite(xx, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * hipsparseTriFactors->nnz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  const PetscScalar                  *barray;
  PetscScalar                        *xarray;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor         = (Mat_SeqAIJHIPSPARSETriFactorStruct *)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                        *tempGPU             = (THRUSTARRAY *)hipsparseTriFactors->workVector;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  PetscCall(VecHIPGetArrayWrite(xx, &xarray));
  PetscCall(VecHIPGetArrayRead(bb, &barray));

  PetscCall(PetscLogGpuTimeBegin());
  /* First, solve L */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, loTriFactor->solveOp, loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, loTriFactor->descr, loTriFactor->csrMat->values->data().get(),
                                           loTriFactor->csrMat->row_offsets->data().get(), loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo, barray, tempGPU->data().get(), loTriFactor->solvePolicy, loTriFactor->solveBuffer));

  /* Next, solve U */
  PetscCallHIPSPARSE(hipsparseXcsrsv_solve(hipsparseTriFactors->handle, upTriFactor->solveOp, upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, &PETSC_HIPSPARSE_ONE, upTriFactor->descr, upTriFactor->csrMat->values->data().get(),
                                           upTriFactor->csrMat->row_offsets->data().get(), upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo, tempGPU->data().get(), xarray, upTriFactor->solvePolicy, upTriFactor->solveBuffer));

  PetscCall(VecHIPRestoreArrayRead(bb, &barray));
  PetscCall(VecHIPRestoreArrayWrite(xx, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * hipsparseTriFactors->nnz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
/* hipsparseSpSV_solve() and related functions first appeared in ROCm-4.5.0*/
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_ILU0(Mat fact, Vec b, Vec x)
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
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L,                                     /* L Y = X */
                                         fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L)); // hipsparseSpSV_solve() secretely uses the external buffer used in hipsparseSpSV_analysis()!

  /* Solve U*x = y */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, xarray));
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, /* U X = Y */
                                         fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, fs->spsvBuffer_U));

  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));

  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * aij->nz - fact->rmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_ILU0(Mat fact, Vec b, Vec x)
{
  Mat_SeqAIJHIPSPARSETriFactors *fs  = (Mat_SeqAIJHIPSPARSETriFactors *)fact->spptr;
  Mat_SeqAIJ                    *aij = (Mat_SeqAIJ *)fact->data;
  const PetscScalar             *barray;
  PetscScalar                   *xarray;

  PetscFunctionBegin;
  if (!fs->createdTransposeSpSVDescr) { /* Call MatSolveTranspose() for the first time */
    PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Lt));
    PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* The matrix is still L. We only do transpose solve with it */
                                                fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, &fs->spsvBufferSize_Lt));

    PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Ut));
    PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Ut, &fs->spsvBufferSize_Ut));
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Lt, fs->spsvBufferSize_Lt));
    PetscCallHIP(hipMalloc((void **)&fs->spsvBuffer_Ut, fs->spsvBufferSize_Ut));
    fs->createdTransposeSpSVDescr = PETSC_TRUE;
  }

  if (!fs->updatedTransposeSpSVAnalysis) {
    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));

    PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));
    fs->updatedTransposeSpSVAnalysis = PETSC_TRUE;
  }

  PetscCall(VecHIPGetArrayWrite(x, &xarray));
  PetscCall(VecHIPGetArrayRead(b, &barray));
  PetscCall(PetscLogGpuTimeBegin());

  /* Solve Ut*y = b */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, (void *)barray));
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_Y, fs->Y));
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, /* Ut Y = X */
                                         fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Ut, fs->spsvBuffer_Ut));

  /* Solve Lt*x = y */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, xarray));
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* Lt X = Y */
                                         fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));

  PetscCall(VecHIPRestoreArrayRead(b, &barray));
  PetscCall(VecHIPRestoreArrayWrite(x, &xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0 * aij->nz - fact->rmap->n));
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

  /* Factorize fact inplace */
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                          fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ilu0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    int               numerical_zero;
    hipsparseStatus_t status;
    status = hipsparseXcsrilu02_zeroPivot(fs->handle, fs->ilu0Info_M, &numerical_zero);
    PetscAssert(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Numerical zero pivot detected in csrilu02: A(%d,%d) is zero", numerical_zero, numerical_zero);
  }

  /* hipsparseSpSV_analysis() is numeric, i.e., it requires valid matrix values, therefore, we do it after hipsparseXcsrilu02() */
  PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

  PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, fs->spsvBuffer_U));

  /* L, U values have changed, reset the flag to indicate we need to redo hipsparseSpSV_analysis() for transpose solve */
  fs->updatedTransposeSpSVAnalysis = PETSC_FALSE;

  fact->offloadmask            = PETSC_OFFLOAD_GPU;
  fact->ops->solve             = MatSolve_SeqAIJHIPSPARSE_ILU0;
  fact->ops->solvetranspose    = MatSolveTranspose_SeqAIJHIPSPARSE_ILU0;
  fact->ops->matsolve          = NULL;
  fact->ops->matsolvetranspose = NULL;
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
    PetscInt  i;
    PetscBool flg, missing;

    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
    PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
    PetscCall(MatMissingDiagonal(A, &missing, &i));
    PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, i);
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
  const int *Ai, *Aj;

  m  = fact->rmap->n;
  nz = aij->nz;

  PetscCallHIP(hipMalloc((void **)&fs->csrRowPtr, sizeof(int) * (m + 1)));
  PetscCallHIP(hipMalloc((void **)&fs->csrColIdx, sizeof(int) * nz));
  PetscCallHIP(hipMalloc((void **)&fs->csrVal, sizeof(PetscScalar) * nz));
  PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &Ai, &Aj)); /* Do not use compressed Ai */
  PetscCallHIP(hipMemcpyAsync(fs->csrRowPtr, Ai, sizeof(int) * (m + 1), hipMemcpyDeviceToDevice, PetscDefaultHipStream));
  PetscCallHIP(hipMemcpyAsync(fs->csrColIdx, Aj, sizeof(int) * nz, hipMemcpyDeviceToDevice, PetscDefaultHipStream));

  /* ====================================================================== */
  /* Create descriptors for M, L, U                                         */
  /* ====================================================================== */
  hipsparseFillMode_t fillMode;
  hipsparseDiagType_t diagType;

  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&fs->matDescr_M));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(fs->matDescr_M, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(fs->matDescr_M, HIPSPARSE_MATRIX_TYPE_GENERAL));

  /* https://docs.amd.com/bundle/hipSPARSE-Documentation---hipSPARSE-documentation/page/usermanual.html/#hipsparse_8h_1a79e036b6c0680cb37e2aa53d3542a054
    hipsparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
    assumed to be present, but if HIPSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
    all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
    assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
  */
  fillMode = HIPSPARSE_FILL_MODE_LOWER;
  diagType = HIPSPARSE_DIAG_TYPE_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_L, m, m, nz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  fillMode = HIPSPARSE_FILL_MODE_UPPER;
  diagType = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_U, m, m, nz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_U, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  /* ========================================================================= */
  /* Query buffer sizes for csrilu0, SpSV and allocate buffers                 */
  /* ========================================================================= */
  PetscCallHIPSPARSE(hipsparseCreateCsrilu02Info(&fs->ilu0Info_M));
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02_bufferSize(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                                     fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ilu0Info_M, &fs->factBufferSize_M));

  PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(PetscScalar) * m));
  PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(PetscScalar) * m));

  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_L));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, &fs->spsvBufferSize_L));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_U));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_U, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_U, &fs->spsvBufferSize_U));

  /* It appears spsvBuffer_L/U can not be shared (i.e., the same) for our case, but factBuffer_M can share with either of spsvBuffer_L/U.
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
  int structural_zero;

  fs->policy_M = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  if (m)
    PetscCallHIPSPARSE(hipsparseXcsrilu02_analysis(fs->handle, m, nz, /* hipsparseXcsrilu02 errors out with empty matrices (m=0) */
                                                   fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ilu0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    /* Function hipsparseXcsrilu02_zeroPivot() is a blocking call. It calls hipDeviceSynchronize() to make sure all previous kernels are done. */
    hipsparseStatus_t status;
    status = hipsparseXcsrilu02_zeroPivot(fs->handle, fs->ilu0Info_M, &structural_zero);
    PetscCheck(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Structural zero pivot detected in csrilu02: A(%d,%d) is missing", structural_zero, structural_zero);
  }

  /* Estimate FLOPs of the numeric factorization */
  {
    Mat_SeqAIJ    *Aseq = (Mat_SeqAIJ *)A->data;
    PetscInt      *Ai, *Adiag, nzRow, nzLeft;
    PetscLogDouble flops = 0.0;

    PetscCall(MatMarkDiagonal_SeqAIJ(A));
    Ai    = Aseq->i;
    Adiag = Aseq->diag;
    for (PetscInt i = 0; i < m; i++) {
      if (Ai[i] < Adiag[i] && Adiag[i] < Ai[i + 1]) { /* There are nonzeros left to the diagonal of row i */
        nzRow  = Ai[i + 1] - Ai[i];
        nzLeft = Adiag[i] - Ai[i];
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
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* L Y = X */
                                         fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

  /* Solve Lt*x = y */
  PetscCallHIPSPARSE(hipsparseDnVecSetValues(fs->dnVecDescr_X, xarray));
  PetscCallHIPSPARSE(hipsparseSpSV_solve(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, /* Lt X = Y */
                                         fs->dnVecDescr_Y, fs->dnVecDescr_X, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));

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
  /* Function csric02() only takes the lower triangular part of matrix A to perform factorization.
     The matrix type must be HIPSPARSE_MATRIX_TYPE_GENERAL, the fill mode and diagonal type are ignored,
     and the strictly upper triangular part is ignored and never touched. It does not matter if A is Hermitian or not.
     In other words, from the point of view of csric02() A is Hermitian and only the lower triangular part is provided.
   */
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ic0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    int               numerical_zero;
    hipsparseStatus_t status;
    status = hipsparseXcsric02_zeroPivot(fs->handle, fs->ic0Info_M, &numerical_zero);
    PetscAssert(HIPSPARSE_STATUS_ZERO_PIVOT != status, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Numerical zero pivot detected in csric02: A(%d,%d) is zero", numerical_zero, numerical_zero);
  }

  PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, fs->spsvBuffer_L));

  /* Note that hipsparse reports this error if we use double and HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    ** On entry to hipsparseSpSV_analysis(): conjugate transpose (opA) is not supported for matA data type, current -> CUDA_R_64F
  */
  PetscCallHIPSPARSE(hipsparseSpSV_analysis(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, fs->spsvBuffer_Lt));

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
    PetscInt  i;
    PetscBool flg, missing;

    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJHIPSPARSE, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_GPU, "Expected MATSEQAIJHIPSPARSE, but input is %s", ((PetscObject)A)->type_name);
    PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
    PetscCall(MatMissingDiagonal(A, &missing, &i));
    PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, i);
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
  const int *Ai, *Aj;

  m  = fact->rmap->n;
  nz = aij->nz;

  PetscCallHIP(hipMalloc((void **)&fs->csrRowPtr, sizeof(int) * (m + 1)));
  PetscCallHIP(hipMalloc((void **)&fs->csrColIdx, sizeof(int) * nz));
  PetscCallHIP(hipMalloc((void **)&fs->csrVal, sizeof(PetscScalar) * nz));
  PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &Ai, &Aj)); /* Do not use compressed Ai */
  PetscCallHIP(hipMemcpyAsync(fs->csrRowPtr, Ai, sizeof(int) * (m + 1), hipMemcpyDeviceToDevice, PetscDefaultHipStream));
  PetscCallHIP(hipMemcpyAsync(fs->csrColIdx, Aj, sizeof(int) * nz, hipMemcpyDeviceToDevice, PetscDefaultHipStream));

  /* ====================================================================== */
  /* Create mat descriptors for M, L                                        */
  /* ====================================================================== */
  hipsparseFillMode_t fillMode;
  hipsparseDiagType_t diagType;

  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&fs->matDescr_M));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(fs->matDescr_M, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(fs->matDescr_M, HIPSPARSE_MATRIX_TYPE_GENERAL));

  /* https://docs.amd.com/bundle/hipSPARSE-Documentation---hipSPARSE-documentation/page/usermanual.html/#hipsparse_8h_1a79e036b6c0680cb37e2aa53d3542a054
    hipsparseDiagType_t: This type indicates if the matrix diagonal entries are unity. The diagonal elements are always
    assumed to be present, but if HIPSPARSE_DIAG_TYPE_UNIT is passed to an API routine, then the routine assumes that
    all diagonal entries are unity and will not read or modify those entries. Note that in this case the routine
    assumes the diagonal entries are equal to one, regardless of what those entries are actually set to in memory.
  */
  fillMode = HIPSPARSE_FILL_MODE_LOWER;
  diagType = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  PetscCallHIPSPARSE(hipsparseCreateCsr(&fs->spMatDescr_L, m, m, nz, fs->csrRowPtr, fs->csrColIdx, fs->csrVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
  PetscCallHIPSPARSE(hipsparseSpMatSetAttribute(fs->spMatDescr_L, HIPSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

  /* ========================================================================= */
  /* Query buffer sizes for csric0, SpSV of L and Lt, and allocate buffers     */
  /* ========================================================================= */
  PetscCallHIPSPARSE(hipsparseCreateCsric02Info(&fs->ic0Info_M));
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02_bufferSize(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ic0Info_M, &fs->factBufferSize_M));

  PetscCallHIP(hipMalloc((void **)&fs->X, sizeof(PetscScalar) * m));
  PetscCallHIP(hipMalloc((void **)&fs->Y, sizeof(PetscScalar) * m));

  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_X, m, fs->X, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseCreateDnVec(&fs->dnVecDescr_Y, m, fs->Y, hipsparse_scalartype));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_L));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_L, &fs->spsvBufferSize_L));

  PetscCallHIPSPARSE(hipsparseSpSV_createDescr(&fs->spsvDescr_Lt));
  PetscCallHIPSPARSE(hipsparseSpSV_bufferSize(fs->handle, HIPSPARSE_OPERATION_TRANSPOSE, &PETSC_HIPSPARSE_ONE, fs->spMatDescr_L, fs->dnVecDescr_X, fs->dnVecDescr_Y, hipsparse_scalartype, HIPSPARSE_SPSV_ALG_DEFAULT, fs->spsvDescr_Lt, &fs->spsvBufferSize_Lt));

  /* To save device memory, we make the factorization buffer share with one of the solver buffer.
     See also comments in `MatILUFactorSymbolic_SeqAIJHIPSPARSE_ILU0()`.
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
  int structural_zero;

  fs->policy_M = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  if (m) PetscCallHIPSPARSE(hipsparseXcsric02_analysis(fs->handle, m, nz, fs->matDescr_M, fs->csrVal, fs->csrRowPtr, fs->csrColIdx, fs->ic0Info_M, fs->policy_M, fs->factBuffer_M));
  if (PetscDefined(USE_DEBUG)) {
    hipsparseStatus_t status;
    /* Function hipsparseXcsric02_zeroPivot() is a blocking call. It calls hipDeviceSynchronize() to make sure all previous kernels are done. */
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
#endif

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
  PetscBool row_identity = PETSC_FALSE, col_identity = PETSC_FALSE;
  if (hipsparseTriFactors->factorizeOnDevice) {
    PetscCall(ISIdentity(isrow, &row_identity));
    PetscCall(ISIdentity(iscol, &col_identity));
  }
  if (!info->levels && row_identity && col_identity) PetscCall(MatILUFactorSymbolic_SeqAIJHIPSPARSE_ILU0(B, A, isrow, iscol, info));
  else
#endif
  {
    PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
    PetscCall(MatILUFactorSymbolic_SeqAIJ(B, A, isrow, iscol, info));
    B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  }
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

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
  PetscBool perm_identity = PETSC_FALSE;
  if (hipsparseTriFactors->factorizeOnDevice) PetscCall(ISIdentity(perm, &perm_identity));
  if (!info->levels && perm_identity) PetscCall(MatICCFactorSymbolic_SeqAIJHIPSPARSE_ICC0(B, A, perm, info));
  else
#endif
  {
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

PetscErrorCode MatFactorGetSolverType_seqaij_hipsparse(Mat A, MatSolverType *type)
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

.seealso: [](chapter_matrices), `Mat`, `MATSEQAIJHIPSPARSE`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCreateSeqAIJHIPSPARSE()`, `MATAIJHIPSPARSE`, `MatCreateAIJHIPSPARSE()`, `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat A, MatFactorType ftype, Mat *B)
{
  PetscInt  n = A->rmap->n;
  PetscBool factOnDevice, factOnHost;
  char     *prefix;
  char      factPlace[32] = "device"; /* the default */

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  (*B)->factortype = ftype;
  PetscCall(MatSetType(*B, MATSEQAIJHIPSPARSE));

  prefix = (*B)->factorprefix ? (*B)->factorprefix : ((PetscObject)A)->prefix;
  PetscOptionsBegin(PetscObjectComm((PetscObject)(*B)), prefix, "MatGetFactor", "Mat");
  PetscCall(PetscOptionsString("-mat_factor_bind_factorization", "Do matrix factorization on host or device when possible", "MatGetFactor", NULL, factPlace, sizeof(factPlace), NULL));
  PetscOptionsEnd();
  PetscCall(PetscStrcasecmp("device", factPlace, &factOnDevice));
  PetscCall(PetscStrcasecmp("host", factPlace, &factOnHost));
  PetscCheck(factOnDevice || factOnHost, PetscObjectComm((PetscObject)(*B)), PETSC_ERR_ARG_OUTOFRANGE, "Wrong option %s to -mat_factor_bind_factorization <string>. Only host and device are allowed", factPlace);
  ((Mat_SeqAIJHIPSPARSETriFactors *)(*B)->spptr)->factorizeOnDevice = factOnDevice;

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
  PetscCall(PetscObjectComposeFunction((PetscObject)(*B), "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_hipsparse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat A)
{
  Mat_SeqAIJ          *a    = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
  Mat_SeqAIJHIPSPARSETriFactors *fs = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
#endif

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    PetscCall(PetscLogEventBegin(MAT_HIPSPARSECopyFromGPU, A, 0, 0, 0));
    if (A->factortype == MAT_FACTOR_NONE) {
      CsrMatrix *matrix = (CsrMatrix *)cusp->mat->mat;
      PetscCallHIP(hipMemcpy(a->a, matrix->values->data().get(), a->nz * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    }
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
    else if (fs->csrVal) {
      /* We have a factorized matrix on device and are able to copy it to host */
      PetscCallHIP(hipMemcpy(a->a, fs->csrVal, a->nz * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    }
#endif
    else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for copying this type of factorized matrix from device to host");
    PetscCall(PetscLogGpuToCpu(a->nz * sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(MAT_HIPSPARSECopyFromGPU, A, 0, 0, 0));
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  *array = ((Mat_SeqAIJ *)A->data)->a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJHIPSPARSE(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSECopyFromGPU(A));
  *array = ((Mat_SeqAIJ *)A->data)->a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJHIPSPARSE(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = ((Mat_SeqAIJ *)A->data)->a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJHIPSPARSE(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
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

  if (i) {
#if !defined(PETSC_USE_64BIT_INDICES)
    *i = matrix->row_offsets->data().get();
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSparse does not supported 64-bit indices");
#endif
  }
  if (j) {
#if !defined(PETSC_USE_64BIT_INDICES)
    *j = matrix->column_indices->data().get();
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSparse does not supported 64-bit indices");
#endif
  }
  if (a) *a = matrix->values->data().get();
  if (mtype) *mtype = PETSC_MEMTYPE_HIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct       = hipsparsestruct->mat;
  Mat_SeqAIJ                    *a               = (Mat_SeqAIJ *)A->data;
  PetscBool                      both            = PETSC_TRUE;
  PetscInt                       m               = A->rmap->n, *ii, *ridx, tmp;

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
      PetscCall(PetscLogCpuToGpu((a->nz) * sizeof(PetscScalar)));
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

        /* create hipsparse matrix */
        hipsparsestruct->nrows = m;
        matstruct              = new Mat_SeqAIJHIPSPARSEMultStruct;
        PetscCallHIPSPARSE(hipsparseCreateMatDescr(&matstruct->descr));
        PetscCallHIPSPARSE(hipsparseSetMatIndexBase(matstruct->descr, HIPSPARSE_INDEX_BASE_ZERO));
        PetscCallHIPSPARSE(hipsparseSetMatType(matstruct->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));

        PetscCallHIP(hipMalloc((void **)&(matstruct->alpha_one), sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&(matstruct->beta_zero), sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&(matstruct->beta_one), sizeof(PetscScalar)));
        PetscCallHIP(hipMemcpy(matstruct->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(matstruct->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(matstruct->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIPSPARSE(hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE));

        /* Build a hybrid/ellpack matrix if this option is chosen for the storage */
        if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
          /* set the matrix */
          CsrMatrix *mat      = new CsrMatrix;
          mat->num_rows       = m;
          mat->num_cols       = A->cmap->n;
          mat->num_entries    = nnz;
          mat->row_offsets    = new THRUSTINTARRAY32(m + 1);
          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->values         = new THRUSTARRAY(nnz);
          mat->row_offsets->assign(ii, ii + m + 1);
          mat->column_indices->assign(a->j, a->j + nnz);
          if (a->a) mat->values->assign(a->a, a->a + nnz);

          /* assign the pointer */
          matstruct->mat = mat;
          if (mat->num_rows) { /* hipsparse errors on empty matrices! */
            PetscCallHIPSPARSE(hipsparseCreateCsr(&matstruct->matDescr, mat->num_rows, mat->num_cols, mat->num_entries, mat->row_offsets->data().get(), mat->column_indices->data().get(), mat->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, /* row offset, col idx types due to THRUSTINTARRAY32 */
                                                  HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
          }
        } else if (hipsparsestruct->format == MAT_HIPSPARSE_ELL || hipsparsestruct->format == MAT_HIPSPARSE_HYB) {
          CsrMatrix *mat      = new CsrMatrix;
          mat->num_rows       = m;
          mat->num_cols       = A->cmap->n;
          mat->num_entries    = nnz;
          mat->row_offsets    = new THRUSTINTARRAY32(m + 1);
          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->values         = new THRUSTARRAY(nnz);
          mat->row_offsets->assign(ii, ii + m + 1);
          mat->column_indices->assign(a->j, a->j + nnz);
          if (a->a) mat->values->assign(a->a, a->a + nnz);

          hipsparseHybMat_t hybMat;
          PetscCallHIPSPARSE(hipsparseCreateHybMat(&hybMat));
          hipsparseHybPartition_t partition = hipsparsestruct->format == MAT_HIPSPARSE_ELL ? HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
          PetscCallHIPSPARSE(hipsparse_csr2hyb(hipsparsestruct->handle, mat->num_rows, mat->num_cols, matstruct->descr, mat->values->data().get(), mat->row_offsets->data().get(), mat->column_indices->data().get(), hybMat, 0, partition));
          /* assign the pointer */
          matstruct->mat = hybMat;

          if (mat) {
            if (mat->values) delete (THRUSTARRAY *)mat->values;
            if (mat->column_indices) delete (THRUSTINTARRAY32 *)mat->column_indices;
            if (mat->row_offsets) delete (THRUSTINTARRAY32 *)mat->row_offsets;
            delete (CsrMatrix *)mat;
          }
        }

        /* assign the compressed row indices */
        if (a->compressedrow.use) {
          hipsparsestruct->workVector = new THRUSTARRAY(m);
          matstruct->cprowIndices     = new THRUSTINTARRAY(m);
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

struct MatMatHipsparse {
  PetscBool             cisdense;
  PetscScalar          *Bt;
  Mat                   X;
  PetscBool             reusesym; /* Hipsparse does not have split symbolic and numeric phases for sparse matmat operations */
  PetscLogDouble        flops;
  CsrMatrix            *Bcsr;
  hipsparseSpMatDescr_t matSpBDescr;
  PetscBool             initialized; /* C = alpha op(A) op(B) + beta C */
  hipsparseDnMatDescr_t matBDescr;
  hipsparseDnMatDescr_t matCDescr;
  PetscInt              Blda, Clda; /* Record leading dimensions of B and C here to detect changes*/
#if PETSC_PKG_HIP_VERSION_GE(5, 1, 0)
  void *dBuffer4, *dBuffer5;
#endif
  size_t                 mmBufferSize;
  void                  *mmBuffer, *mmBuffer2; /* SpGEMM WorkEstimation buffer */
  hipsparseSpGEMMDescr_t spgemmDesc;
};

static PetscErrorCode MatDestroy_MatMatHipsparse(void *data)
{
  MatMatHipsparse *mmdata = (MatMatHipsparse *)data;

  PetscFunctionBegin;
  PetscCallHIP(hipFree(mmdata->Bt));
  delete mmdata->Bcsr;
  if (mmdata->matSpBDescr) PetscCallHIPSPARSE(hipsparseDestroySpMat(mmdata->matSpBDescr));
  if (mmdata->matBDescr) PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matBDescr));
  if (mmdata->matCDescr) PetscCallHIPSPARSE(hipsparseDestroyDnMat(mmdata->matCDescr));
  if (mmdata->spgemmDesc) PetscCallHIPSPARSE(hipsparseSpGEMM_destroyDescr(mmdata->spgemmDesc));
#if PETSC_PKG_HIP_VERSION_GE(5, 1, 0)
  if (mmdata->dBuffer4) PetscCallHIP(hipFree(mmdata->dBuffer4));
  if (mmdata->dBuffer5) PetscCallHIP(hipFree(mmdata->dBuffer5));
#endif
  if (mmdata->mmBuffer) PetscCallHIP(hipFree(mmdata->mmBuffer));
  if (mmdata->mmBuffer2) PetscCallHIP(hipFree(mmdata->mmBuffer2));
  PetscCall(MatDestroy(&mmdata->X));
  PetscCall(PetscFree(data));
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
  MatMatHipsparse               *mmdata;
  Mat_SeqAIJHIPSPARSEMultStruct *mat;
  CsrMatrix                     *csrmat;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data empty");
  mmdata = (MatMatHipsparse *)product->data;
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
  if (!biship) { PetscCall(MatConvert(B, MATSEQDENSEHIP, MAT_INPLACE_MATRIX, &B)); }
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
      PetscCallHIPSPARSE(hipsparseCreateCsr(&mat->matDescr, csrmat->num_rows, csrmat->num_cols, csrmat->num_entries, csrmat->row_offsets->data().get(), csrmat->column_indices->data().get(), csrmat->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, /* row offset, col idx types due to THRUSTINTARRAY32 */
                                            HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
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
  Mat_Product         *product = C->product;
  Mat                  A, B;
  PetscInt             m, n;
  PetscBool            cisdense, flg;
  MatMatHipsparse     *mmdata;
  Mat_SeqAIJHIPSPARSE *cusp;

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
  C->product->destroy    = MatDestroy_MatMatHipsparse;
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
  MatMatHipsparse               *mmdata;
  hipsparseSpMatDescr_t          BmatSpDescr;
  hipsparseOperation_t           opA = HIPSPARSE_OPERATION_NON_TRANSPOSE, opB = HIPSPARSE_OPERATION_NON_TRANSPOSE; /* hipSPARSE spgemm doesn't support transpose yet */

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Product data empty");
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATSEQAIJHIPSPARSE, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)C), PETSC_ERR_GPU, "Not for C of type %s", ((PetscObject)C)->type_name);
  mmdata = (MatMatHipsparse *)C->product->data;
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
    Amat = Acusp->matTranspose;
    Bmat = Bcusp->mat;
    break;
  case MATPRODUCT_ABt:
    Amat = Acusp->mat;
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
#if PETSC_PKG_HIP_VERSION_GE(5, 0, 0)
  BmatSpDescr = mmdata->Bcsr ? mmdata->matSpBDescr : Bmat->matDescr; /* B may be in compressed row storage */
  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  #if PETSC_PKG_HIP_VERSION_GE(5, 1, 0)
  PetscCallHIPSPARSE(hipsparseSpGEMMreuse_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
  #else
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer));
  PetscCallHIPSPARSE(hipsparseSpGEMM_copy(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
  #endif
#else
  PetscCallHIPSPARSE(hipsparse_csr_spgemm(Ccusp->handle, opA, opB, Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols, Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(), Bmat->descr,
                                          Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(), Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(),
                                          Ccsr->column_indices->data().get()));
#endif
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogGpuTimeEnd());
  C->offloadmask = PETSC_OFFLOAD_GPU;
finalize:
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  PetscCall(PetscInfo(C, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n", C->rmap->n, C->cmap->n, c->nz));
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
  MatMatHipsparse               *mmdata;
  PetscLogDouble                 flops;
  PetscBool                      biscompressed, ciscompressed;
#if PETSC_PKG_HIP_VERSION_GE(5, 0, 0)
  int64_t               C_num_rows1, C_num_cols1, C_nnz1;
  hipsparseSpMatDescr_t BmatSpDescr;
#else
  int cnz;
#endif
  hipsparseOperation_t opA = HIPSPARSE_OPERATION_NON_TRANSPOSE, opB = HIPSPARSE_OPERATION_NON_TRANSPOSE; /* hipSPARSE spgemm doesn't support transpose yet */

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
  C->product->destroy = MatDestroy_MatMatHipsparse;

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
  Ccsr->row_offsets = new THRUSTINTARRAY32(Ccusp->nrows + 1);
  PetscCallHIPSPARSE(hipsparseCreateMatDescr(&Cmat->descr));
  PetscCallHIPSPARSE(hipsparseSetMatIndexBase(Cmat->descr, HIPSPARSE_INDEX_BASE_ZERO));
  PetscCallHIPSPARSE(hipsparseSetMatType(Cmat->descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  PetscCallHIP(hipMalloc((void **)&(Cmat->alpha_one), sizeof(PetscScalar)));
  PetscCallHIP(hipMalloc((void **)&(Cmat->beta_zero), sizeof(PetscScalar)));
  PetscCallHIP(hipMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar)));
  PetscCallHIP(hipMemcpy(Cmat->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
  PetscCallHIP(hipMemcpy(Cmat->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
  PetscCallHIP(hipMemcpy(Cmat->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
  if (!Ccsr->num_rows || !Ccsr->num_cols || !a->nz || !b->nz) { /* hipsparse raise errors in different calls when matrices have zero rows/columns! */
    thrust::fill(thrust::device, Ccsr->row_offsets->begin(), Ccsr->row_offsets->end(), 0);
    c->nz                = 0;
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
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
      Bcusp->rowoffsets_gpu = new THRUSTINTARRAY32(B->rmap->n + 1);
      Bcusp->rowoffsets_gpu->assign(b->i, b->i + B->rmap->n + 1);
      PetscCall(PetscLogCpuToGpu((B->rmap->n + 1) * sizeof(PetscInt)));
    }
    Bcsr->row_offsets = Bcusp->rowoffsets_gpu;
    mmdata->Bcsr      = Bcsr;
    if (Bcsr->num_rows && Bcsr->num_cols) {
      PetscCallHIPSPARSE(hipsparseCreateCsr(&mmdata->matSpBDescr, Bcsr->num_rows, Bcsr->num_cols, Bcsr->num_entries, Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(), Bcsr->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
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
#if PETSC_PKG_HIP_VERSION_GE(5, 0, 0)
  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  PetscCallHIPSPARSE(hipsparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, 0, Ccsr->row_offsets->data().get(), NULL, NULL, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
  PetscCallHIPSPARSE(hipsparseSpGEMM_createDescr(&mmdata->spgemmDesc));
  #if PETSC_PKG_HIP_VERSION_GE(5, 1, 0)
  {
    /* hipsparseSpGEMMreuse has more reasonable APIs than hipsparseSpGEMM, so we prefer to use it.
     We follow the sample code at https://github.com/ROCmSoftwarePlatform/hipSPARSE/blob/develop/clients/include/testing_spgemmreuse_csr.hpp
  */
    void *dBuffer1 = NULL;
    void *dBuffer2 = NULL;
    void *dBuffer3 = NULL;
    /* dBuffer4, dBuffer5 are needed by hipsparseSpGEMMreuse_compute, and therefore are stored in mmdata */
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;

    /* ask bufferSize1 bytes for external memory */
    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_workEstimation(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize1, NULL));
    PetscCallHIP(hipMalloc((void **)&dBuffer1, bufferSize1));
    /* inspect the matrices A and B to understand the memory requirement for the next step */
    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_workEstimation(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize1, dBuffer1));

    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize2, NULL, &bufferSize3, NULL, &bufferSize4, NULL));
    PetscCallHIP(hipMalloc((void **)&dBuffer2, bufferSize2));
    PetscCallHIP(hipMalloc((void **)&dBuffer3, bufferSize3));
    PetscCallHIP(hipMalloc((void **)&mmdata->dBuffer4, bufferSize4));
    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize2, dBuffer2, &bufferSize3, dBuffer3, &bufferSize4, mmdata->dBuffer4));
    PetscCallHIP(hipFree(dBuffer1));
    PetscCallHIP(hipFree(dBuffer2));

    /* get matrix C non-zero entries C_nnz1 */
    PetscCallHIPSPARSE(hipsparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1));
    c->nz = (PetscInt)C_nnz1;
    /* allocate matrix C */
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
    Ccsr->values = new THRUSTARRAY(c->nz);
    PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
    /* update matC with the new pointers */
    PetscCallHIPSPARSE(hipsparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get()));

    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize5, NULL));
    PetscCallHIP(hipMalloc((void **)&mmdata->dBuffer5, bufferSize5));
    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufferSize5, mmdata->dBuffer5));
    PetscCallHIP(hipFree(dBuffer3));
    PetscCallHIPSPARSE(hipsparseSpGEMMreuse_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
    PetscCall(PetscInfo(C, "Buffer sizes for type %s, result %" PetscInt_FMT " x %" PetscInt_FMT " (k %" PetscInt_FMT ", nzA %" PetscInt_FMT ", nzB %" PetscInt_FMT ", nzC %" PetscInt_FMT ") are: %ldKB %ldKB\n", MatProductTypes[ptype], m, n, k, a->nz, b->nz, c->nz, bufferSize4 / 1024, bufferSize5 / 1024));
  }
  #else
  size_t bufSize2;
  /* ask bufferSize bytes for external memory */
  PetscCallHIPSPARSE(hipsparseSpGEMM_workEstimation(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufSize2, NULL));
  PetscCallHIP(hipMalloc((void **)&mmdata->mmBuffer2, bufSize2));
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  PetscCallHIPSPARSE(hipsparseSpGEMM_workEstimation(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &bufSize2, mmdata->mmBuffer2));
  /* ask bufferSize again bytes for external memory */
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, NULL));
  /* Similar to CUSPARSE, we need both buffers to perform the operations properly!
     mmdata->mmBuffer2 does not appear anywhere in the compute/copy API
     it only appears for the workEstimation stuff, but it seems it is needed in compute, so probably the address
     is stored in the descriptor! What a messy API... */
  PetscCallHIP(hipMalloc((void **)&mmdata->mmBuffer, mmdata->mmBufferSize));
  /* compute the intermediate product of A * B */
  PetscCallHIPSPARSE(hipsparseSpGEMM_compute(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer));
  /* get matrix C non-zero entries C_nnz1 */
  PetscCallHIPSPARSE(hipsparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1));
  c->nz = (PetscInt)C_nnz1;
  PetscCall(PetscInfo(C, "Buffer sizes for type %s, result %" PetscInt_FMT " x %" PetscInt_FMT " (k %" PetscInt_FMT ", nzA %" PetscInt_FMT ", nzB %" PetscInt_FMT ", nzC %" PetscInt_FMT ") are: %ldKB %ldKB\n", MatProductTypes[ptype], m, n, k, a->nz, b->nz, c->nz, bufSize2 / 1024,
                      mmdata->mmBufferSize / 1024));
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
  PetscCallHIPSPARSE(hipsparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get()));
  PetscCallHIPSPARSE(hipsparseSpGEMM_copy(Ccusp->handle, opA, opB, Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr, hipsparse_scalartype, HIPSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc));
  #endif
#else
  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_HOST));
  PetscCallHIPSPARSE(hipsparseXcsrgemmNnz(Ccusp->handle, opA, opB, Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols, Amat->descr, Acsr->num_entries, Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(), Bmat->descr, Bcsr->num_entries,
                                          Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(), Cmat->descr, Ccsr->row_offsets->data().get(), &cnz));
  c->nz                = cnz;
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  PetscCallHIP(hipPeekAtLastError()); /* catch out of memory errors */

  PetscCallHIPSPARSE(hipsparseSetPointerMode(Ccusp->handle, HIPSPARSE_POINTER_MODE_DEVICE));
  /* with the old gemm interface (removed from 11.0 on) we cannot compute the symbolic factorization only.
      I have tried using the gemm2 interface (alpha * A * B + beta * D), which allows to do symbolic by passing NULL for values, but it seems quite buggy when
      D is NULL, despite the fact that CUSPARSE documentation claims it is supported! */
  PetscCallHIPSPARSE(hipsparse_csr_spgemm(Ccusp->handle, opA, opB, Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols, Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(), Bmat->descr,
                                          Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(), Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(),
                                          Ccsr->column_indices->data().get()));
#endif
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCall(PetscLogGpuTimeEnd());
finalizesym:
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  PetscCall(PetscMalloc1(m + 1, &c->i));
  PetscCall(PetscMalloc1(c->nz, &c->j));
  if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
    PetscInt      *d_i = c->i;
    THRUSTINTARRAY ii(Ccsr->row_offsets->size());
    THRUSTINTARRAY jj(Ccsr->column_indices->size());
    ii = *Ccsr->row_offsets;
    jj = *Ccsr->column_indices;
    if (ciscompressed) d_i = c->compressedrow.i;
    PetscCallHIP(hipMemcpy(d_i, ii.data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    PetscCallHIP(hipMemcpy(c->j, jj.data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
  } else {
    PetscInt *d_i = c->i;
    if (ciscompressed) d_i = c->compressedrow.i;
    PetscCallHIP(hipMemcpy(d_i, Ccsr->row_offsets->data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    PetscCallHIP(hipMemcpy(c->j, Ccsr->column_indices->data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
  }
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
    c->nonzerorowcnt += (PetscInt) !!nn;
    c->rmax = PetscMax(c->rmax, nn);
  }
  PetscCall(MatMarkDiagonal_SeqAIJ(C));
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
#if PETSC_PKG_HIP_VERSION_GE(5, 1, 0)
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
      PetscCallHIPSPARSE(hipsparse_csr_spmv(hipsparsestruct->handle, opA, mat->num_rows, mat->num_cols, mat->num_entries, matstruct->alpha_one, matstruct->descr, mat->values->data().get(), mat->row_offsets->data().get(), mat->column_indices->data().get(), xptr, beta, dptr));
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
  PetscObjectState     onnz = A->nonzerostate;
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqAIJ(A, mode));
  if (onnz != A->nonzerostate && cusp->deviceMat) {
    PetscCall(PetscInfo(A, "Destroy device mat since nonzerostate changed\n"));
    PetscCallHIP(hipFree(cusp->deviceMat));
    cusp->deviceMat = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCreateSeqAIJHIPSPARSE - Creates a sparse matrix in `MATAIJHIPSPARSE` (compressed row) format.
   This matrix will ultimately pushed down to AMD GPUs and use the HIPSPARSE library for calculations.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to `PETSC_COMM_SELF`
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows), ignored if `nnz` is set
-  nnz - array containing the number of nonzeros in the various rows (possibly different for each row) or `NULL`

   Output Parameter:
.  A - the matrix

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

.seealso: [](chapter_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`, `MATSEQAIJHIPSPARSE`, `MATAIJHIPSPARSE`
@*/
PetscErrorCode MatCreateSeqAIJHIPSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQAIJHIPSPARSE));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A, nz, (PetscInt *)nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqAIJHIPSPARSE(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) PetscCall(MatSeqAIJHIPSPARSE_Destroy((Mat_SeqAIJHIPSPARSE **)&A->spptr));
  else PetscCall(MatSeqAIJHIPSPARSETriFactors_Destroy((Mat_SeqAIJHIPSPARSETriFactors **)&A->spptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHIPSPARSESetFormat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHIPSPARSESetUseCPUSolve_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaijhipsparse_hypre_C", NULL));
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_SeqAIJHIPSPARSE(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqAIJ(A, cpvalues, B));
  PetscCall(MatConvert_SeqAIJ_SeqAIJHIPSPARSE(*B, MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_SeqAIJ          *x = (Mat_SeqAIJ *)X->data, *y = (Mat_SeqAIJ *)Y->data;
  Mat_SeqAIJHIPSPARSE *cy;
  Mat_SeqAIJHIPSPARSE *cx;
  PetscScalar         *ay;
  const PetscScalar   *ax;
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
  /* see if we can turn this into a hipblas axpy */
  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz && !x->compressedrow.use && !y->compressedrow.use) {
    bool eq = thrust::equal(thrust::device, csry->row_offsets->begin(), csry->row_offsets->end(), csrx->row_offsets->begin());
    if (eq) eq = thrust::equal(thrust::device, csry->column_indices->begin(), csry->column_indices->end(), csrx->column_indices->begin());
    if (eq) str = SAME_NONZERO_PATTERN;
  }
  /* spgeam is buggy with one column */
  if (Y->cmap->n == 1 && str != SAME_NONZERO_PATTERN) str = DIFFERENT_NONZERO_PATTERN;
  if (str == SUBSET_NONZERO_PATTERN) {
    PetscScalar b = 1.0;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
    size_t bufferSize;
    void  *buffer;
#endif

    PetscCall(MatSeqAIJHIPSPARSEGetArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSEGetArray(Y, &ay));
    PetscCallHIPSPARSE(hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_HOST));
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
    PetscCallHIPSPARSE(hipsparse_csr_spgeam_bufferSize(cy->handle, Y->rmap->n, Y->cmap->n, &a, cx->mat->descr, x->nz, ax, csrx->row_offsets->data().get(), csrx->column_indices->data().get(), &b, cy->mat->descr, y->nz, ay, csry->row_offsets->data().get(),
                                                       csry->column_indices->data().get(), cy->mat->descr, ay, csry->row_offsets->data().get(), csry->column_indices->data().get(), &bufferSize));
    PetscCallHIP(hipMalloc(&buffer, bufferSize));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallHIPSPARSE(hipsparse_csr_spgeam(cy->handle, Y->rmap->n, Y->cmap->n, &a, cx->mat->descr, x->nz, ax, csrx->row_offsets->data().get(), csrx->column_indices->data().get(), &b, cy->mat->descr, y->nz, ay, csry->row_offsets->data().get(),
                                            csry->column_indices->data().get(), cy->mat->descr, ay, csry->row_offsets->data().get(), csry->column_indices->data().get(), buffer));
    PetscCall(PetscLogGpuFlops(x->nz + y->nz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCallHIP(hipFree(buffer));
#else
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallHIPSPARSE(hipsparse_csr_spgeam(cy->handle, Y->rmap->n, Y->cmap->n, &a, cx->mat->descr, x->nz, ax, csrx->row_offsets->data().get(), csrx->column_indices->data().get(), &b, cy->mat->descr, y->nz, ay, csry->row_offsets->data().get(),
                                            csry->column_indices->data().get(), cy->mat->descr, ay, csry->row_offsets->data().get(), csry->column_indices->data().get()));
    PetscCall(PetscLogGpuFlops(x->nz + y->nz));
    PetscCall(PetscLogGpuTimeEnd());
#endif
    PetscCallHIPSPARSE(hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_DEVICE));
    PetscCall(MatSeqAIJHIPSPARSERestoreArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(Y, &ay));
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  } else if (str == SAME_NONZERO_PATTERN) {
    hipblasHandle_t hipblasv2handle;
    PetscBLASInt    one = 1, bnz = 1;

    PetscCall(MatSeqAIJHIPSPARSEGetArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSEGetArray(Y, &ay));
    PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
    PetscCall(PetscBLASIntCast(x->nz, &bnz));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallHIPBLAS(hipblasXaxpy(hipblasv2handle, bnz, &a, ax, one, ay, one));
    PetscCall(PetscLogGpuFlops(2.0 * bnz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(MatSeqAIJHIPSPARSERestoreArrayRead(X, &ax));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(Y, &ay));
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  } else {
    PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(Y, PETSC_FALSE));
    PetscCall(MatAXPY_SeqAIJ(Y, a, X, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat Y, PetscScalar a)
{
  Mat_SeqAIJ     *y = (Mat_SeqAIJ *)Y->data;
  PetscScalar    *ay;
  hipblasHandle_t hipblasv2handle;
  PetscBLASInt    one = 1, bnz = 1;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJHIPSPARSEGetArray(Y, &ay));
  PetscCall(PetscHIPBLASGetHandle(&hipblasv2handle));
  PetscCall(PetscBLASIntCast(y->nz, &bnz));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallHIPBLAS(hipblasXscal(hipblasv2handle, bnz, &a, ay, one));
  PetscCall(PetscLogGpuFlops(bnz));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatSeqAIJHIPSPARSERestoreArray(Y, &ay));
  PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqAIJHIPSPARSE(Mat A)
{
  PetscBool   both = PETSC_FALSE;
  Mat_SeqAIJ *a    = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJHIPSPARSE *spptr = (Mat_SeqAIJHIPSPARSE *)A->spptr;
    if (spptr->mat) {
      CsrMatrix *matrix = (CsrMatrix *)spptr->mat->mat;
      if (matrix->values) {
        both = PETSC_TRUE;
        thrust::fill(thrust::device, matrix->values->begin(), matrix->values->end(), 0.);
      }
    }
    if (spptr->matTranspose) {
      CsrMatrix *matrix = (CsrMatrix *)spptr->matTranspose->mat;
      if (matrix->values) { thrust::fill(thrust::device, matrix->values->begin(), matrix->values->end(), 0.); }
    }
  }
  //PetscCall(MatZeroEntries_SeqAIJ(A));
  PetscCall(PetscArrayzero(a->a, a->i[A->rmap->n]));
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
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
    A->ops->axpy                      = MatAXPY_SeqAIJ;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJ;
    A->ops->mult                      = MatMult_SeqAIJ;
    A->ops->multadd                   = MatMultAdd_SeqAIJ;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJ;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJ;
    A->ops->multhermitiantranspose    = NULL;
    A->ops->multhermitiantransposeadd = NULL;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJ;
    PetscCall(PetscMemzero(a->ops, sizeof(Mat_SeqAIJOps)));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqdense_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C", NULL));
  } else {
    A->ops->scale                     = MatScale_SeqAIJHIPSPARSE;
    A->ops->axpy                      = MatAXPY_SeqAIJHIPSPARSE;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJHIPSPARSE;
    A->ops->mult                      = MatMult_SeqAIJHIPSPARSE;
    A->ops->multadd                   = MatMultAdd_SeqAIJHIPSPARSE;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJHIPSPARSE;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJHIPSPARSE;
    A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJHIPSPARSE;
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
  if (flg && a->inode.size) a->inode.use = PETSC_TRUE;
  else a->inode.use = PETSC_FALSE;

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
      spptr->format = MAT_HIPSPARSE_CSR;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
      spptr->spmvAlg = HIPSPARSE_SPMV_CSR_ALG1;
#else
      spptr->spmvAlg = HIPSPARSE_CSRMV_ALG1; /* default, since we only support csr */
#endif
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
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJHIPSPARSE;
  B->ops->destroy        = MatDestroy_SeqAIJHIPSPARSE;
  B->ops->setoption      = MatSetOption_SeqAIJHIPSPARSE;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJHIPSPARSE;
  B->ops->bindtocpu      = MatBindToCPU_SeqAIJHIPSPARSE;
  B->ops->duplicate      = MatDuplicate_SeqAIJHIPSPARSE;

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

   A matrix type type whose data resides on AMD GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format.
   All matrix calculations are performed on AMD/NVIDIA GPUs using the HIPSPARSE library.

   Options Database Keys:
+  -mat_type aijhipsparse - sets the matrix type to `MATSEQAIJHIPSPARSE`
.  -mat_hipsparse_storage_format csr - sets the storage format of matrices (for `MatMult()` and factors in `MatSolve()`).
                                       Other options include ell (ellpack) or hyb (hybrid).
. -mat_hipsparse_mult_storage_format csr - sets the storage format of matrices (for `MatMult()`). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_use_cpu_solve - Do `MatSolve()` on the CPU

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MatCreateSeqAIJHIPSPARSE()`, `MATAIJHIPSPARSE`, `MatCreateAIJHIPSPARSE()`, `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
M*/

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_HIPSPARSE(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSEBAND, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaijhipsparse_hipsparse_band));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_LU, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_ILU, MatGetFactor_seqaijhipsparse_hipsparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIPSPARSE, MATSEQAIJHIPSPARSE, MAT_FACTOR_ICC, MatGetFactor_seqaijhipsparse_hipsparse));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatResetPreallocationCOO_SeqAIJHIPSPARSE(Mat mat)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)mat->spptr;

  PetscFunctionBegin;
  if (!cusp) PetscFunctionReturn(PETSC_SUCCESS);
  delete cusp->cooPerm;
  delete cusp->cooPerm_a;
  cusp->cooPerm   = NULL;
  cusp->cooPerm_a = NULL;
  if (cusp->use_extended_coo) {
    PetscCallHIP(hipFree(cusp->jmap_d));
    PetscCallHIP(hipFree(cusp->perm_d));
  }
  cusp->use_extended_coo = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE **hipsparsestruct)
{
  PetscFunctionBegin;
  if (*hipsparsestruct) {
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->mat, (*hipsparsestruct)->format));
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->matTranspose, (*hipsparsestruct)->format));
    delete (*hipsparsestruct)->workVector;
    delete (*hipsparsestruct)->rowoffsets_gpu;
    delete (*hipsparsestruct)->cooPerm;
    delete (*hipsparsestruct)->cooPerm_a;
    delete (*hipsparsestruct)->csr2csc_i;
    if ((*hipsparsestruct)->handle) PetscCallHIPSPARSE(hipsparseDestroy((*hipsparsestruct)->handle));
    if ((*hipsparsestruct)->jmap_d) PetscCallHIP(hipFree((*hipsparsestruct)->jmap_d));
    if ((*hipsparsestruct)->perm_d) PetscCallHIP(hipFree((*hipsparsestruct)->perm_d));
    PetscCall(PetscFree(*hipsparsestruct));
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

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct **trifactor)
{
  PetscFunctionBegin;
  if (*trifactor) {
    if ((*trifactor)->descr) PetscCallHIPSPARSE(hipsparseDestroyMatDescr((*trifactor)->descr));
    if ((*trifactor)->solveInfo) PetscCallHIPSPARSE(hipsparseDestroyCsrsvInfo((*trifactor)->solveInfo));
    PetscCall(CsrMatrix_Destroy(&(*trifactor)->csrMat));
    if ((*trifactor)->solveBuffer) PetscCallHIP(hipFree((*trifactor)->solveBuffer));
    if ((*trifactor)->AA_h) PetscCallHIP(hipHostFree((*trifactor)->AA_h));
    if ((*trifactor)->csr2cscBuffer) PetscCallHIP(hipFree((*trifactor)->csr2cscBuffer));
    PetscCall(PetscFree(*trifactor));
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
    if ((*matstruct)->alpha_one) PetscCallHIP(hipFree((*matstruct)->alpha_one));
    if ((*matstruct)->beta_zero) PetscCallHIP(hipFree((*matstruct)->beta_zero));
    if ((*matstruct)->beta_one) PetscCallHIP(hipFree((*matstruct)->beta_one));

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
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&fs->loTriFactorPtr));
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&fs->upTriFactorPtr));
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&fs->loTriFactorPtrTranspose));
    PetscCall(MatSeqAIJHIPSPARSEMultStruct_Destroy(&fs->upTriFactorPtrTranspose));
    delete fs->rpermIndices;
    delete fs->cpermIndices;
    delete fs->workVector;
    fs->rpermIndices = NULL;
    fs->cpermIndices = NULL;
    fs->workVector   = NULL;
    if (fs->a_band_d) PetscCallHIP(hipFree(fs->a_band_d));
    if (fs->i_band_d) PetscCallHIP(hipFree(fs->i_band_d));
    fs->init_dev_prop = PETSC_FALSE;
#if PETSC_PKG_HIP_VERSION_GE(4, 5, 0)
    PetscCallHIP(hipFree(fs->csrRowPtr));
    PetscCallHIP(hipFree(fs->csrColIdx));
    PetscCallHIP(hipFree(fs->csrVal));
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

    fs->createdTransposeSpSVDescr    = PETSC_FALSE;
    fs->updatedTransposeSpSVAnalysis = PETSC_FALSE;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors **trifactors)
{
  hipsparseHandle_t handle;

  PetscFunctionBegin;
  if (*trifactors) {
    PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(trifactors));
    if ((handle = (*trifactors)->handle)) PetscCallHIPSPARSE(hipsparseDestroy(handle));
    PetscCall(PetscFree(*trifactors));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct IJCompare {
  __host__ __device__ inline bool operator()(const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct IJEqual {
  __host__ __device__ inline bool operator()(const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() != t2.get<0>() || t1.get<1>() != t2.get<1>()) return false;
    return true;
  }
};

struct IJDiff {
  __host__ __device__ inline PetscInt operator()(const PetscInt &t1, const PetscInt &t2) { return t1 == t2 ? 0 : 1; }
};

struct IJSum {
  __host__ __device__ inline PetscInt operator()(const PetscInt &t1, const PetscInt &t2) { return t1 || t2; }
};

PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE_Basic(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJHIPSPARSE                  *cusp      = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJ                           *a         = (Mat_SeqAIJ *)A->data;
  THRUSTARRAY                          *cooPerm_v = NULL;
  thrust::device_ptr<const PetscScalar> d_v;
  CsrMatrix                            *matrix;
  PetscInt                              n;

  PetscFunctionBegin;
  PetscCheck(cusp, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIPSPARSE struct");
  PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIPSPARSE CsrMatrix");
  if (!cusp->cooPerm) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  matrix = (CsrMatrix *)cusp->mat->mat;
  PetscCheck(matrix->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIP memory");
  if (!v) {
    if (imode == INSERT_VALUES) thrust::fill(thrust::device, matrix->values->begin(), matrix->values->end(), 0.);
    goto finalize;
  }
  n = cusp->cooPerm->size();
  if (isHipMem(v)) d_v = thrust::device_pointer_cast(v);
  else {
    cooPerm_v = new THRUSTARRAY(n);
    cooPerm_v->assign(v, v + n);
    d_v = cooPerm_v->data();
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscScalar)));
  }
  PetscCall(PetscLogGpuTimeBegin());
  if (imode == ADD_VALUES) { /* ADD VALUES means add to existing ones */
    if (cusp->cooPerm_a) {   /* there are repeated entries in d_v[], and we need to add these them */
      THRUSTARRAY *cooPerm_w = new THRUSTARRAY(matrix->values->size());
      auto         vbit      = thrust::make_permutation_iterator(d_v, cusp->cooPerm->begin());
      /* thrust::reduce_by_key(keys_first,keys_last,values_first,keys_output,values_output)
        cooPerm_a = [0,0,1,2,3,4]. The length is n, number of nonozeros in d_v[].
        cooPerm_a is ordered. d_v[i] is the cooPerm_a[i]-th unique nonzero.
      */
      thrust::reduce_by_key(cusp->cooPerm_a->begin(), cusp->cooPerm_a->end(), vbit, thrust::make_discard_iterator(), cooPerm_w->begin(), thrust::equal_to<PetscInt>(), thrust::plus<PetscScalar>());
      thrust::transform(cooPerm_w->begin(), cooPerm_w->end(), matrix->values->begin(), matrix->values->begin(), thrust::plus<PetscScalar>());
      delete cooPerm_w;
    } else {
      /* all nonzeros in d_v[] are unique entries */
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->cooPerm->begin()), matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->cooPerm->end()), matrix->values->end()));
      thrust::for_each(zibit, zieit, VecHIPPlusEquals()); /* values[i] += d_v[cooPerm[i]]  */
    }
  } else {
    if (cusp->cooPerm_a) { /* repeated entries in COO, with INSERT_VALUES -> reduce */
      auto vbit = thrust::make_permutation_iterator(d_v, cusp->cooPerm->begin());
      thrust::reduce_by_key(cusp->cooPerm_a->begin(), cusp->cooPerm_a->end(), vbit, thrust::make_discard_iterator(), matrix->values->begin(), thrust::equal_to<PetscInt>(), thrust::plus<PetscScalar>());
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->cooPerm->begin()), matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->cooPerm->end()), matrix->values->end()));
      thrust::for_each(zibit, zieit, VecHIPEquals());
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
finalize:
  delete cooPerm_v;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  PetscCall(PetscInfo(A, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n", A->rmap->n, A->cmap->n, a->nz));
  PetscCall(PetscInfo(A, "Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(A, "Maximum nonzeros in any row is %" PetscInt_FMT "\n", a->rmax));
  a->reallocs = 0;
  A->info.mallocs += 0;
  A->info.nz_unneeded = 0;
  A->assembled = A->was_assembled = PETSC_TRUE;
  A->num_ass++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat A, PetscBool destroy)
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

PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE_Basic(Mat A, PetscCount n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJ          *a    = (Mat_SeqAIJ *)A->data;
  PetscInt             cooPerm_n, nzr = 0;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  cooPerm_n = cusp->cooPerm ? cusp->cooPerm->size() : 0;
  if (n != cooPerm_n) {
    delete cusp->cooPerm;
    delete cusp->cooPerm_a;
    cusp->cooPerm   = NULL;
    cusp->cooPerm_a = NULL;
  }
  if (n) {
    thrust::device_ptr<PetscInt> d_i, d_j;
    PetscInt                    *d_raw_i, *d_raw_j;
    PetscBool                    free_raw_i = PETSC_FALSE, free_raw_j = PETSC_FALSE;
    PetscMemType                 imtype, jmtype;

    PetscCall(PetscGetMemType(coo_i, &imtype));
    if (PetscMemTypeHost(imtype)) {
      PetscCallHIP(hipMalloc(&d_raw_i, sizeof(PetscInt) * n));
      PetscCallHIP(hipMemcpy(d_raw_i, coo_i, sizeof(PetscInt) * n, hipMemcpyHostToDevice));
      d_i        = thrust::device_pointer_cast(d_raw_i);
      free_raw_i = PETSC_TRUE;
      PetscCall(PetscLogCpuToGpu(1. * n * sizeof(PetscInt)));
    } else {
      d_i = thrust::device_pointer_cast(coo_i);
    }

    PetscCall(PetscGetMemType(coo_j, &jmtype));
    if (PetscMemTypeHost(jmtype)) { // MatSetPreallocationCOO_MPIAIJHIPSPARSE_Basic() passes device coo_i[] and host coo_j[]!
      PetscCallHIP(hipMalloc(&d_raw_j, sizeof(PetscInt) * n));
      PetscCallHIP(hipMemcpy(d_raw_j, coo_j, sizeof(PetscInt) * n, hipMemcpyHostToDevice));
      d_j        = thrust::device_pointer_cast(d_raw_j);
      free_raw_j = PETSC_TRUE;
      PetscCall(PetscLogCpuToGpu(1. * n * sizeof(PetscInt)));
    } else {
      d_j = thrust::device_pointer_cast(coo_j);
    }

    THRUSTINTARRAY ii(A->rmap->n);

    if (!cusp->cooPerm) cusp->cooPerm = new THRUSTINTARRAY(n);
    if (!cusp->cooPerm_a) cusp->cooPerm_a = new THRUSTINTARRAY(n);
    /* Ex.
      n = 6
      coo_i = [3,3,1,4,1,4]
      coo_j = [3,2,2,5,2,6]
    */
    auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i, d_j));
    auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i + n, d_j + n));

    PetscCall(PetscLogGpuTimeBegin());
    thrust::sequence(thrust::device, cusp->cooPerm->begin(), cusp->cooPerm->end(), 0);
    thrust::sort_by_key(fkey, ekey, cusp->cooPerm->begin(), IJCompare()); /* sort by row, then by col */
    (*cusp->cooPerm_a).assign(d_i, d_i + n);                              /* copy the sorted array */
    THRUSTINTARRAY w(d_j, d_j + n);
    /*
      d_i     = [1,1,3,3,4,4]
      d_j     = [2,2,2,3,5,6]
      cooPerm = [2,4,1,0,3,5]
    */
    auto nekey = thrust::unique(fkey, ekey, IJEqual()); /* unique (d_i, d_j) */
    /*
      d_i     = [1,3,3,4,4,x]
                            ^ekey
      d_j     = [2,2,3,5,6,x]
                           ^nekye
    */
    if (nekey == ekey) { /* all entries are unique */
      delete cusp->cooPerm_a;
      cusp->cooPerm_a = NULL;
    } else { /* Stefano: I couldn't come up with a more elegant algorithm */
      /* idea: any change in i or j in the (i,j) sequence implies a new nonzero */
      adjacent_difference(cusp->cooPerm_a->begin(), cusp->cooPerm_a->end(), cusp->cooPerm_a->begin(), IJDiff()); /* cooPerm_a: [1,1,3,3,4,4] => [1,0,1,0,1,0]*/
      adjacent_difference(w.begin(), w.end(), w.begin(), IJDiff());                                              /* w:         [2,2,2,3,5,6] => [2,0,0,1,1,1]*/
      (*cusp->cooPerm_a)[0] = 0;                                                                                 /* clear the first entry, though accessing an entry on device implies a hipMemcpy */
      w[0]                  = 0;
      thrust::transform(cusp->cooPerm_a->begin(), cusp->cooPerm_a->end(), w.begin(), cusp->cooPerm_a->begin(), IJSum());            /* cooPerm_a =          [0,0,1,1,1,1]*/
      thrust::inclusive_scan(cusp->cooPerm_a->begin(), cusp->cooPerm_a->end(), cusp->cooPerm_a->begin(), thrust::plus<PetscInt>()); /*cooPerm_a=[0,0,1,2,3,4]*/
    }
    thrust::counting_iterator<PetscInt> search_begin(0);
    thrust::upper_bound(d_i, nekey.get_iterator_tuple().get<0>(), /* binary search entries of [0,1,2,3,4,5,6) in ordered array d_i = [1,3,3,4,4], supposing A->rmap->n = 6. */
                        search_begin, search_begin + A->rmap->n,  /* return in ii[] the index of last position in d_i[] where value could be inserted without violating the ordering */
                        ii.begin());                              /* ii = [0,1,1,3,5,5]. A leading 0 will be added later */
    PetscCall(PetscLogGpuTimeEnd());

    PetscCall(MatSeqXAIJFreeAIJ(A, &a->a, &a->j, &a->i));
    a->singlemalloc = PETSC_FALSE;
    a->free_a       = PETSC_TRUE;
    a->free_ij      = PETSC_TRUE;
    PetscCall(PetscMalloc1(A->rmap->n + 1, &a->i));
    a->i[0] = 0; /* a->i = [0,0,1,1,3,5,5] */
    PetscCallHIP(hipMemcpy(a->i + 1, ii.data().get(), A->rmap->n * sizeof(PetscInt), hipMemcpyDeviceToHost));
    a->nz = a->maxnz = a->i[A->rmap->n];
    a->rmax          = 0;
    PetscCall(PetscMalloc1(a->nz, &a->a));
    PetscCall(PetscMalloc1(a->nz, &a->j));
    PetscCallHIP(hipMemcpy(a->j, thrust::raw_pointer_cast(d_j), a->nz * sizeof(PetscInt), hipMemcpyDeviceToHost));
    if (!a->ilen) PetscCall(PetscMalloc1(A->rmap->n, &a->ilen));
    if (!a->imax) PetscCall(PetscMalloc1(A->rmap->n, &a->imax));
    for (PetscInt i = 0; i < A->rmap->n; i++) {
      const PetscInt nnzr = a->i[i + 1] - a->i[i];
      nzr += (PetscInt) !!(nnzr);
      a->ilen[i] = a->imax[i] = nnzr;
      a->rmax                 = PetscMax(a->rmax, nnzr);
    }
    a->nonzerorowcnt = nzr;
    A->preallocated  = PETSC_TRUE;
    PetscCall(PetscLogGpuToCpu((A->rmap->n + a->nz) * sizeof(PetscInt)));
    PetscCall(MatMarkDiagonal_SeqAIJ(A));
    if (free_raw_i) PetscCallHIP(hipFree(d_raw_i));
    if (free_raw_j) PetscCallHIP(hipFree(d_raw_j));
  } else PetscCall(MatSeqAIJSetPreallocation(A, 0, NULL));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  /* We want to allocate the HIPSPARSE struct for matvec now.
     The code is so convoluted now that I prefer to copy zeros */
  PetscCall(PetscArrayzero(a->a, a->nz));
  PetscCall(MatCheckCompressedRow(A, nzr, &a->compressedrow, a->i, A->rmap->n, 0.6));
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_SeqAIJ          *seq;
  Mat_SeqAIJHIPSPARSE *dev;
  PetscBool            coo_basic = PETSC_TRUE;
  PetscMemType         mtype     = PETSC_MEMTYPE_DEVICE;

  PetscFunctionBegin;
  PetscCall(MatResetPreallocationCOO_SeqAIJ(mat));
  PetscCall(MatResetPreallocationCOO_SeqAIJHIPSPARSE(mat));
  if (coo_i) {
    PetscCall(PetscGetMemType(coo_i, &mtype));
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k = 0; k < coo_n; k++) {
        if (coo_i[k] < 0 || coo_j[k] < 0) {
          coo_basic = PETSC_FALSE;
          break;
        }
      }
    }
  }

  if (coo_basic) { /* i,j are on device or do not contain negative indices */
    PetscCall(MatSetPreallocationCOO_SeqAIJHIPSPARSE_Basic(mat, coo_n, coo_i, coo_j));
  } else {
    PetscCall(MatSetPreallocationCOO_SeqAIJ(mat, coo_n, coo_i, coo_j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    PetscCall(MatSeqAIJHIPSPARSECopyToGPU(mat));
    seq = static_cast<Mat_SeqAIJ *>(mat->data);
    dev = static_cast<Mat_SeqAIJHIPSPARSE *>(mat->spptr);
    PetscCallHIP(hipMalloc((void **)&dev->jmap_d, (seq->nz + 1) * sizeof(PetscCount)));
    PetscCallHIP(hipMemcpy(dev->jmap_d, seq->jmap, (seq->nz + 1) * sizeof(PetscCount), hipMemcpyHostToDevice));
    PetscCallHIP(hipMalloc((void **)&dev->perm_d, seq->Atot * sizeof(PetscCount)));
    PetscCallHIP(hipMemcpy(dev->perm_d, seq->perm, seq->Atot * sizeof(PetscCount), hipMemcpyHostToDevice));
    dev->use_extended_coo = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

__global__ static void MatAddCOOValues(const PetscScalar kv[], PetscCount nnz, const PetscCount jmap[], const PetscCount perm[], InsertMode imode, PetscScalar a[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < nnz; i += grid_size) {
    PetscScalar sum = 0.0;
    for (PetscCount k = jmap[i]; k < jmap[i + 1]; k++) sum += kv[perm[k]];
    a[i] = (imode == INSERT_VALUES ? 0.0 : a[i]) + sum;
  }
}

PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJ          *seq  = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJHIPSPARSE *dev  = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  PetscCount           Annz = seq->nz;
  PetscMemType         memtype;
  const PetscScalar   *v1 = v;
  PetscScalar         *Aa;

  PetscFunctionBegin;
  if (dev->use_extended_coo) {
    PetscCall(PetscGetMemType(v, &memtype));
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
      PetscCallHIP(hipMalloc((void **)&v1, seq->coo_n * sizeof(PetscScalar)));
      PetscCallHIP(hipMemcpy((void *)v1, v, seq->coo_n * sizeof(PetscScalar), hipMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) PetscCall(MatSeqAIJHIPSPARSEGetArrayWrite(A, &Aa));
    else PetscCall(MatSeqAIJHIPSPARSEGetArray(A, &Aa));

    if (Annz) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(MatAddCOOValues), dim3((Annz + 255) / 256), dim3(256), 0, PetscDefaultHipStream, v1, Annz, dev->jmap_d, dev->perm_d, imode, Aa);
      PetscCallHIP(hipPeekAtLastError());
    }

    if (imode == INSERT_VALUES) PetscCall(MatSeqAIJHIPSPARSERestoreArrayWrite(A, &Aa));
    else PetscCall(MatSeqAIJHIPSPARSERestoreArray(A, &Aa));

    if (PetscMemTypeHost(memtype)) PetscCallHIP(hipFree((void *)v1));
  } else {
    PetscCall(MatSetValuesCOO_SeqAIJHIPSPARSE_Basic(A, v, imode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    MatSeqAIJHIPSPARSEGetIJ - returns the device row storage `i` and `j` indices for `MATSEQAIJHIPSPARSE` matrices.

    Not Collective

    Input Parameters:
+   A - the matrix
-   compressed - `PETSC_TRUE` or `PETSC_FALSE` indicating the matrix data structure should be always returned in compressed form

    Output Parameters:
+   i - the CSR row pointers
-   j - the CSR column indices

    Level: developer

    Note:
      When compressed is true, the CSR structure does not contain empty rows

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSERestoreIJ()`, `MatSeqAIJHIPSPARSEGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetIJ(Mat A, PetscBool compressed, const int **i, const int **j)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJ          *a    = (Mat_SeqAIJ *)A->data;
  CsrMatrix           *csr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!i || !j) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCheck(cusp->format != MAT_HIPSPARSE_ELL && cusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix *)cusp->mat->mat;
  if (i) {
    if (!compressed && a->compressedrow.use) { /* need full row offset */
      if (!cusp->rowoffsets_gpu) {
        cusp->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n + 1);
        cusp->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
        PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
      }
      *i = cusp->rowoffsets_gpu->data().get();
    } else *i = csr->row_offsets->data().get();
  }
  if (j) *j = csr->column_indices->data().get();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    MatSeqAIJHIPSPARSERestoreIJ - restore the device row storage `i` and `j` indices obtained with `MatSeqAIJHIPSPARSEGetIJ()`

    Not Collective

    Input Parameters:
+   A - the matrix
.   compressed - `PETSC_TRUE` or `PETSC_FALSE` indicating the matrix data structure should be always returned in compressed form
.   i - the CSR row pointers
-   j - the CSR column indices

    Level: developer

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetIJ()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreIJ(Mat A, PetscBool compressed, const int **i, const int **j)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  if (i) *i = NULL;
  if (j) *j = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatSeqAIJHIPSPARSEGetArrayRead - gives read-only access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

   Not Collective

   Input Parameter:
.   A - a `MATSEQAIJHIPSPARSE` matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Note:
   May trigger host-device copies if the up-to-date matrix data is on host

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`, `MatSeqAIJHIPSPARSEGetArrayWrite()`, `MatSeqAIJHIPSPARSERestoreArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArrayRead(Mat A, const PetscScalar **a)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  CsrMatrix           *csr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCheck(cusp->format != MAT_HIPSPARSE_ELL && cusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix *)cusp->mat->mat;
  PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIP memory");
  *a = csr->values->data().get();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatSeqAIJHIPSPARSERestoreArrayRead - restore the read-only access array obtained from `MatSeqAIJHIPSPARSEGetArrayRead()`

   Not Collective

   Input Parameters:
+   A - a `MATSEQAIJHIPSPARSE` matrix
-   a - pointer to the device data

   Level: developer

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayRead()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatSeqAIJHIPSPARSEGetArray - gives read-write access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

   Not Collective

   Input Parameter:
.   A - a `MATSEQAIJHIPSPARSE` matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Note:
   May trigger host-device copies if up-to-date matrix data is on host

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayRead()`, `MatSeqAIJHIPSPARSEGetArrayWrite()`, `MatSeqAIJHIPSPARSERestoreArray()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArray(Mat A, PetscScalar **a)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  CsrMatrix           *csr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCheck(cusp->format != MAT_HIPSPARSE_ELL && cusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatSeqAIJHIPSPARSECopyToGPU(A));
  PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix *)cusp->mat->mat;
  PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIP memory");
  *a             = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
   MatSeqAIJHIPSPARSERestoreArray - restore the read-write access array obtained from `MatSeqAIJHIPSPARSEGetArray()`

   Not Collective

   Input Parameters:
+   A - a `MATSEQAIJHIPSPARSE` matrix
-   a - pointer to the device data

   Level: developer

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatSeqAIJHIPSPARSEGetArrayWrite - gives write access to the array where the device data for a `MATSEQAIJHIPSPARSE` matrix is stored

   Not Collective

   Input Parameter:
.   A - a `MATSEQAIJHIPSPARSE` matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Note:
   Does not trigger host-device copies and flags data validity on the GPU

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArray()`, `MatSeqAIJHIPSPARSEGetArrayRead()`, `MatSeqAIJHIPSPARSERestoreArrayWrite()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSEGetArrayWrite(Mat A, PetscScalar **a)
{
  Mat_SeqAIJHIPSPARSE *cusp = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  CsrMatrix           *csr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCheck(cusp->format != MAT_HIPSPARSE_ELL && cusp->format != MAT_HIPSPARSE_HYB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix *)cusp->mat->mat;
  PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing HIP memory");
  *a             = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(MatSeqAIJHIPSPARSEInvalidateTranspose(A, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatSeqAIJHIPSPARSERestoreArrayWrite - restore the write-only access array obtained from `MatSeqAIJHIPSPARSEGetArrayWrite()`

   Not Collective

   Input Parameters:
+   A - a `MATSEQAIJHIPSPARSE` matrix
-   a - pointer to the device data

   Level: developer

.seealso: [](chapter_matrices), `Mat`, `MatSeqAIJHIPSPARSEGetArrayWrite()`
@*/
PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCheckTypeName(A, MATSEQAIJHIPSPARSE);
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
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

/* merges two SeqAIJHIPSPARSE matrices A, B by concatenating their rows. [A';B']' operation in matlab notation */
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
  PetscValidPointer(C, 4);
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
    PetscCallHIP(hipMalloc((void **)&(Cmat->alpha_one), sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&(Cmat->beta_zero), sizeof(PetscScalar)));
    PetscCallHIP(hipMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar)));
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
    Ccsr->row_offsets    = new THRUSTINTARRAY32(m + 1);
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values         = new THRUSTARRAY(c->nz);
    Ccsr->num_entries    = c->nz;
    Ccusp->cooPerm       = new THRUSTINTARRAY(c->nz);
    if (c->nz) {
      auto              Acoo = new THRUSTINTARRAY32(Annz);
      auto              Bcoo = new THRUSTINTARRAY32(Bnnz);
      auto              Ccoo = new THRUSTINTARRAY32(c->nz);
      THRUSTINTARRAY32 *Aroff, *Broff;

      if (a->compressedrow.use) { /* need full row offset */
        if (!Acusp->rowoffsets_gpu) {
          Acusp->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n + 1);
          Acusp->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
        }
        Aroff = Acusp->rowoffsets_gpu;
      } else Aroff = Acsr->row_offsets;
      if (b->compressedrow.use) { /* need full row offset */
        if (!Bcusp->rowoffsets_gpu) {
          Bcusp->rowoffsets_gpu = new THRUSTINTARRAY32(B->rmap->n + 1);
          Bcusp->rowoffsets_gpu->assign(b->i, b->i + B->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((B->rmap->n + 1) * sizeof(PetscInt)));
        }
        Broff = Bcusp->rowoffsets_gpu;
      } else Broff = Bcsr->row_offsets;
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallHIPSPARSE(hipsparseXcsr2coo(Acusp->handle, Aroff->data().get(), Annz, m, Acoo->data().get(), HIPSPARSE_INDEX_BASE_ZERO));
      PetscCallHIPSPARSE(hipsparseXcsr2coo(Bcusp->handle, Broff->data().get(), Bnnz, m, Bcoo->data().get(), HIPSPARSE_INDEX_BASE_ZERO));
      /* Issues when using bool with large matrices on SUMMIT 10.2.89 */
      auto Aperm = thrust::make_constant_iterator(1);
      auto Bperm = thrust::make_constant_iterator(0);
      auto Bcib  = thrust::make_transform_iterator(Bcsr->column_indices->begin(), Shift(A->cmap->n));
      auto Bcie  = thrust::make_transform_iterator(Bcsr->column_indices->end(), Shift(A->cmap->n));
      auto wPerm = new THRUSTINTARRAY32(Annz + Bnnz);
      auto Azb   = thrust::make_zip_iterator(thrust::make_tuple(Acoo->begin(), Acsr->column_indices->begin(), Acsr->values->begin(), Aperm));
      auto Aze   = thrust::make_zip_iterator(thrust::make_tuple(Acoo->end(), Acsr->column_indices->end(), Acsr->values->end(), Aperm));
      auto Bzb   = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->begin(), Bcib, Bcsr->values->begin(), Bperm));
      auto Bze   = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->end(), Bcie, Bcsr->values->end(), Bperm));
      auto Czb   = thrust::make_zip_iterator(thrust::make_tuple(Ccoo->begin(), Ccsr->column_indices->begin(), Ccsr->values->begin(), wPerm->begin()));
      auto p1    = Ccusp->cooPerm->begin();
      auto p2    = Ccusp->cooPerm->begin();
      thrust::advance(p2, Annz);
      PetscCallThrust(thrust::merge(thrust::device, Azb, Aze, Bzb, Bze, Czb, IJCompare4()));
      auto cci = thrust::make_counting_iterator(zero);
      auto cce = thrust::make_counting_iterator(c->nz);
#if 0 //Errors on SUMMIT cuda 11.1.0
      PetscCallThrust(thrust::partition_copy(thrust::device, cci, cce, wPerm->begin(), p1, p2, thrust::identity<int>()));
#else
      auto pred = thrust::identity<int>();
      PetscCallThrust(thrust::copy_if(thrust::device, cci, cce, wPerm->begin(), p1, pred));
      PetscCallThrust(thrust::remove_copy_if(thrust::device, cci, cce, wPerm->begin(), p2, pred));
#endif
      PetscCallHIPSPARSE(hipsparseXcoo2csr(Ccusp->handle, Ccoo->data().get(), c->nz, m, Ccsr->row_offsets->data().get(), HIPSPARSE_INDEX_BASE_ZERO));
      PetscCall(PetscLogGpuTimeEnd());
      delete wPerm;
      delete Acoo;
      delete Bcoo;
      delete Ccoo;
      PetscCallHIPSPARSE(hipsparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, Ccsr->num_entries, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));

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
        CcsrT->row_offsets            = new THRUSTINTARRAY32(n + 1);
        CcsrT->column_indices         = new THRUSTINTARRAY32(c->nz);
        CcsrT->values                 = new THRUSTARRAY(c->nz);

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
        PetscCallHIP(hipMalloc((void **)&(CmatT->alpha_one), sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&(CmatT->beta_zero), sizeof(PetscScalar)));
        PetscCallHIP(hipMalloc((void **)&(CmatT->beta_one), sizeof(PetscScalar)));
        PetscCallHIP(hipMemcpy(CmatT->alpha_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(CmatT->beta_zero, &PETSC_HIPSPARSE_ZERO, sizeof(PetscScalar), hipMemcpyHostToDevice));
        PetscCallHIP(hipMemcpy(CmatT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar), hipMemcpyHostToDevice));

        PetscCallHIPSPARSE(hipsparseCreateCsr(&CmatT->matDescr, CcsrT->num_rows, CcsrT->num_cols, CcsrT->num_entries, CcsrT->row_offsets->data().get(), CcsrT->column_indices->data().get(), CcsrT->values->data().get(), HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, hipsparse_scalartype));
        Ccusp->matTranspose = CmatT;
      }
    }

    c->singlemalloc = PETSC_FALSE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
    PetscCall(PetscMalloc1(m + 1, &c->i));
    PetscCall(PetscMalloc1(c->nz, &c->j));
    if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
      THRUSTINTARRAY ii(Ccsr->row_offsets->size());
      THRUSTINTARRAY jj(Ccsr->column_indices->size());
      ii = *Ccsr->row_offsets;
      jj = *Ccsr->column_indices;
      PetscCallHIP(hipMemcpy(c->i, ii.data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
      PetscCallHIP(hipMemcpy(c->j, jj.data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    } else {
      PetscCallHIP(hipMemcpy(c->i, Ccsr->row_offsets->data().get(), Ccsr->row_offsets->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
      PetscCallHIP(hipMemcpy(c->j, Ccsr->column_indices->data().get(), Ccsr->column_indices->size() * sizeof(PetscInt), hipMemcpyDeviceToHost));
    }
    PetscCall(PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size()) * sizeof(PetscInt)));
    PetscCall(PetscMalloc1(m, &c->ilen));
    PetscCall(PetscMalloc1(m, &c->imax));
    c->maxnz         = c->nz;
    c->nonzerorowcnt = 0;
    c->rmax          = 0;
    for (i = 0; i < m; i++) {
      const PetscInt nn = c->i[i + 1] - c->i[i];
      c->ilen[i] = c->imax[i] = nn;
      c->nonzerorowcnt += (PetscInt) !!nn;
      c->rmax = PetscMax(c->rmax, nn);
    }
    PetscCall(MatMarkDiagonal_SeqAIJ(*C));
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
      PetscCheck(Ccusp->cooPerm, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing cooPerm");
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
      PetscCheck(Ccusp->cooPerm->size() == Ccsr->values->size(), PETSC_COMM_SELF, PETSC_ERR_COR, "permSize %" PetscInt_FMT " != %" PetscInt_FMT, (PetscInt)Ccusp->cooPerm->size(), (PetscInt)Ccsr->values->size());
      auto pmid = Ccusp->cooPerm->begin();
      thrust::advance(pmid, Acsr->num_entries);
      PetscCall(PetscLogGpuTimeBegin());
      auto zibait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->begin(), thrust::make_permutation_iterator(Ccsr->values->begin(), Ccusp->cooPerm->begin())));
      auto zieait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->end(), thrust::make_permutation_iterator(Ccsr->values->begin(), pmid)));
      thrust::for_each(zibait, zieait, VecHIPEquals());
      auto zibbit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->begin(), thrust::make_permutation_iterator(Ccsr->values->begin(), pmid)));
      auto ziebit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->end(), thrust::make_permutation_iterator(Ccsr->values->begin(), Ccusp->cooPerm->end())));
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
  bool               dmem;
  const PetscScalar *av;

  PetscFunctionBegin;
  dmem = isHipMem(v);
  PetscCall(MatSeqAIJHIPSPARSEGetArrayRead(A, &av));
  if (n && idx) {
    THRUSTINTARRAY widx(n);
    widx.assign(idx, idx + n);
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));

    THRUSTARRAY                    *w = NULL;
    thrust::device_ptr<PetscScalar> dv;
    if (dmem) dv = thrust::device_pointer_cast(v);
    else {
      w  = new THRUSTARRAY(n);
      dv = w->data();
    }
    thrust::device_ptr<const PetscScalar> dav = thrust::device_pointer_cast(av);

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav, widx.begin()), dv));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav, widx.end()), dv + n));
    thrust::for_each(zibit, zieit, VecHIPEquals());
    if (w) PetscCallHIP(hipMemcpy(v, w->data().get(), n * sizeof(PetscScalar), hipMemcpyDeviceToHost));
    delete w;
  } else PetscCallHIP(hipMemcpy(v, av, n * sizeof(PetscScalar), dmem ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost));

  if (!dmem) PetscCall(PetscLogCpuToGpu(n * sizeof(PetscScalar)));
  PetscCall(MatSeqAIJHIPSPARSERestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}
