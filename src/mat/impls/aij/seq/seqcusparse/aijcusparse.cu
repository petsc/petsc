/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the CUSPARSE library,
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <thrust/async/for_each.h>

const char *const MatCUSPARSEStorageFormats[]    = {"CSR","ELL","HYB","MatCUSPARSEStorageFormat","MAT_CUSPARSE_",0};
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  /* The following are copied from cusparse.h in CUDA-11.0. In MatCUSPARSESpMVAlgorithms[] etc, we copy them in
    0-based integer value order, since we want to use PetscOptionsEnum() to parse user command line options for them.

  typedef enum {
      CUSPARSE_MV_ALG_DEFAULT = 0,
      CUSPARSE_COOMV_ALG      = 1,
      CUSPARSE_CSRMV_ALG1     = 2,
      CUSPARSE_CSRMV_ALG2     = 3
  } cusparseSpMVAlg_t;

  typedef enum {
      CUSPARSE_MM_ALG_DEFAULT     CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_ALG_DEFAULT) = 0,
      CUSPARSE_COOMM_ALG1         CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG1)    = 1,
      CUSPARSE_COOMM_ALG2         CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG2)    = 2,
      CUSPARSE_COOMM_ALG3         CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG3)    = 3,
      CUSPARSE_CSRMM_ALG1         CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_CSR_ALG1)    = 4,
      CUSPARSE_SPMM_ALG_DEFAULT = 0,
      CUSPARSE_SPMM_COO_ALG1    = 1,
      CUSPARSE_SPMM_COO_ALG2    = 2,
      CUSPARSE_SPMM_COO_ALG3    = 3,
      CUSPARSE_SPMM_COO_ALG4    = 5,
      CUSPARSE_SPMM_CSR_ALG1    = 4,
      CUSPARSE_SPMM_CSR_ALG2    = 6,
  } cusparseSpMMAlg_t;

  typedef enum {
      CUSPARSE_CSR2CSC_ALG1 = 1, // faster than V2 (in general), deterministc
      CUSPARSE_CSR2CSC_ALG2 = 2  // low memory requirement, non-deterministc
  } cusparseCsr2CscAlg_t;
  */
  const char *const MatCUSPARSESpMVAlgorithms[]    = {"MV_ALG_DEFAULT","COOMV_ALG", "CSRMV_ALG1","CSRMV_ALG2", "cusparseSpMVAlg_t","CUSPARSE_",0};
  const char *const MatCUSPARSESpMMAlgorithms[]    = {"ALG_DEFAULT","COO_ALG1","COO_ALG2","COO_ALG3","CSR_ALG1","COO_ALG4","CSR_ALG2","cusparseSpMMAlg_t","CUSPARSE_SPMM_",0};
  const char *const MatCUSPARSECsr2CscAlgorithms[] = {"INVALID"/*cusparse does not have enum 0! We created one*/,"ALG1","ALG2","cusparseCsr2CscAlg_t","CUSPARSE_CSR2CSC_",0};
#endif

static PetscErrorCode MatICCFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJCUSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatSolve_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolve_SeqAIJCUSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat);
static PetscErrorCode MatAXPY_SeqAIJCUSPARSE(Mat,PetscScalar,Mat,MatStructure);
static PetscErrorCode MatScale_SeqAIJCUSPARSE(Mat,PetscScalar);
static PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultHermitianTranspose_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultAddKernel_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec,PetscBool,PetscBool);

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix**);
static PetscErrorCode MatSeqAIJCUSPARSEMultStruct_Destroy(Mat_SeqAIJCUSPARSETriFactorStruct**);
static PetscErrorCode MatSeqAIJCUSPARSEMultStruct_Destroy(Mat_SeqAIJCUSPARSEMultStruct**,MatCUSPARSEStorageFormat);
static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Destroy(Mat_SeqAIJCUSPARSETriFactors**);
static PetscErrorCode MatSeqAIJCUSPARSE_Destroy(Mat_SeqAIJCUSPARSE**);

PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat);
static PetscErrorCode MatSeqAIJCUSPARSECopyFromGPU(Mat);
static PetscErrorCode MatSeqAIJCUSPARSEInvalidateTranspose(Mat,PetscBool);

PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat,const PetscScalar[],InsertMode);

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJCUSPARSE(Mat,PetscInt,const PetscInt[],PetscScalar[]);

PetscErrorCode MatCUSPARSESetStream(Mat A,const cudaStream_t stream)
{
  cusparseStatus_t   stat;
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (!cusparsestruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  cusparsestruct->stream = stream;
  stat = cusparseSetStream(cusparsestruct->handle,cusparsestruct->stream);CHKERRCUSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPARSESetHandle(Mat A,const cusparseHandle_t handle)
{
  cusparseStatus_t   stat;
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (!cusparsestruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (cusparsestruct->handle != handle) {
    if (cusparsestruct->handle) {
      stat = cusparseDestroy(cusparsestruct->handle);CHKERRCUSPARSE(stat);
    }
    cusparsestruct->handle = handle;
  }
  stat = cusparseSetPointerMode(cusparsestruct->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPARSEClearHandle(Mat A)
{
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg || !cusparsestruct) PetscFunctionReturn(0);
  if (cusparsestruct->handle) cusparsestruct->handle = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_cusparse(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUSPARSE;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERCUSPARSE = "cusparse" - A matrix type providing triangular solvers for seq matrices
  on a single GPU of type, seqaijcusparse, aijcusparse, or seqaijcusp, aijcusp. Currently supported
  algorithms are ILU(k) and ICC(k). Typically, deeper factorizations (larger k) results in poorer
  performance in the triangular solves. Full LU, and Cholesky decompositions can be solved through the
  CUSPARSE triangular solve algorithm. However, the performance can be quite poor and thus these
  algorithms are not recommended. This class does NOT support direct solver operations.

  Level: beginner

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatCreateSeqAIJCUSPARSE(), MATAIJCUSPARSE, MatCreateAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijcusparse_cusparse(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  ierr = MatSetType(*B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSE;
    ierr = PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
    ierr = PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILU]);CHKERRQ(ierr);
    ierr = PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILUDT]);CHKERRQ(ierr);
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJCUSPARSE;
    ierr = PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
    ierr = PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ICC]);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSE Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  (*B)->canuseordering = PETSC_TRUE;
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_cusparse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatCUSPARSESetFormat_SeqAIJCUSPARSE(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT:
    cusparsestruct->format = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparsestruct->format = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. MAT_CUSPARSE_MULT and MAT_CUSPARSE_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

/*@
   MatCUSPARSESetFormat - Sets the storage format of CUSPARSE matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for MPIAIJCUSPARSE matrices.
   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJCUSPARSE
.  op - MatCUSPARSEFormatOperation. SEQAIJCUSPARSE matrices support MAT_CUSPARSE_MULT and MAT_CUSPARSE_ALL. MPIAIJCUSPARSE matrices support MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_OFFDIAG, and MAT_CUSPARSE_ALL.
-  format - MatCUSPARSEStorageFormat (one of MAT_CUSPARSE_CSR, MAT_CUSPARSE_ELL, MAT_CUSPARSE_HYB. The latter two require CUDA 4.2)

   Output Parameter:

   Level: intermediate

.seealso: MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
@*/
PetscErrorCode MatCUSPARSESetFormat(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  ierr = PetscTryMethod(A,"MatCUSPARSESetFormat_C",(Mat,MatCUSPARSEFormatOperation,MatCUSPARSEStorageFormat),(A,op,format));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqAIJCUSPARSE(Mat A,MatOption op,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
    case MAT_FORM_EXPLICIT_TRANSPOSE:
      /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
      if (A->form_explicit_transpose && !flg) {ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);}
      A->form_explicit_transpose = flg;
      break;
    default:
      ierr = MatSetOption_SeqAIJ(A,op,flg);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(Mat A);

static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row,iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSECopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_SeqAIJCUSPARSE       *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJCUSPARSE options");CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT,format);CHKERRQ(ierr);}

    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV and TriSolve",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);}
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    ierr = PetscOptionsEnum("-mat_cusparse_spmv_alg","sets cuSPARSE algorithm used in sparse-mat dense-vector multiplication (SpMV)",
                            "cusparseSpMVAlg_t",MatCUSPARSESpMVAlgorithms,(PetscEnum)cusparsestruct->spmvAlg,(PetscEnum*)&cusparsestruct->spmvAlg,&flg);CHKERRQ(ierr);
    /* If user did use this option, check its consistency with cuSPARSE, since PetscOptionsEnum() sets enum values based on their position in MatCUSPARSESpMVAlgorithms[] */
#if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
    if (flg && CUSPARSE_SPMV_CSR_ALG1 != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");
#else
    if (flg && CUSPARSE_CSRMV_ALG1 != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");
#endif
    ierr = PetscOptionsEnum("-mat_cusparse_spmm_alg","sets cuSPARSE algorithm used in sparse-mat dense-mat multiplication (SpMM)",
                            "cusparseSpMMAlg_t",MatCUSPARSESpMMAlgorithms,(PetscEnum)cusparsestruct->spmmAlg,(PetscEnum*)&cusparsestruct->spmmAlg,&flg);CHKERRQ(ierr);
    if (flg && CUSPARSE_SPMM_CSR_ALG1 != 4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMMAlg_t has been changed but PETSc has not been updated accordingly");

    ierr = PetscOptionsEnum("-mat_cusparse_csr2csc_alg","sets cuSPARSE algorithm used in converting CSR matrices to CSC matrices",
                            "cusparseCsr2CscAlg_t",MatCUSPARSECsr2CscAlgorithms,(PetscEnum)cusparsestruct->csr2cscAlg,(PetscEnum*)&cusparsestruct->csr2cscAlg,&flg);CHKERRQ(ierr);
    if (flg && CUSPARSE_CSR2CSC_ALG1 != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseCsr2CscAlg_t has been changed but PETSc has not been updated accordingly");
   #endif
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors);CHKERRQ(ierr);
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors);CHKERRQ(ierr);
  ierr = MatICCFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  cusparseStatus_t                  stat;
  const PetscInt                    *ai = a->i,*aj = a->j,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiLo, *AjLo;
  PetscInt                          i,nz, nzLower, offset, rowOffset;
  PetscErrorCode                    ierr;
  cudaError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];
      if (!loTriFactor) {
        PetscScalar                       *AALo;

        cerr = cudaMallocHost((void**) &AALo, nzLower*sizeof(PetscScalar));CHKERRCUDA(cerr);

        /* Allocate Space for the lower triangular matrix */
        cerr = cudaMallocHost((void**) &AiLo, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMallocHost((void**) &AjLo, nzLower*sizeof(PetscInt));CHKERRCUDA(cerr);

        /* Fill the lower triangular matrix */
        AiLo[0]  = (PetscInt) 0;
        AiLo[n]  = nzLower;
        AjLo[0]  = (PetscInt) 0;
        AALo[0]  = (MatScalar) 1.0;
        v        = aa;
        vi       = aj;
        offset   = 1;
        rowOffset= 1;
        for (i=1; i<n; i++) {
          nz = ai[i+1] - ai[i];
          /* additional 1 for the term on the diagonal */
          AiLo[i]    = rowOffset;
          rowOffset += nz+1;

          ierr = PetscArraycpy(&(AjLo[offset]), vi, nz);CHKERRQ(ierr);
          ierr = PetscArraycpy(&(AALo[offset]), v, nz);CHKERRQ(ierr);

          offset      += nz;
          AjLo[offset] = (PetscInt) i;
          AALo[offset] = (MatScalar) 1.0;
          offset      += 1;

          v  += nz;
          vi += nz;
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&loTriFactor);CHKERRQ(ierr);
        loTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
        /* Create the matrix description */
        stat = cusparseCreateMatDescr(&loTriFactor->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(loTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
       #else
        stat = cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSPARSE(stat);
       #endif
        stat = cusparseSetMatFillMode(loTriFactor->descr, CUSPARSE_FILL_MODE_LOWER);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatDiagType(loTriFactor->descr, CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSPARSE(stat);

        /* set the operation */
        loTriFactor->solveOp = CUSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat = new CsrMatrix;
        loTriFactor->csrMat->num_rows = n;
        loTriFactor->csrMat->num_cols = n;
        loTriFactor->csrMat->num_entries = nzLower;

        loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
        loTriFactor->csrMat->row_offsets->assign(AiLo, AiLo+n+1);

        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzLower);
        loTriFactor->csrMat->column_indices->assign(AjLo, AjLo+nzLower);

        loTriFactor->csrMat->values = new THRUSTARRAY(nzLower);
        loTriFactor->csrMat->values->assign(AALo, AALo+nzLower);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = cusparse_create_analysis_info(&loTriFactor->solveInfo);CHKERRCUSPARSE(stat);
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactor->solveOp,
                                       loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                       loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                       &loTriFactor->solveBufferSize);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize);CHKERRCUDA(cerr);
      #endif

        /* perform the solve analysis */
        stat = cusparse_analysis(cusparseTriFactors->handle, loTriFactor->solveOp,
                                 loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                 loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                 loTriFactor->csrMat->column_indices->data().get(),
                               #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 loTriFactor->solveInfo,
                                 loTriFactor->solvePolicy, loTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                               #else
                                 loTriFactor->solveInfo);CHKERRCUSPARSE(stat);
                               #endif
        cerr = WaitForCUDA();CHKERRCUDA(cerr);
        ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;
        loTriFactor->AA_h = AALo;
        cerr = cudaFreeHost(AiLo);CHKERRCUDA(cerr);
        cerr = cudaFreeHost(AjLo);CHKERRCUDA(cerr);
        ierr = PetscLogCpuToGpu((n+1+nzLower)*sizeof(int)+nzLower*sizeof(PetscScalar));CHKERRQ(ierr);
      } else { /* update values only */
        if (!loTriFactor->AA_h) {
          cerr = cudaMallocHost((void**) &loTriFactor->AA_h, nzLower*sizeof(PetscScalar));CHKERRCUDA(cerr);
        }
        /* Fill the lower triangular matrix */
        loTriFactor->AA_h[0]  = 1.0;
        v        = aa;
        vi       = aj;
        offset   = 1;
        for (i=1; i<n; i++) {
          nz = ai[i+1] - ai[i];
          ierr = PetscArraycpy(&(loTriFactor->AA_h[offset]), v, nz);CHKERRQ(ierr);
          offset      += nz;
          loTriFactor->AA_h[offset] = 1.0;
          offset      += 1;
          v  += nz;
        }
        loTriFactor->csrMat->values->assign(loTriFactor->AA_h, loTriFactor->AA_h+nzLower);
        ierr = PetscLogCpuToGpu(nzLower*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEBuildILUUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t                  stat;
  const PetscInt                    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiUp, *AjUp;
  PetscInt                          i,nz, nzUpper, offset;
  PetscErrorCode                    ierr;
  cudaError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];
      if (!upTriFactor) {
        PetscScalar *AAUp;

        cerr = cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);

        /* Allocate Space for the upper triangular matrix */
        cerr = cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUDA(cerr);

        /* Fill the upper triangular matrix */
        AiUp[0]=(PetscInt) 0;
        AiUp[n]=nzUpper;
        offset = nzUpper;
        for (i=n-1; i>=0; i--) {
          v  = aa + adiag[i+1] + 1;
          vi = aj + adiag[i+1] + 1;

          /* number of elements NOT on the diagonal */
          nz = adiag[i] - adiag[i+1]-1;

          /* decrement the offset */
          offset -= (nz+1);

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt) i;
          AAUp[offset] = (MatScalar)1./v[nz];
          AiUp[i]      = AiUp[i+1] - (nz+1);

          ierr = PetscArraycpy(&(AjUp[offset+1]), vi, nz);CHKERRQ(ierr);
          ierr = PetscArraycpy(&(AAUp[offset+1]), v, nz);CHKERRQ(ierr);
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&upTriFactor);CHKERRQ(ierr);
        upTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = cusparseCreateMatDescr(&upTriFactor->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(upTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
       #else
        stat = cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSPARSE(stat);
       #endif
        stat = cusparseSetMatFillMode(upTriFactor->descr, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatDiagType(upTriFactor->descr, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSPARSE(stat);

        /* set the operation */
        upTriFactor->solveOp = CUSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        upTriFactor->csrMat = new CsrMatrix;
        upTriFactor->csrMat->num_rows = n;
        upTriFactor->csrMat->num_cols = n;
        upTriFactor->csrMat->num_entries = nzUpper;

        upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+n+1);

        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzUpper);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+nzUpper);

        upTriFactor->csrMat->values = new THRUSTARRAY(nzUpper);
        upTriFactor->csrMat->values->assign(AAUp, AAUp+nzUpper);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = cusparse_create_analysis_info(&upTriFactor->solveInfo);CHKERRCUSPARSE(stat);
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactor->solveOp,
                                     upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                     upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                     upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                     &upTriFactor->solveBufferSize);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize);CHKERRCUDA(cerr);
      #endif

        /* perform the solve analysis */
        stat = cusparse_analysis(cusparseTriFactors->handle, upTriFactor->solveOp,
                                 upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                 upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                 upTriFactor->csrMat->column_indices->data().get(),
                               #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 upTriFactor->solveInfo,
                                 upTriFactor->solvePolicy, upTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                               #else
                                 upTriFactor->solveInfo);CHKERRCUSPARSE(stat);
                               #endif
        cerr = WaitForCUDA();CHKERRCUDA(cerr);
        ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;
        upTriFactor->AA_h = AAUp;
        cerr = cudaFreeHost(AiUp);CHKERRCUDA(cerr);
        cerr = cudaFreeHost(AjUp);CHKERRCUDA(cerr);
        ierr = PetscLogCpuToGpu((n+1+nzUpper)*sizeof(int)+nzUpper*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        if (!upTriFactor->AA_h) {
          cerr = cudaMallocHost((void**) &upTriFactor->AA_h, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);
        }
        /* Fill the upper triangular matrix */
        offset = nzUpper;
        for (i=n-1; i>=0; i--) {
          v  = aa + adiag[i+1] + 1;

          /* number of elements NOT on the diagonal */
          nz = adiag[i] - adiag[i+1]-1;

          /* decrement the offset */
          offset -= (nz+1);

          /* first, set the diagonal elements */
          upTriFactor->AA_h[offset] = 1./v[nz];
          ierr = PetscArraycpy(&(upTriFactor->AA_h[offset+1]), v, nz);CHKERRQ(ierr);
        }
        upTriFactor->csrMat->values->assign(upTriFactor->AA_h, upTriFactor->AA_h+nzUpper);
        ierr = PetscLogCpuToGpu(nzUpper*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           isrow = a->row,iscol = a->icol;
  PetscBool                    row_identity,col_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  if (!cusparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
  ierr = MatSeqAIJCUSPARSEBuildILULowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSEBuildILUUpperTriMatrix(A);CHKERRQ(ierr);

  if (!cusparseTriFactors->workVector) { cusparseTriFactors->workVector = new THRUSTARRAY(n); }
  cusparseTriFactors->nnz=a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH;
  /* lower triangular indices */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity && !cusparseTriFactors->rpermIndices) {
    const PetscInt *r;

    ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
    cusparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->rpermIndices->assign(r, r+n);
    ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }

  /* upper triangular indices */
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity && !cusparseTriFactors->cpermIndices) {
    const PetscInt *c;

    ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
    cusparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->cpermIndices->assign(c, c+n);
    ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t                  stat;
  PetscErrorCode                    ierr;
  cudaError_t                       cerr;
  PetscInt                          *AiUp, *AjUp;
  PetscScalar                       *AAUp;
  PetscScalar                       *AALo;
  PetscInt                          nzUpper = a->nz,n = A->rmap->n,i,offset,nz,j;
  Mat_SeqSBAIJ                      *b = (Mat_SeqSBAIJ*)A->data;
  const PetscInt                    *ai = b->i,*aj = b->j,*vj;
  const MatScalar                   *aa = b->a,*v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      cerr = cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);
      cerr = cudaMallocHost((void**) &AALo, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);
      if (!upTriFactor && !loTriFactor) {
        /* Allocate Space for the upper triangular matrix */
        cerr = cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUDA(cerr);

        /* Fill the upper triangular matrix */
        AiUp[0]=(PetscInt) 0;
        AiUp[n]=nzUpper;
        offset = 0;
        for (i=0; i<n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          vj = aj + ai[i];
          nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt) i;
          AAUp[offset] = (MatScalar)1.0/v[nz];
          AiUp[i]      = offset;
          AALo[offset] = (MatScalar)1.0/v[nz];

          offset+=1;
          if (nz>0) {
            ierr = PetscArraycpy(&(AjUp[offset]), vj, nz);CHKERRQ(ierr);
            ierr = PetscArraycpy(&(AAUp[offset]), v, nz);CHKERRQ(ierr);
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&upTriFactor);CHKERRQ(ierr);
        upTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = cusparseCreateMatDescr(&upTriFactor->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(upTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
       #else
        stat = cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSPARSE(stat);
       #endif
        stat = cusparseSetMatFillMode(upTriFactor->descr, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatDiagType(upTriFactor->descr, CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSPARSE(stat);

        /* set the matrix */
        upTriFactor->csrMat = new CsrMatrix;
        upTriFactor->csrMat->num_rows = A->rmap->n;
        upTriFactor->csrMat->num_cols = A->cmap->n;
        upTriFactor->csrMat->num_entries = a->nz;

        upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

        upTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
        upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);

        /* set the operation */
        upTriFactor->solveOp = CUSPARSE_OPERATION_NON_TRANSPOSE;

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = cusparse_create_analysis_info(&upTriFactor->solveInfo);CHKERRCUSPARSE(stat);
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactor->solveOp,
                                       upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                       upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                       upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                       &upTriFactor->solveBufferSize);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize);CHKERRCUDA(cerr);
      #endif

        /* perform the solve analysis */
        stat = cusparse_analysis(cusparseTriFactors->handle, upTriFactor->solveOp,
                                 upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                 upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                 upTriFactor->csrMat->column_indices->data().get(),
                                #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 upTriFactor->solveInfo,
                                 upTriFactor->solvePolicy, upTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                                #else
                                  upTriFactor->solveInfo);CHKERRCUSPARSE(stat);
                                #endif
        cerr = WaitForCUDA();CHKERRCUDA(cerr);
        ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&loTriFactor);CHKERRQ(ierr);
        loTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = cusparseCreateMatDescr(&loTriFactor->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(loTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
       #else
        stat = cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSPARSE(stat);
       #endif
        stat = cusparseSetMatFillMode(loTriFactor->descr, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatDiagType(loTriFactor->descr, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSPARSE(stat);

        /* set the operation */
        loTriFactor->solveOp = CUSPARSE_OPERATION_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat = new CsrMatrix;
        loTriFactor->csrMat->num_rows = A->rmap->n;
        loTriFactor->csrMat->num_cols = A->cmap->n;
        loTriFactor->csrMat->num_entries = a->nz;

        loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
        loTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        loTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

        loTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = cusparse_create_analysis_info(&loTriFactor->solveInfo);CHKERRCUSPARSE(stat);
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactor->solveOp,
                                       loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                       loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                       &loTriFactor->solveBufferSize);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize);CHKERRCUDA(cerr);
      #endif

        /* perform the solve analysis */
        stat = cusparse_analysis(cusparseTriFactors->handle, loTriFactor->solveOp,
                                 loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                 loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                 loTriFactor->csrMat->column_indices->data().get(),
                                #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 loTriFactor->solveInfo,
                                 loTriFactor->solvePolicy, loTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                                #else
                                 loTriFactor->solveInfo);CHKERRCUSPARSE(stat);
                                #endif
        cerr = WaitForCUDA();CHKERRCUDA(cerr);
        ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;

        ierr = PetscLogCpuToGpu(2*(((A->rmap->n+1)+(a->nz))*sizeof(int)+(a->nz)*sizeof(PetscScalar)));CHKERRQ(ierr);
        cerr = cudaFreeHost(AiUp);CHKERRCUDA(cerr);
        cerr = cudaFreeHost(AjUp);CHKERRCUDA(cerr);
      } else {
        /* Fill the upper triangular matrix */
        offset = 0;
        for (i=0; i<n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AAUp[offset] = 1.0/v[nz];
          AALo[offset] = 1.0/v[nz];

          offset+=1;
          if (nz>0) {
            ierr = PetscArraycpy(&(AAUp[offset]), v, nz);CHKERRQ(ierr);
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }
        if (!upTriFactor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
        if (!loTriFactor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
        upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);
        ierr = PetscLogCpuToGpu(2*(a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      cerr = cudaFreeHost(AAUp);CHKERRCUDA(cerr);
      cerr = cudaFreeHost(AALo);CHKERRCUDA(cerr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           ip = a->row;
  PetscBool                    perm_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  if (!cusparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
  ierr = MatSeqAIJCUSPARSEBuildICCTriMatrices(A);CHKERRQ(ierr);
  if (!cusparseTriFactors->workVector) { cusparseTriFactors->workVector = new THRUSTARRAY(n); }
  cusparseTriFactors->nnz=(a->nz-n)*2 + n;

  A->offloadmask = PETSC_OFFLOAD_BOTH;

  /* lower triangular indices */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) {
    IS             iip;
    const PetscInt *irip,*rip;

    ierr = ISInvertPermutation(ip,PETSC_DECIDE,&iip);CHKERRQ(ierr);
    ierr = ISGetIndices(iip,&irip);CHKERRQ(ierr);
    ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
    cusparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->rpermIndices->assign(rip, rip+n);
    cusparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->cpermIndices->assign(irip, irip+n);
    ierr = ISRestoreIndices(iip,&irip);CHKERRQ(ierr);
    ierr = ISDestroy(&iip);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSECopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity) {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactorT;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactorT;
  cusparseStatus_t                  stat;
  cusparseIndexBase_t               indexBase;
  cusparseMatrixType_t              matrixType;
  cusparseFillMode_t                fillMode;
  cusparseDiagType_t                diagType;
  cudaError_t                       cerr;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  /* allocate space for the transpose of the lower triangular factor */
  ierr = PetscNew(&loTriFactorT);CHKERRQ(ierr);
  loTriFactorT->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the lower triangular factor */
  matrixType = cusparseGetMatType(loTriFactor->descr);
  indexBase = cusparseGetMatIndexBase(loTriFactor->descr);
  fillMode = cusparseGetMatFillMode(loTriFactor->descr)==CUSPARSE_FILL_MODE_UPPER ?
    CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
  diagType = cusparseGetMatDiagType(loTriFactor->descr);

  /* Create the matrix description */
  stat = cusparseCreateMatDescr(&loTriFactorT->descr);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatIndexBase(loTriFactorT->descr, indexBase);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatType(loTriFactorT->descr, matrixType);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatFillMode(loTriFactorT->descr, fillMode);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatDiagType(loTriFactorT->descr, diagType);CHKERRCUSPARSE(stat);

  /* set the operation */
  loTriFactorT->solveOp = CUSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the lower triangular factor*/
  loTriFactorT->csrMat = new CsrMatrix;
  loTriFactorT->csrMat->num_rows       = loTriFactor->csrMat->num_cols;
  loTriFactorT->csrMat->num_cols       = loTriFactor->csrMat->num_rows;
  loTriFactorT->csrMat->num_entries    = loTriFactor->csrMat->num_entries;
  loTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_rows+1);
  loTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_entries);
  loTriFactorT->csrMat->values         = new THRUSTARRAY(loTriFactorT->csrMat->num_entries);

  /* compute the transpose of the lower triangular factor, i.e. the CSC */
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  stat = cusparseCsr2cscEx2_bufferSize(cusparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                                       loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                                       loTriFactor->csrMat->values->data().get(),
                                       loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(),
                                       loTriFactorT->csrMat->values->data().get(),
                                       loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                       CUSPARSE_ACTION_NUMERIC,indexBase,
                                       CUSPARSE_CSR2CSC_ALG1, &loTriFactor->csr2cscBufferSize);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc(&loTriFactor->csr2cscBuffer,loTriFactor->csr2cscBufferSize);CHKERRCUDA(cerr);
#endif

  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  stat = cusparse_csr2csc(cusparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                          loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                          loTriFactor->csrMat->values->data().get(),
                          loTriFactor->csrMat->row_offsets->data().get(),
                          loTriFactor->csrMat->column_indices->data().get(),
                          loTriFactorT->csrMat->values->data().get(),
                        #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                          loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                          CUSPARSE_ACTION_NUMERIC, indexBase,
                          CUSPARSE_CSR2CSC_ALG1, loTriFactor->csr2cscBuffer);CHKERRCUSPARSE(stat);
                        #else
                          loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                          CUSPARSE_ACTION_NUMERIC, indexBase);CHKERRCUSPARSE(stat);
                        #endif
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);

  /* Create the solve analysis information */
  ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
  stat = cusparse_create_analysis_info(&loTriFactorT->solveInfo);CHKERRCUSPARSE(stat);
#if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactorT->solveOp,
                                loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                                loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo,
                                &loTriFactorT->solveBufferSize);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc(&loTriFactorT->solveBuffer,loTriFactorT->solveBufferSize);CHKERRCUDA(cerr);
#endif

  /* perform the solve analysis */
  stat = cusparse_analysis(cusparseTriFactors->handle, loTriFactorT->solveOp,
                           loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                           loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                           loTriFactorT->csrMat->column_indices->data().get(),
                          #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                           loTriFactorT->solveInfo,
                           loTriFactorT->solvePolicy, loTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                          #else
                           loTriFactorT->solveInfo);CHKERRCUSPARSE(stat);
                          #endif
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtrTranspose = loTriFactorT;

  /*********************************************/
  /* Now the Transpose of the Upper Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the upper triangular factor */
  ierr = PetscNew(&upTriFactorT);CHKERRQ(ierr);
  upTriFactorT->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the upper triangular factor */
  matrixType = cusparseGetMatType(upTriFactor->descr);
  indexBase = cusparseGetMatIndexBase(upTriFactor->descr);
  fillMode = cusparseGetMatFillMode(upTriFactor->descr)==CUSPARSE_FILL_MODE_UPPER ?
    CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
  diagType = cusparseGetMatDiagType(upTriFactor->descr);

  /* Create the matrix description */
  stat = cusparseCreateMatDescr(&upTriFactorT->descr);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatIndexBase(upTriFactorT->descr, indexBase);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatType(upTriFactorT->descr, matrixType);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatFillMode(upTriFactorT->descr, fillMode);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatDiagType(upTriFactorT->descr, diagType);CHKERRCUSPARSE(stat);

  /* set the operation */
  upTriFactorT->solveOp = CUSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the upper triangular factor*/
  upTriFactorT->csrMat = new CsrMatrix;
  upTriFactorT->csrMat->num_rows       = upTriFactor->csrMat->num_cols;
  upTriFactorT->csrMat->num_cols       = upTriFactor->csrMat->num_rows;
  upTriFactorT->csrMat->num_entries    = upTriFactor->csrMat->num_entries;
  upTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_rows+1);
  upTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_entries);
  upTriFactorT->csrMat->values         = new THRUSTARRAY(upTriFactorT->csrMat->num_entries);

  /* compute the transpose of the upper triangular factor, i.e. the CSC */
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  stat = cusparseCsr2cscEx2_bufferSize(cusparseTriFactors->handle,upTriFactor->csrMat->num_rows,
                                upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                                upTriFactor->csrMat->values->data().get(),
                                upTriFactor->csrMat->row_offsets->data().get(),
                                upTriFactor->csrMat->column_indices->data().get(),
                                upTriFactorT->csrMat->values->data().get(),
                                upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                CUSPARSE_ACTION_NUMERIC,indexBase,
                                CUSPARSE_CSR2CSC_ALG1, &upTriFactor->csr2cscBufferSize);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc(&upTriFactor->csr2cscBuffer,upTriFactor->csr2cscBufferSize);CHKERRCUDA(cerr);
#endif

  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  stat = cusparse_csr2csc(cusparseTriFactors->handle, upTriFactor->csrMat->num_rows,
                          upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                          upTriFactor->csrMat->values->data().get(),
                          upTriFactor->csrMat->row_offsets->data().get(),
                          upTriFactor->csrMat->column_indices->data().get(),
                          upTriFactorT->csrMat->values->data().get(),
                        #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                          upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                          CUSPARSE_ACTION_NUMERIC, indexBase,
                          CUSPARSE_CSR2CSC_ALG1, upTriFactor->csr2cscBuffer);CHKERRCUSPARSE(stat);
                        #else
                          upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                          CUSPARSE_ACTION_NUMERIC, indexBase);CHKERRCUSPARSE(stat);
                        #endif

  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);

  /* Create the solve analysis information */
  ierr = PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
  stat = cusparse_create_analysis_info(&upTriFactorT->solveInfo);CHKERRCUSPARSE(stat);
  #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  stat = cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactorT->solveOp,
                                 upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                                 upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                 upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo,
                                 &upTriFactorT->solveBufferSize);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc(&upTriFactorT->solveBuffer,upTriFactorT->solveBufferSize);CHKERRCUDA(cerr);
  #endif

  /* perform the solve analysis */
  stat = cusparse_analysis(cusparseTriFactors->handle, upTriFactorT->solveOp,
                           upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                           upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                           upTriFactorT->csrMat->column_indices->data().get(),
                          #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                           upTriFactorT->solveInfo,
                           upTriFactorT->solvePolicy, upTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                          #else
                           upTriFactorT->solveInfo);CHKERRCUSPARSE(stat);
                          #endif

  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtrTranspose = upTriFactorT;
  PetscFunctionReturn(0);
}

struct PetscScalarToPetscInt
{
  __host__ __device__
  PetscInt operator()(PetscScalar s)
  {
    return (PetscInt)PetscRealPart(s);
  }
};

static PetscErrorCode MatSeqAIJCUSPARSEFormExplicitTranspose(Mat A)
{
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct, *matstructT;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  cusparseStatus_t             stat;
  cusparseIndexBase_t          indexBase;
  cudaError_t                  err;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
  if (!matstruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing mat struct");
  matstructT = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->matTranspose;
  if (A->transupdated && !matstructT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing matTranspose struct");
  if (A->transupdated) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (cusparsestruct->format != MAT_CUSPARSE_CSR) {
    ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!cusparsestruct->matTranspose) { /* create cusparse matrix */
    matstructT = new Mat_SeqAIJCUSPARSEMultStruct;
    stat = cusparseCreateMatDescr(&matstructT->descr);CHKERRCUSPARSE(stat);
    indexBase = cusparseGetMatIndexBase(matstruct->descr);
    stat = cusparseSetMatIndexBase(matstructT->descr, indexBase);CHKERRCUSPARSE(stat);
    stat = cusparseSetMatType(matstructT->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);

    /* set alpha and beta */
    err = cudaMalloc((void **)&(matstructT->alpha_one),sizeof(PetscScalar));CHKERRCUDA(err);
    err = cudaMalloc((void **)&(matstructT->beta_zero),sizeof(PetscScalar));CHKERRCUDA(err);
    err = cudaMalloc((void **)&(matstructT->beta_one), sizeof(PetscScalar));CHKERRCUDA(err);
    err = cudaMemcpy(matstructT->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMemcpy(matstructT->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMemcpy(matstructT->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);

    if (cusparsestruct->format == MAT_CUSPARSE_CSR) {
      CsrMatrix *matrixT = new CsrMatrix;
      matstructT->mat = matrixT;
      matrixT->num_rows = A->cmap->n;
      matrixT->num_cols = A->rmap->n;
      matrixT->num_entries = a->nz;
      matrixT->row_offsets = new THRUSTINTARRAY32(matrixT->num_rows+1);
      matrixT->column_indices = new THRUSTINTARRAY32(a->nz);
      matrixT->values = new THRUSTARRAY(a->nz);

      if (!cusparsestruct->rowoffsets_gpu) { cusparsestruct->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1); }
      cusparsestruct->rowoffsets_gpu->assign(a->i,a->i+A->rmap->n+1);

     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      #if PETSC_PKG_CUDA_VERSION_GE(11,2,1)
        stat = cusparseCreateCsr(&matstructT->matDescr,
                               matrixT->num_rows, matrixT->num_cols, matrixT->num_entries,
                               matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(),
                               matrixT->values->data().get(),
                               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, /* row offset, col idx type due to THRUSTINTARRAY32 */
                               indexBase,cusparse_scalartype);CHKERRCUSPARSE(stat);
      #else
        /* cusparse-11.x returns errors with zero-sized matrices until 11.2.1,
           see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cusparse-11.2.1

           I don't know what a proper value should be for matstructT->matDescr with empty matrices, so I just set
           it to NULL to blow it up if one relies on it. Per https://docs.nvidia.com/cuda/cusparse/index.html#csr2cscEx2,
           when nnz = 0, matrixT->row_offsets[] should be filled with indexBase. So I also set it accordingly.
        */
        if (matrixT->num_entries) {
          stat = cusparseCreateCsr(&matstructT->matDescr,
                                 matrixT->num_rows, matrixT->num_cols, matrixT->num_entries,
                                 matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(),
                                 matrixT->values->data().get(),
                                 CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,
                                 indexBase,cusparse_scalartype);CHKERRCUSPARSE(stat);

        } else {
          matstructT->matDescr = NULL;
          matrixT->row_offsets->assign(matrixT->row_offsets->size(),indexBase);
        }
      #endif
     #endif
    } else if (cusparsestruct->format == MAT_CUSPARSE_ELL || cusparsestruct->format == MAT_CUSPARSE_HYB) {
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
   #else
      CsrMatrix *temp  = new CsrMatrix;
      CsrMatrix *tempT = new CsrMatrix;
      /* First convert HYB to CSR */
      temp->num_rows = A->rmap->n;
      temp->num_cols = A->cmap->n;
      temp->num_entries = a->nz;
      temp->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      temp->column_indices = new THRUSTINTARRAY32(a->nz);
      temp->values = new THRUSTARRAY(a->nz);

      stat = cusparse_hyb2csr(cusparsestruct->handle,
                              matstruct->descr, (cusparseHybMat_t)matstruct->mat,
                              temp->values->data().get(),
                              temp->row_offsets->data().get(),
                              temp->column_indices->data().get());CHKERRCUSPARSE(stat);

      /* Next, convert CSR to CSC (i.e. the matrix transpose) */
      tempT->num_rows = A->rmap->n;
      tempT->num_cols = A->cmap->n;
      tempT->num_entries = a->nz;
      tempT->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      tempT->column_indices = new THRUSTINTARRAY32(a->nz);
      tempT->values = new THRUSTARRAY(a->nz);

      stat = cusparse_csr2csc(cusparsestruct->handle, temp->num_rows,
                              temp->num_cols, temp->num_entries,
                              temp->values->data().get(),
                              temp->row_offsets->data().get(),
                              temp->column_indices->data().get(),
                              tempT->values->data().get(),
                              tempT->column_indices->data().get(),
                              tempT->row_offsets->data().get(),
                              CUSPARSE_ACTION_NUMERIC, indexBase);CHKERRCUSPARSE(stat);

      /* Last, convert CSC to HYB */
      cusparseHybMat_t hybMat;
      stat = cusparseCreateHybMat(&hybMat);CHKERRCUSPARSE(stat);
      cusparseHybPartition_t partition = cusparsestruct->format==MAT_CUSPARSE_ELL ?
        CUSPARSE_HYB_PARTITION_MAX : CUSPARSE_HYB_PARTITION_AUTO;
      stat = cusparse_csr2hyb(cusparsestruct->handle, A->rmap->n, A->cmap->n,
                              matstructT->descr, tempT->values->data().get(),
                              tempT->row_offsets->data().get(),
                              tempT->column_indices->data().get(),
                              hybMat, 0, partition);CHKERRCUSPARSE(stat);

      /* assign the pointer */
      matstructT->mat = hybMat;
      A->transupdated = PETSC_TRUE;
      /* delete temporaries */
      if (tempT) {
        if (tempT->values) delete (THRUSTARRAY*) tempT->values;
        if (tempT->column_indices) delete (THRUSTINTARRAY32*) tempT->column_indices;
        if (tempT->row_offsets) delete (THRUSTINTARRAY32*) tempT->row_offsets;
        delete (CsrMatrix*) tempT;
      }
      if (temp) {
        if (temp->values) delete (THRUSTARRAY*) temp->values;
        if (temp->column_indices) delete (THRUSTINTARRAY32*) temp->column_indices;
        if (temp->row_offsets) delete (THRUSTINTARRAY32*) temp->row_offsets;
        delete (CsrMatrix*) temp;
      }
     #endif
    }
  }
  if (cusparsestruct->format == MAT_CUSPARSE_CSR) { /* transpose mat struct may be already present, update data */
    CsrMatrix *matrix  = (CsrMatrix*)matstruct->mat;
    CsrMatrix *matrixT = (CsrMatrix*)matstructT->mat;
    if (!matrix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix");
    if (!matrix->row_offsets) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix rows");
    if (!matrix->column_indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix cols");
    if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix values");
    if (!matrixT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT");
    if (!matrixT->row_offsets) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT rows");
    if (!matrixT->column_indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT cols");
    if (!matrixT->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT values");
    if (!cusparsestruct->rowoffsets_gpu) { /* this may be absent when we did not construct the transpose with csr2csc */
      cusparsestruct->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
      cusparsestruct->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
      ierr = PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!cusparsestruct->csr2csc_i) {
      THRUSTARRAY csr2csc_a(matrix->num_entries);
      PetscStackCallThrust(thrust::sequence(thrust::device, csr2csc_a.begin(), csr2csc_a.end(), 0.0));

      indexBase = cusparseGetMatIndexBase(matstruct->descr);
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      void   *csr2cscBuffer;
      size_t csr2cscBufferSize;
      stat = cusparseCsr2cscEx2_bufferSize(cusparsestruct->handle, A->rmap->n,
                                           A->cmap->n, matrix->num_entries,
                                           matrix->values->data().get(),
                                           cusparsestruct->rowoffsets_gpu->data().get(),
                                           matrix->column_indices->data().get(),
                                           matrixT->values->data().get(),
                                           matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), cusparse_scalartype,
                                           CUSPARSE_ACTION_NUMERIC,indexBase,
                                           cusparsestruct->csr2cscAlg, &csr2cscBufferSize);CHKERRCUSPARSE(stat);
      err = cudaMalloc(&csr2cscBuffer,csr2cscBufferSize);CHKERRCUDA(err);
     #endif

      if (matrix->num_entries) {
        /* When there are no nonzeros, this routine mistakenly returns CUSPARSE_STATUS_INVALID_VALUE in
           mat_tests-ex62_15_mpiaijcusparse on ranks 0 and 2 with CUDA-11. But CUDA-10 is OK.
           I checked every parameters and they were just fine. I have no clue why cusparse complains.

           Per https://docs.nvidia.com/cuda/cusparse/index.html#csr2cscEx2, when nnz = 0, matrixT->row_offsets[]
           should be filled with indexBase. So I just take a shortcut here.
        */
        stat = cusparse_csr2csc(cusparsestruct->handle, A->rmap->n,
                              A->cmap->n,matrix->num_entries,
                              csr2csc_a.data().get(),
                              cusparsestruct->rowoffsets_gpu->data().get(),
                              matrix->column_indices->data().get(),
                              matrixT->values->data().get(),
                             #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                              matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), cusparse_scalartype,
                              CUSPARSE_ACTION_NUMERIC,indexBase,
                              cusparsestruct->csr2cscAlg, csr2cscBuffer);CHKERRCUSPARSE(stat);
                             #else
                              matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(),
                              CUSPARSE_ACTION_NUMERIC, indexBase);CHKERRCUSPARSE(stat);
                             #endif
      } else {
        matrixT->row_offsets->assign(matrixT->row_offsets->size(),indexBase);
      }

      cusparsestruct->csr2csc_i = new THRUSTINTARRAY(matrix->num_entries);
      PetscStackCallThrust(thrust::transform(thrust::device,matrixT->values->begin(),matrixT->values->end(),cusparsestruct->csr2csc_i->begin(),PetscScalarToPetscInt()));
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      err = cudaFree(csr2cscBuffer);CHKERRCUDA(err);
     #endif
    }
    PetscStackCallThrust(thrust::copy(thrust::device,thrust::make_permutation_iterator(matrix->values->begin(), cusparsestruct->csr2csc_i->begin()),
                                                     thrust::make_permutation_iterator(matrix->values->begin(), cusparsestruct->csr2csc_i->end()),
                                                     matrixT->values->begin()));
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  /* the compressed row indices is not used for matTranspose */
  matstructT->cprowIndices = NULL;
  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSE*)A->spptr)->matTranspose = matstructT;
  A->transupdated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Why do we need to analyze the transposed matrix again? Can't we just use op(A) = CUSPARSE_OPERATION_TRANSPOSE in MatSolve_SeqAIJCUSPARSE? */
static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE(Mat A,Vec bb,Vec xx)
{
  PetscInt                              n = xx->map->n;
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  cusparseStatus_t                      stat;
  Mat_SeqAIJCUSPARSETriFactors          *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct     *loTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJCUSPARSETriFactorStruct     *upTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)cusparseTriFactors->workVector;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU+n, cusparseTriFactors->rpermIndices->end()),
               xGPU);

  /* First, solve U */
  stat = cusparse_solve(cusparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        upTriFactorT->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        xarray,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        tempGPU->data().get(),
                        upTriFactorT->solvePolicy, upTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        tempGPU->data().get());CHKERRCUSPARSE(stat);
                      #endif

  /* Then, solve L */
  stat = cusparse_solve(cusparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        loTriFactorT->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(),
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        xarray,
                        loTriFactorT->solvePolicy, loTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                         xarray);CHKERRCUSPARSE(stat);
                      #endif

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(xGPU, cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(xGPU+n, cusparseTriFactors->cpermIndices->end()),
               tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  cusparseStatus_t                  stat;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)cusparseTriFactors->workVector;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT       = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT       = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve U */
  stat = cusparse_solve(cusparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        upTriFactorT->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        barray,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        tempGPU->data().get(),
                        upTriFactorT->solvePolicy, upTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        tempGPU->data().get());CHKERRCUSPARSE(stat);
                      #endif

  /* Then, solve L */
  stat = cusparse_solve(cusparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        loTriFactorT->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(),
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        xarray,
                        loTriFactorT->solvePolicy, loTriFactorT->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        xarray);CHKERRCUSPARSE(stat);
                      #endif

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJCUSPARSE(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  cusparseStatus_t                      stat;
  Mat_SeqAIJCUSPARSETriFactors          *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct     *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct     *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)cusparseTriFactors->workVector;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->end()),
               tempGPU->begin());

  /* Next, solve L */
  stat = cusparse_solve(cusparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        loTriFactor->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        tempGPU->data().get(),
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                         xarray,
                         loTriFactor->solvePolicy, loTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                         xarray);CHKERRCUSPARSE(stat);
                      #endif

  /* Then, solve U */
  stat = cusparse_solve(cusparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        upTriFactor->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,xarray,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        tempGPU->data().get(),
                        upTriFactor->solvePolicy, upTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        tempGPU->data().get());CHKERRCUSPARSE(stat);
                      #endif

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->end()),
               xGPU);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJCUSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  cusparseStatus_t                  stat;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)cusparseTriFactors->workVector;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve L */
  stat = cusparse_solve(cusparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        loTriFactor->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        barray,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        tempGPU->data().get(),
                        loTriFactor->solvePolicy,loTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        tempGPU->data().get());CHKERRCUSPARSE(stat);
                      #endif

  /* Next, solve U */
  stat = cusparse_solve(cusparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows,
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        upTriFactor->csrMat->num_entries,
                      #endif
                        &PETSC_CUSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,
                        tempGPU->data().get(),
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        xarray,
                        upTriFactor->solvePolicy, upTriFactor->solveBuffer);CHKERRCUSPARSE(stat);
                      #else
                        xarray);CHKERRCUSPARSE(stat);
                      #endif

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSECopyFromGPU(Mat A)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  cudaError_t        cerr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    CsrMatrix *matrix = (CsrMatrix*)cusp->mat->mat;

    ierr = PetscLogEventBegin(MAT_CUSPARSECopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    cerr = cudaMemcpy(a->a, matrix->values->data().get(), a->nz*sizeof(PetscScalar), cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
    ierr = PetscLogGpuToCpu(a->nz*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_CUSPARSECopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJCUSPARSE(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSECopyFromGPU(A);CHKERRQ(ierr);
  *array = a->a;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct = cusparsestruct->mat;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  PetscInt                     m = A->rmap->n,*ii,*ridx,tmp;
  PetscErrorCode               ierr;
  cusparseStatus_t             stat;
  PetscBool                    both = PETSC_TRUE;
  cudaError_t                  err;

  PetscFunctionBegin;
  if (A->boundtocpu) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Cannot copy to GPU");
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (A->nonzerostate == cusparsestruct->nonzerostate && cusparsestruct->format == MAT_CUSPARSE_CSR) { /* Copy values only */
      CsrMatrix *matrix;
      matrix = (CsrMatrix*)cusparsestruct->mat->mat;

      if (a->nz && !a->a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR values");
      ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      matrix->values->assign(a->a, a->a+a->nz);
      err  = WaitForCUDA();CHKERRCUDA(err);
      ierr = PetscLogCpuToGpu((a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      PetscInt nnz;
      ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusparsestruct->mat,cusparsestruct->format);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);
      delete cusparsestruct->workVector;
      delete cusparsestruct->rowoffsets_gpu;
      cusparsestruct->workVector = NULL;
      cusparsestruct->rowoffsets_gpu = NULL;
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
        if (!ii) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR row data");
        if (m && !a->j) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR column data");
        if (!a->a) { nnz = ii[m]; both = PETSC_FALSE; }
        else nnz = a->nz;

        /* create cusparse matrix */
        cusparsestruct->nrows = m;
        matstruct = new Mat_SeqAIJCUSPARSEMultStruct;
        stat = cusparseCreateMatDescr(&matstruct->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(matstruct->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatType(matstruct->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);

        err = cudaMalloc((void **)&(matstruct->alpha_one),sizeof(PetscScalar));CHKERRCUDA(err);
        err = cudaMalloc((void **)&(matstruct->beta_zero),sizeof(PetscScalar));CHKERRCUDA(err);
        err = cudaMalloc((void **)&(matstruct->beta_one), sizeof(PetscScalar));CHKERRCUDA(err);
        err = cudaMemcpy(matstruct->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
        err = cudaMemcpy(matstruct->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
        err = cudaMemcpy(matstruct->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
        stat = cusparseSetPointerMode(cusparsestruct->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);

        /* Build a hybrid/ellpack matrix if this option is chosen for the storage */
        if (cusparsestruct->format==MAT_CUSPARSE_CSR) {
          /* set the matrix */
          CsrMatrix *mat= new CsrMatrix;
          mat->num_rows = m;
          mat->num_cols = A->cmap->n;
          mat->num_entries = nnz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->column_indices->assign(a->j, a->j+nnz);

          mat->values = new THRUSTARRAY(nnz);
          if (a->a) mat->values->assign(a->a, a->a+nnz);

          /* assign the pointer */
          matstruct->mat = mat;
         #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
          if (mat->num_rows) { /* cusparse errors on empty matrices! */
            stat = cusparseCreateCsr(&matstruct->matDescr,
                                    mat->num_rows, mat->num_cols, mat->num_entries,
                                    mat->row_offsets->data().get(), mat->column_indices->data().get(),
                                    mat->values->data().get(),
                                    CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, /* row offset, col idx types due to THRUSTINTARRAY32 */
                                    CUSPARSE_INDEX_BASE_ZERO,cusparse_scalartype);CHKERRCUSPARSE(stat);
          }
         #endif
        } else if (cusparsestruct->format==MAT_CUSPARSE_ELL || cusparsestruct->format==MAT_CUSPARSE_HYB) {
         #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
         #else
          CsrMatrix *mat= new CsrMatrix;
          mat->num_rows = m;
          mat->num_cols = A->cmap->n;
          mat->num_entries = nnz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->column_indices->assign(a->j, a->j+nnz);

          mat->values = new THRUSTARRAY(nnz);
          if (a->a) mat->values->assign(a->a, a->a+nnz);

          cusparseHybMat_t hybMat;
          stat = cusparseCreateHybMat(&hybMat);CHKERRCUSPARSE(stat);
          cusparseHybPartition_t partition = cusparsestruct->format==MAT_CUSPARSE_ELL ?
            CUSPARSE_HYB_PARTITION_MAX : CUSPARSE_HYB_PARTITION_AUTO;
          stat = cusparse_csr2hyb(cusparsestruct->handle, mat->num_rows, mat->num_cols,
              matstruct->descr, mat->values->data().get(),
              mat->row_offsets->data().get(),
              mat->column_indices->data().get(),
              hybMat, 0, partition);CHKERRCUSPARSE(stat);
          /* assign the pointer */
          matstruct->mat = hybMat;

          if (mat) {
            if (mat->values) delete (THRUSTARRAY*)mat->values;
            if (mat->column_indices) delete (THRUSTINTARRAY32*)mat->column_indices;
            if (mat->row_offsets) delete (THRUSTINTARRAY32*)mat->row_offsets;
            delete (CsrMatrix*)mat;
          }
         #endif
        }

        /* assign the compressed row indices */
        if (a->compressedrow.use) {
          cusparsestruct->workVector = new THRUSTARRAY(m);
          matstruct->cprowIndices    = new THRUSTINTARRAY(m);
          matstruct->cprowIndices->assign(ridx,ridx+m);
          tmp = m;
        } else {
          cusparsestruct->workVector = NULL;
          matstruct->cprowIndices    = NULL;
          tmp = 0;
        }
        ierr = PetscLogCpuToGpu(((m+1)+(a->nz))*sizeof(int)+tmp*sizeof(PetscInt)+(3+(a->nz))*sizeof(PetscScalar));CHKERRQ(ierr);

        /* assign the pointer */
        cusparsestruct->mat = matstruct;
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
      }
      err  = WaitForCUDA();CHKERRCUDA(err);
      ierr = PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      cusparsestruct->nonzerostate = A->nonzerostate;
    }
    if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

struct VecCUDAPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

struct VecCUDAEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

struct VecCUDAEqualsReverse
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t);
  }
};

struct MatMatCusparse {
  PetscBool             cisdense;
  PetscScalar           *Bt;
  Mat                   X;
  PetscBool             reusesym; /* Cusparse does not have split symbolic and numeric phases for sparse matmat operations */
  PetscLogDouble        flops;
  CsrMatrix             *Bcsr;

#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseSpMatDescr_t  matSpBDescr;
  PetscBool             initialized;   /* C = alpha op(A) op(B) + beta C */
  cusparseDnMatDescr_t  matBDescr;
  cusparseDnMatDescr_t  matCDescr;
  PetscInt              Blda,Clda; /* Record leading dimensions of B and C here to detect changes*/
 #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
  void                  *dBuffer4;
  void                  *dBuffer5;
 #endif
  size_t                mmBufferSize;
  void                  *mmBuffer;
  void                  *mmBuffer2; /* SpGEMM WorkEstimation buffer */
  cusparseSpGEMMDescr_t spgemmDesc;
#endif
};

static PetscErrorCode MatDestroy_MatMatCusparse(void *data)
{
  PetscErrorCode   ierr;
  MatMatCusparse   *mmdata = (MatMatCusparse *)data;
  cudaError_t      cerr;
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseStatus_t stat;
 #endif

  PetscFunctionBegin;
  cerr = cudaFree(mmdata->Bt);CHKERRCUDA(cerr);
  delete mmdata->Bcsr;
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  if (mmdata->matSpBDescr) { stat = cusparseDestroySpMat(mmdata->matSpBDescr);CHKERRCUSPARSE(stat); }
  if (mmdata->matBDescr)   { stat = cusparseDestroyDnMat(mmdata->matBDescr);CHKERRCUSPARSE(stat); }
  if (mmdata->matCDescr)   { stat = cusparseDestroyDnMat(mmdata->matCDescr);CHKERRCUSPARSE(stat); }
  if (mmdata->spgemmDesc)  { stat = cusparseSpGEMM_destroyDescr(mmdata->spgemmDesc);CHKERRCUSPARSE(stat); }
 #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
  if (mmdata->dBuffer4)  { cerr = cudaFree(mmdata->dBuffer4);CHKERRCUDA(cerr); }
  if (mmdata->dBuffer5)  { cerr = cudaFree(mmdata->dBuffer5);CHKERRCUDA(cerr); }
 #endif
  if (mmdata->mmBuffer)  { cerr = cudaFree(mmdata->mmBuffer);CHKERRCUDA(cerr); }
  if (mmdata->mmBuffer2) { cerr = cudaFree(mmdata->mmBuffer2);CHKERRCUDA(cerr); }
 #endif
  ierr = MatDestroy(&mmdata->X);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(Mat,Mat,Mat,PetscBool,PetscBool);

static PetscErrorCode MatProductNumeric_SeqAIJCUSPARSE_SeqDENSECUDA(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  PetscInt                     m,n,blda,clda;
  PetscBool                    flg,biscuda;
  Mat_SeqAIJCUSPARSE           *cusp;
  cusparseStatus_t             stat;
  cusparseOperation_t          opA;
  const PetscScalar            *barray;
  PetscScalar                  *carray;
  PetscErrorCode               ierr;
  MatMatCusparse               *mmdata;
  Mat_SeqAIJCUSPARSEMultStruct *mat;
  CsrMatrix                    *csrmat;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data empty");
  mmdata = (MatMatCusparse*)product->data;
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  /* currently CopyToGpu does not copy if the matrix is bound to CPU
     Instead of silently accepting the wrong answer, I prefer to raise the error */
  if (A->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  ierr   = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  cusp   = (Mat_SeqAIJCUSPARSE*)A->spptr;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_PtAP:
    mat = cusp->mat;
    opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    if (!A->form_explicit_transpose) {
      mat = cusp->mat;
      opA = CUSPARSE_OPERATION_TRANSPOSE;
    } else {
      ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(A);CHKERRQ(ierr);
      mat  = cusp->matTranspose;
      opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    }
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
  case MATPRODUCT_RARt:
    mat = cusp->mat;
    opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  if (!mat) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csrmat = (CsrMatrix*)mat->mat;
  /* if the user passed a CPU matrix, copy the data to the GPU */
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSECUDA,&biscuda);CHKERRQ(ierr);
  if (!biscuda) {ierr = MatConvert(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);}
  ierr = MatDenseCUDAGetArrayRead(B,&barray);CHKERRQ(ierr);

  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    ierr = MatDenseCUDAGetArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(mmdata->X,&clda);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDAGetArrayWrite(C,&carray);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  }

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseOperation_t opB = (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  /* (re)allocate mmBuffer if not initialized or LDAs are different */
  if (!mmdata->initialized || mmdata->Blda != blda || mmdata->Clda != clda) {
    size_t mmBufferSize;
    if (mmdata->initialized && mmdata->Blda != blda) {stat = cusparseDestroyDnMat(mmdata->matBDescr);CHKERRCUSPARSE(stat); mmdata->matBDescr = NULL;}
    if (!mmdata->matBDescr) {
      stat         = cusparseCreateDnMat(&mmdata->matBDescr,B->rmap->n,B->cmap->n,blda,(void*)barray,cusparse_scalartype,CUSPARSE_ORDER_COL);CHKERRCUSPARSE(stat);
      mmdata->Blda = blda;
    }

    if (mmdata->initialized && mmdata->Clda != clda) {stat = cusparseDestroyDnMat(mmdata->matCDescr);CHKERRCUSPARSE(stat); mmdata->matCDescr = NULL;}
    if (!mmdata->matCDescr) { /* matCDescr is for C or mmdata->X */
      stat         = cusparseCreateDnMat(&mmdata->matCDescr,m,n,clda,(void*)carray,cusparse_scalartype,CUSPARSE_ORDER_COL);CHKERRCUSPARSE(stat);
      mmdata->Clda = clda;
    }

    if (!mat->matDescr) {
      stat = cusparseCreateCsr(&mat->matDescr,
                               csrmat->num_rows, csrmat->num_cols, csrmat->num_entries,
                               csrmat->row_offsets->data().get(), csrmat->column_indices->data().get(),
                               csrmat->values->data().get(),
                               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, /* row offset, col idx types due to THRUSTINTARRAY32 */
                               CUSPARSE_INDEX_BASE_ZERO,cusparse_scalartype);CHKERRCUSPARSE(stat);
    }
    stat = cusparseSpMM_bufferSize(cusp->handle,opA,opB,mat->alpha_one,
                                   mat->matDescr,mmdata->matBDescr,mat->beta_zero,
                                   mmdata->matCDescr,cusparse_scalartype,
                                   cusp->spmmAlg,&mmBufferSize);CHKERRCUSPARSE(stat);
    if ((mmdata->mmBuffer && mmdata->mmBufferSize < mmBufferSize) || !mmdata->mmBuffer) {
      cudaError_t cerr;
      cerr = cudaFree(mmdata->mmBuffer);CHKERRCUDA(cerr);
      cerr = cudaMalloc(&mmdata->mmBuffer,mmBufferSize);CHKERRCUDA(cerr);
      mmdata->mmBufferSize = mmBufferSize;
    }
    mmdata->initialized = PETSC_TRUE;
  } else {
    /* to be safe, always update pointers of the mats */
    stat = cusparseSpMatSetValues(mat->matDescr,csrmat->values->data().get());CHKERRCUSPARSE(stat);
    stat = cusparseDnMatSetValues(mmdata->matBDescr,(void*)barray);CHKERRCUSPARSE(stat);
    stat = cusparseDnMatSetValues(mmdata->matCDescr,(void*)carray);CHKERRCUSPARSE(stat);
  }

  /* do cusparseSpMM, which supports transpose on B */
  stat = cusparseSpMM(cusp->handle,opA,opB,mat->alpha_one,
                      mat->matDescr,mmdata->matBDescr,mat->beta_zero,
                      mmdata->matCDescr,cusparse_scalartype,
                      cusp->spmmAlg,mmdata->mmBuffer);CHKERRCUSPARSE(stat);
 #else
  PetscInt k;
  /* cusparseXcsrmm does not support transpose on B */
  if (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) {
    cublasHandle_t cublasv2handle;
    cublasStatus_t cerr;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    cerr = cublasXgeam(cublasv2handle,CUBLAS_OP_T,CUBLAS_OP_T,
                       B->cmap->n,B->rmap->n,
                       &PETSC_CUSPARSE_ONE ,barray,blda,
                       &PETSC_CUSPARSE_ZERO,barray,blda,
                       mmdata->Bt,B->cmap->n);CHKERRCUBLAS(cerr);
    blda = B->cmap->n;
    k    = B->cmap->n;
  } else {
    k    = B->rmap->n;
  }

  /* perform the MatMat operation, op(A) is m x k, op(B) is k x n */
  stat = cusparse_csr_spmm(cusp->handle,opA,m,n,k,
                           csrmat->num_entries,mat->alpha_one,mat->descr,
                           csrmat->values->data().get(),
                           csrmat->row_offsets->data().get(),
                           csrmat->column_indices->data().get(),
                           mmdata->Bt ? mmdata->Bt : barray,blda,mat->beta_zero,
                           carray,clda);CHKERRCUSPARSE(stat);
 #endif
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n*2.0*csrmat->num_entries);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(B,&barray);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_RARt) {
    ierr = MatDenseCUDARestoreArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(B,mmdata->X,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  } else if (product->type == MATPRODUCT_PtAP) {
    ierr = MatDenseCUDARestoreArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(B,mmdata->X,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDARestoreArrayWrite(C,&carray);CHKERRQ(ierr);
  }
  if (mmdata->cisdense) {
    ierr = MatConvert(C,MATSEQDENSE,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
  }
  if (!biscuda) {
    ierr = MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJCUSPARSE_SeqDENSECUDA(Mat C)
{
  Mat_Product        *product = C->product;
  Mat                A,B;
  PetscInt           m,n;
  PetscBool          cisdense,flg;
  PetscErrorCode     ierr;
  MatMatCusparse     *mmdata;
  Mat_SeqAIJCUSPARSE *cusp;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data not empty");
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  if (cusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
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
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  /* if C is of type MATSEQDENSE (CPU), perform the operation on the GPU and then copy on the CPU */
  ierr = PetscObjectTypeCompare((PetscObject)C,MATSEQDENSE,&cisdense);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQDENSECUDA);CHKERRQ(ierr);

  /* product data */
  ierr = PetscNew(&mmdata);CHKERRQ(ierr);
  mmdata->cisdense = cisdense;
 #if PETSC_PKG_CUDA_VERSION_LT(11,0,0)
  /* cusparseXcsrmm does not support transpose on B, so we allocate buffer to store B^T */
  if (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) {
    cudaError_t cerr = cudaMalloc((void**)&mmdata->Bt,(size_t)B->rmap->n*(size_t)B->cmap->n*sizeof(PetscScalar));CHKERRCUDA(cerr);
  }
 #endif
  /* for these products we need intermediate storage */
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    ierr = MatCreate(PetscObjectComm((PetscObject)C),&mmdata->X);CHKERRQ(ierr);
    ierr = MatSetType(mmdata->X,MATSEQDENSECUDA);CHKERRQ(ierr);
    if (product->type == MATPRODUCT_RARt) { /* do not preallocate, since the first call to MatDenseCUDAGetArray will preallocate on the GPU for us */
      ierr = MatSetSizes(mmdata->X,A->rmap->n,B->rmap->n,A->rmap->n,B->rmap->n);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(mmdata->X,A->rmap->n,B->cmap->n,A->rmap->n,B->cmap->n);CHKERRQ(ierr);
    }
  }
  C->product->data    = mmdata;
  C->product->destroy = MatDestroy_MatMatCusparse;

  C->ops->productnumeric = MatProductNumeric_SeqAIJCUSPARSE_SeqDENSECUDA;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJCUSPARSE_SeqAIJCUSPARSE(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  Mat_SeqAIJCUSPARSE           *Acusp,*Bcusp,*Ccusp;
  Mat_SeqAIJ                   *c = (Mat_SeqAIJ*)C->data;
  Mat_SeqAIJCUSPARSEMultStruct *Amat,*Bmat,*Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscBool                    flg;
  PetscErrorCode               ierr;
  cusparseStatus_t             stat;
  cudaError_t                  cerr;
  MatProductType               ptype;
  MatMatCusparse               *mmdata;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseSpMatDescr_t         BmatSpDescr;
#endif
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE,opB = CUSPARSE_OPERATION_NON_TRANSPOSE; /* cuSPARSE spgemm doesn't support transpose yet */

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data empty");
  ierr = PetscObjectTypeCompare((PetscObject)C,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for C of type %s",((PetscObject)C)->type_name);
  mmdata = (MatMatCusparse*)C->product->data;
  A = product->A;
  B = product->B;
  if (mmdata->reusesym) { /* this happens when api_user is true, meaning that the matrix values have been already computed in the MatProductSymbolic phase */
    mmdata->reusesym = PETSC_FALSE;
    Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
    if (Ccusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
    Cmat = Ccusp->mat;
    if (!Cmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C mult struct for product type %s",MatProductTypes[C->product->type]);
    Ccsr = (CsrMatrix*)Cmat->mat;
    if (!Ccsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C CSR struct");
    goto finalize;
  }
  if (!c->nz) goto finalize;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for B of type %s",((PetscObject)B)->type_name);
  if (A->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  if (B->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr;
  Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
  if (Acusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  if (Bcusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  if (Ccusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSECopyToGPU(B);CHKERRQ(ierr);

  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) {
    ptype = MATPRODUCT_AB;
    if (!product->symbolic_used_the_fact_A_is_symmetric) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Symbolic should have been built using the fact that A is symmetric");
  }
  if (B->symmetric && ptype == MATPRODUCT_ABt) {
    ptype = MATPRODUCT_AB;
    if (!product->symbolic_used_the_fact_B_is_symmetric) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Symbolic should have been built using the fact that B is symmetric");
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
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  Cmat = Ccusp->mat;
  if (!Amat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  if (!Bmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
  if (!Cmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C mult struct for product type %s",MatProductTypes[ptype]);
  Acsr = (CsrMatrix*)Amat->mat;
  Bcsr = mmdata->Bcsr ? mmdata->Bcsr : (CsrMatrix*)Bmat->mat; /* B may be in compressed row storage */
  Ccsr = (CsrMatrix*)Cmat->mat;
  if (!Acsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A CSR struct");
  if (!Bcsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B CSR struct");
  if (!Ccsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C CSR struct");
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  BmatSpDescr = mmdata->Bcsr ? mmdata->matSpBDescr : Bmat->matDescr; /* B may be in compressed row storage */
  stat = cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);
  #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
    stat = cusparseSpGEMMreuse_compute(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                               mmdata->spgemmDesc);CHKERRCUSPARSE(stat);
  #else
    stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                               mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer);CHKERRCUSPARSE(stat);
    stat = cusparseSpGEMM_copy(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc);CHKERRCUSPARSE(stat);
  #endif
#else
  stat = cusparse_csr_spgemm(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get());CHKERRCUSPARSE(stat);
#endif
  ierr = PetscLogGpuFlops(mmdata->flops);CHKERRQ(ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  C->offloadmask = PETSC_OFFLOAD_GPU;
finalize:
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  ierr = PetscInfo3(C,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",C->rmap->n,C->cmap->n,c->nz);CHKERRQ(ierr);
  ierr = PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(C,"Maximum nonzeros in any row is %D\n",c->rmax);CHKERRQ(ierr);
  c->reallocs         = 0;
  C->info.mallocs    += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJCUSPARSE_SeqAIJCUSPARSE(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  Mat_SeqAIJCUSPARSE           *Acusp,*Bcusp,*Ccusp;
  Mat_SeqAIJ                   *a,*b,*c;
  Mat_SeqAIJCUSPARSEMultStruct *Amat,*Bmat,*Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscInt                     i,j,m,n,k;
  PetscBool                    flg;
  PetscErrorCode               ierr;
  cusparseStatus_t             stat;
  cudaError_t                  cerr;
  MatProductType               ptype;
  MatMatCusparse               *mmdata;
  PetscLogDouble               flops;
  PetscBool                    biscompressed,ciscompressed;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  int64_t                      C_num_rows1, C_num_cols1, C_nnz1;
  cusparseSpMatDescr_t         BmatSpDescr;
#else
  int                          cnz;
#endif
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE,opB = CUSPARSE_OPERATION_NON_TRANSPOSE; /* cuSPARSE spgemm doesn't support transpose yet */

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data not empty");
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for B of type %s",((PetscObject)B)->type_name);
  a = (Mat_SeqAIJ*)A->data;
  b = (Mat_SeqAIJ*)B->data;
  /* product data */
  ierr = PetscNew(&mmdata);CHKERRQ(ierr);
  C->product->data    = mmdata;
  C->product->destroy = MatDestroy_MatMatCusparse;

  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSECopyToGPU(B);CHKERRQ(ierr);
  Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr; /* Access spptr after MatSeqAIJCUSPARSECopyToGPU, not before */
  Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr;
  if (Acusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  if (Bcusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");

  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) {
    ptype = MATPRODUCT_AB;
    product->symbolic_used_the_fact_A_is_symmetric = PETSC_TRUE;
  }
  if (B->symmetric && ptype == MATPRODUCT_ABt) {
    ptype = MATPRODUCT_AB;
    product->symbolic_used_the_fact_B_is_symmetric = PETSC_TRUE;
  }
  biscompressed = PETSC_FALSE;
  ciscompressed = PETSC_FALSE;
  switch (ptype) {
  case MATPRODUCT_AB:
    m = A->rmap->n;
    n = B->cmap->n;
    k = A->cmap->n;
    Amat = Acusp->mat;
    Bmat = Bcusp->mat;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    k = A->rmap->n;
    ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(A);CHKERRQ(ierr);
    Amat = Acusp->matTranspose;
    Bmat = Bcusp->mat;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    k = A->cmap->n;
    ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(B);CHKERRQ(ierr);
    Amat = Acusp->mat;
    Bmat = Bcusp->matTranspose;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  /* create cusparse matrix */
  ierr  = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr  = MatSetType(C,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  c     = (Mat_SeqAIJ*)C->data;
  Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
  Cmat  = new Mat_SeqAIJCUSPARSEMultStruct;
  Ccsr  = new CsrMatrix;

  c->compressedrow.use = ciscompressed;
  if (c->compressedrow.use) { /* if a is in compressed row, than c will be in compressed row format */
    c->compressedrow.nrows = a->compressedrow.nrows;
    ierr = PetscMalloc2(c->compressedrow.nrows+1,&c->compressedrow.i,c->compressedrow.nrows,&c->compressedrow.rindex);CHKERRQ(ierr);
    ierr = PetscArraycpy(c->compressedrow.rindex,a->compressedrow.rindex,c->compressedrow.nrows);CHKERRQ(ierr);
    Ccusp->workVector  = new THRUSTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices = new THRUSTINTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices->assign(c->compressedrow.rindex,c->compressedrow.rindex + c->compressedrow.nrows);
  } else {
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Ccusp->workVector       = NULL;
    Cmat->cprowIndices      = NULL;
  }
  Ccusp->nrows    = ciscompressed ? c->compressedrow.nrows : m;
  Ccusp->mat      = Cmat;
  Ccusp->mat->mat = Ccsr;
  Ccsr->num_rows    = Ccusp->nrows;
  Ccsr->num_cols    = n;
  Ccsr->row_offsets = new THRUSTINTARRAY32(Ccusp->nrows+1);
  stat = cusparseCreateMatDescr(&Cmat->descr);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatIndexBase(Cmat->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
  stat = cusparseSetMatType(Cmat->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(Cmat->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  cerr = cudaMemcpy(Cmat->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  cerr = cudaMemcpy(Cmat->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  if (!Ccsr->num_rows || !Ccsr->num_cols || !a->nz || !b->nz) { /* cusparse raise errors in different calls when matrices have zero rows/columns! */
    thrust::fill(thrust::device,Ccsr->row_offsets->begin(),Ccsr->row_offsets->end(),0);
    c->nz = 0;
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values = new THRUSTARRAY(c->nz);
    goto finalizesym;
  }

  if (!Amat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  if (!Bmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
  Acsr = (CsrMatrix*)Amat->mat;
  if (!biscompressed) {
    Bcsr = (CsrMatrix*)Bmat->mat;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    BmatSpDescr = Bmat->matDescr;
#endif
  } else { /* we need to use row offsets for the full matrix */
    CsrMatrix *cBcsr = (CsrMatrix*)Bmat->mat;
    Bcsr = new CsrMatrix;
    Bcsr->num_rows       = B->rmap->n;
    Bcsr->num_cols       = cBcsr->num_cols;
    Bcsr->num_entries    = cBcsr->num_entries;
    Bcsr->column_indices = cBcsr->column_indices;
    Bcsr->values         = cBcsr->values;
    if (!Bcusp->rowoffsets_gpu) {
      Bcusp->rowoffsets_gpu  = new THRUSTINTARRAY32(B->rmap->n + 1);
      Bcusp->rowoffsets_gpu->assign(b->i,b->i + B->rmap->n + 1);
      ierr = PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
    }
    Bcsr->row_offsets = Bcusp->rowoffsets_gpu;
    mmdata->Bcsr = Bcsr;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    if (Bcsr->num_rows && Bcsr->num_cols) {
      stat = cusparseCreateCsr(&mmdata->matSpBDescr, Bcsr->num_rows, Bcsr->num_cols, Bcsr->num_entries,
                               Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                               Bcsr->values->data().get(),
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);CHKERRCUSPARSE(stat);
    }
    BmatSpDescr = mmdata->matSpBDescr;
#endif
  }
  if (!Acsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A CSR struct");
  if (!Bcsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B CSR struct");
  /* precompute flops count */
  if (ptype == MATPRODUCT_AB) {
    for (i=0, flops = 0; i<A->rmap->n; i++) {
      const PetscInt st = a->i[i];
      const PetscInt en = a->i[i+1];
      for (j=st; j<en; j++) {
        const PetscInt brow = a->j[j];
        flops += 2.*(b->i[brow+1] - b->i[brow]);
      }
    }
  } else if (ptype == MATPRODUCT_AtB) {
    for (i=0, flops = 0; i<A->rmap->n; i++) {
      const PetscInt anzi = a->i[i+1] - a->i[i];
      const PetscInt bnzi = b->i[i+1] - b->i[i];
      flops += (2.*anzi)*bnzi;
    }
  } else { /* TODO */
    flops = 0.;
  }

  mmdata->flops = flops;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  stat = cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);
  stat = cusparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, 0,
                          NULL, NULL, NULL,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);CHKERRCUSPARSE(stat);
  stat = cusparseSpGEMM_createDescr(&mmdata->spgemmDesc);CHKERRCUSPARSE(stat);
 #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
 {
  /* cusparseSpGEMMreuse has more reasonable APIs than cusparseSpGEMM, so we prefer to use it.
     We follow the sample code at https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm_reuse
  */
  void*  dBuffer1 = NULL;
  void*  dBuffer2 = NULL;
  void*  dBuffer3 = NULL;
  /* dBuffer4, dBuffer5 are needed by cusparseSpGEMMreuse_compute, and therefore are stored in mmdata */
  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  size_t bufferSize3 = 0;
  size_t bufferSize4 = 0;
  size_t bufferSize5 = 0;

  /*----------------------------------------------------------------------*/
  /* ask bufferSize1 bytes for external memory */
  stat = cusparseSpGEMMreuse_workEstimation(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                            CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                            &bufferSize1, NULL);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc((void**) &dBuffer1, bufferSize1);CHKERRCUDA(cerr);
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  stat = cusparseSpGEMMreuse_workEstimation(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                            CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                            &bufferSize1, dBuffer1);CHKERRCUSPARSE(stat);

  /*----------------------------------------------------------------------*/
  stat = cusparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                 CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                 &bufferSize2, NULL, &bufferSize3, NULL, &bufferSize4, NULL);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc((void**) &dBuffer2, bufferSize2);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**) &dBuffer3, bufferSize3);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**) &mmdata->dBuffer4, bufferSize4);CHKERRCUDA(cerr);
  stat = cusparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                 CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                 &bufferSize2, dBuffer2, &bufferSize3, dBuffer3, &bufferSize4, mmdata->dBuffer4);CHKERRCUSPARSE(stat);
  cerr = cudaFree(dBuffer1);CHKERRCUDA(cerr);
  cerr = cudaFree(dBuffer2);CHKERRCUDA(cerr);

  /*----------------------------------------------------------------------*/
  /* get matrix C non-zero entries C_nnz1 */
  stat  = cusparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1);CHKERRCUSPARSE(stat);
  c->nz = (PetscInt) C_nnz1;
  /* allocate matrix C */
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values         = new THRUSTARRAY(c->nz);CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  /* update matC with the new pointers */
  stat = cusparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(),
                                Ccsr->values->data().get());CHKERRCUSPARSE(stat);

  /*----------------------------------------------------------------------*/
  stat = cusparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                  CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                  &bufferSize5, NULL);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc((void**) &mmdata->dBuffer5, bufferSize5);CHKERRCUDA(cerr);
  stat = cusparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                  CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                  &bufferSize5, mmdata->dBuffer5);CHKERRCUSPARSE(stat);
  cerr = cudaFree(dBuffer3);CHKERRCUDA(cerr);
  stat = cusparseSpGEMMreuse_compute(Ccusp->handle, opA, opB,
                                     Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                     cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                     mmdata->spgemmDesc);CHKERRCUSPARSE(stat);
  ierr = PetscInfo9(C,"Buffer sizes for type %s, result %D x %D (k %D, nzA %D, nzB %D, nzC %D) are: %ldKB %ldKB\n",MatProductTypes[ptype],m,n,k,a->nz,b->nz,c->nz,bufferSize4/1024,bufferSize5/1024);CHKERRQ(ierr);
 }
 #else
  size_t bufSize2;
  /* ask bufferSize bytes for external memory */
  stat = cusparseSpGEMM_workEstimation(Ccusp->handle, opA, opB,
                                       Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                       cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                       mmdata->spgemmDesc, &bufSize2, NULL);CHKERRCUSPARSE(stat);
  cerr = cudaMalloc((void**) &mmdata->mmBuffer2, bufSize2);CHKERRCUDA(cerr);
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  stat = cusparseSpGEMM_workEstimation(Ccusp->handle, opA, opB,
                                       Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                       cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                       mmdata->spgemmDesc, &bufSize2, mmdata->mmBuffer2);CHKERRCUSPARSE(stat);
  /* ask bufferSize again bytes for external memory */
  stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                                Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                mmdata->spgemmDesc, &mmdata->mmBufferSize, NULL);CHKERRCUSPARSE(stat);
  /* The CUSPARSE documentation is not clear, nor the API
     We need both buffers to perform the operations properly!
     mmdata->mmBuffer2 does not appear anywhere in the compute/copy API
     it only appears for the workEstimation stuff, but it seems it is needed in compute, so probably the address
     is stored in the descriptor! What a messy API... */
  cerr = cudaMalloc((void**) &mmdata->mmBuffer, mmdata->mmBufferSize);CHKERRCUDA(cerr);
  /* compute the intermediate product of A * B */
  stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                                Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer);CHKERRCUSPARSE(stat);
  /* get matrix C non-zero entries C_nnz1 */
  stat = cusparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1);CHKERRCUSPARSE(stat);
  c->nz = (PetscInt) C_nnz1;
  ierr = PetscInfo9(C,"Buffer sizes for type %s, result %D x %D (k %D, nzA %D, nzB %D, nzC %D) are: %ldKB %ldKB\n",MatProductTypes[ptype],m,n,k,a->nz,b->nz,c->nz,bufSize2/1024,mmdata->mmBufferSize/1024);CHKERRQ(ierr);
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  stat = cusparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(),
                                Ccsr->values->data().get());CHKERRCUSPARSE(stat);
  stat = cusparseSpGEMM_copy(Ccusp->handle, opA, opB,
                             Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                             cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc);CHKERRCUSPARSE(stat);
 #endif // PETSC_PKG_CUDA_VERSION_GE(11,4,0)
#else
  stat = cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_HOST);CHKERRCUSPARSE(stat);
  stat = cusparseXcsrgemmNnz(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->row_offsets->data().get(), &cnz);CHKERRCUSPARSE(stat);
  c->nz = cnz;
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  CHKERRCUDA(cudaPeekAtLastError()); /* catch out of memory errors */

  stat = cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);
  /* with the old gemm interface (removed from 11.0 on) we cannot compute the symbolic factorization only.
     I have tried using the gemm2 interface (alpha * A * B + beta * D), which allows to do symbolic by passing NULL for values, but it seems quite buggy when
     D is NULL, despite the fact that CUSPARSE documentation claims it is supported! */
  stat = cusparse_csr_spgemm(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get());CHKERRCUSPARSE(stat);
#endif
  ierr = PetscLogGpuFlops(mmdata->flops);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
finalizesym:
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
  if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
    PetscInt *d_i = c->i;
    THRUSTINTARRAY ii(Ccsr->row_offsets->size());
    THRUSTINTARRAY jj(Ccsr->column_indices->size());
    ii   = *Ccsr->row_offsets;
    jj   = *Ccsr->column_indices;
    if (ciscompressed) d_i = c->compressedrow.i;
    cerr = cudaMemcpy(d_i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  } else {
    PetscInt *d_i = c->i;
    if (ciscompressed) d_i = c->compressedrow.i;
    cerr = cudaMemcpy(d_i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  }
  if (ciscompressed) { /* need to expand host row offsets */
    PetscInt r = 0;
    c->i[0] = 0;
    for (k = 0; k < c->compressedrow.nrows; k++) {
      const PetscInt next = c->compressedrow.rindex[k];
      const PetscInt old = c->compressedrow.i[k];
      for (; r < next; r++) c->i[r+1] = old;
    }
    for (; r < m; r++) c->i[r+1] = c->compressedrow.i[c->compressedrow.nrows];
  }
  ierr = PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
  c->maxnz = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k+1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax,nn);
  }
  ierr = MatMarkDiagonal_SeqAIJ(C);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
  Ccsr->num_entries = c->nz;

  C->nonzerostate++;
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  Ccusp->nonzerostate = C->nonzerostate;
  C->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_FALSE;
  C->was_assembled = PETSC_FALSE;
  if (product->api_user && A->offloadmask == PETSC_OFFLOAD_BOTH && B->offloadmask == PETSC_OFFLOAD_BOTH) { /* flag the matrix C values as computed, so that the numeric phase will only call MatAssembly */
    mmdata->reusesym = PETSC_TRUE;
    C->offloadmask   = PETSC_OFFLOAD_GPU;
  }
  C->ops->productnumeric = MatProductNumeric_SeqAIJCUSPARSE_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);

/* handles sparse or dense B */
static PetscErrorCode MatProductSetFromOptions_SeqAIJCUSPARSE(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      isdense = PETSC_FALSE,Biscusp = PETSC_FALSE,Ciscusp = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)product->B,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  if (!product->A->boundtocpu && !product->B->boundtocpu) {
    ierr = PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJCUSPARSE,&Biscusp);CHKERRQ(ierr);
  }
  if (product->type == MATPRODUCT_ABC) {
    Ciscusp = PETSC_FALSE;
    if (!product->C->boundtocpu) {
      ierr = PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJCUSPARSE,&Ciscusp);CHKERRQ(ierr);
    }
  }
  if (Biscusp && Ciscusp) { /* we can always select the CPU backend */
    PetscBool usecpu = PETSC_FALSE;
    switch (product->type) {
    case MATPRODUCT_AB:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matmatmult_backend_cpu","Use CPU code","MatMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_AB","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matproduct_ab_backend_cpu","Use CPU code","MatMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      }
      break;
    case MATPRODUCT_AtB:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatTransposeMatMult","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-mattransposematmult_backend_cpu","Use CPU code","MatTransposeMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_AtB","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matproduct_atb_backend_cpu","Use CPU code","MatTransposeMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      }
      break;
    case MATPRODUCT_PtAP:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matptap_backend_cpu","Use CPU code","MatPtAP",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_PtAP","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matproduct_ptap_backend_cpu","Use CPU code","MatPtAP",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      }
      break;
    case MATPRODUCT_RARt:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatRARt","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matrart_backend_cpu","Use CPU code","MatRARt",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_RARt","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matproduct_rart_backend_cpu","Use CPU code","MatRARt",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      }
      break;
    case MATPRODUCT_ABC:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatMatMatMult","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matmatmatmult_backend_cpu","Use CPU code","MatMatMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_ABC","Mat");CHKERRQ(ierr);
        ierr = PetscOptionsBool("-matproduct_abc_backend_cpu","Use CPU code","MatMatMatMult",usecpu,&usecpu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
     if (product->A->boundtocpu) {
        ierr = MatProductSetFromOptions_SeqAIJ_SeqDense(mat);CHKERRQ(ierr);
      } else {
        mat->ops->productsymbolic = MatProductSymbolic_SeqAIJCUSPARSE_SeqDENSECUDA;
      }
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
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJCUSPARSE_SeqAIJCUSPARSE;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else { /* fallback for AIJ */
    ierr = MatProductSetFromOptions_SeqAIJ(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

__global__ static void ScatterAdd(PetscInt n, PetscInt *idx,const PetscScalar *x,PetscScalar *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[idx[i]] += x[i];
}

/* z = op(A) x + y. If trans & !herm, op = ^T; if trans & herm, op = ^H; if !trans, op = no-op */
static PetscErrorCode MatMultAddKernel_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans,PetscBool herm)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct;
  PetscScalar                  *xarray,*zarray,*dptr,*beta,*xptr;
  PetscErrorCode               ierr;
  cusparseStatus_t             stat;
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  PetscBool                    compressed;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  PetscInt                     nx,ny;
#endif

  PetscFunctionBegin;
  if (herm && !trans) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"Hermitian and not transpose not supported");
  if (!a->nonzerorowcnt) {
    if (!yy) {ierr = VecSet_SeqCUDA(zz,0);CHKERRQ(ierr);}
    else {ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!trans) {
    matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    if (!matstruct) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"SeqAIJCUSPARSE does not have a 'mat' (need to fix)");
  } else {
    if (herm || !A->form_explicit_transpose) {
      opA = herm ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
      matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    } else {
      if (!cusparsestruct->matTranspose) {ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(A);CHKERRQ(ierr);}
      matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->matTranspose;
    }
  }
  /* Does the matrix use compressed rows (i.e., drop zero rows)? */
  compressed = matstruct->cprowIndices ? PETSC_TRUE : PETSC_FALSE;

  try {
    ierr = VecCUDAGetArrayRead(xx,(const PetscScalar**)&xarray);CHKERRQ(ierr);
    if (yy == zz) {ierr = VecCUDAGetArray(zz,&zarray);CHKERRQ(ierr);} /* read & write zz, so need to get uptodate zarray on GPU */
    else {ierr = VecCUDAGetArrayWrite(zz,&zarray);CHKERRQ(ierr);} /* write zz, so no need to init zarray on GPU */

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      /* z = A x + beta y.
         If A is compressed (with less rows), then Ax is shorter than the full z, so we need a work vector to store Ax.
         When A is non-compressed, and z = y, we can set beta=1 to compute y = Ax + y in one call.
      */
      xptr = xarray;
      dptr = compressed ? cusparsestruct->workVector->data().get() : zarray;
      beta = (yy == zz && !compressed) ? matstruct->beta_one : matstruct->beta_zero;
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      /* Get length of x, y for y=Ax. ny might be shorter than the work vector's allocated length, since the work vector is
          allocated to accommodate different uses. So we get the length info directly from mat.
       */
      if (cusparsestruct->format == MAT_CUSPARSE_CSR) {
        CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
        nx = mat->num_cols;
        ny = mat->num_rows;
      }
     #endif
    } else {
      /* z = A^T x + beta y
         If A is compressed, then we need a work vector as the shorter version of x to compute A^T x.
         Note A^Tx is of full length, so we set beta to 1.0 if y exists.
       */
      xptr = compressed ? cusparsestruct->workVector->data().get() : xarray;
      dptr = zarray;
      beta = yy ? matstruct->beta_one : matstruct->beta_zero;
      if (compressed) { /* Scatter x to work vector */
        thrust::device_ptr<PetscScalar> xarr = thrust::device_pointer_cast(xarray);
        thrust::for_each(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecCUDAEqualsReverse());
      }
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      if (cusparsestruct->format == MAT_CUSPARSE_CSR) {
        CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
        nx = mat->num_rows;
        ny = mat->num_cols;
      }
     #endif
    }

    /* csr_spmv does y = alpha op(A) x + beta y */
    if (cusparsestruct->format == MAT_CUSPARSE_CSR) {
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      if (opA < 0 || opA > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE ABI on cusparseOperation_t has changed and PETSc has not been updated accordingly");
      if (!matstruct->cuSpMV[opA].initialized) { /* built on demand */
        cudaError_t cerr;
        stat = cusparseCreateDnVec(&matstruct->cuSpMV[opA].vecXDescr,nx,xptr,cusparse_scalartype);CHKERRCUSPARSE(stat);
        stat = cusparseCreateDnVec(&matstruct->cuSpMV[opA].vecYDescr,ny,dptr,cusparse_scalartype);CHKERRCUSPARSE(stat);
        stat = cusparseSpMV_bufferSize(cusparsestruct->handle, opA, matstruct->alpha_one,
                                matstruct->matDescr,
                                matstruct->cuSpMV[opA].vecXDescr, beta,
                                matstruct->cuSpMV[opA].vecYDescr,
                                cusparse_scalartype,
                                cusparsestruct->spmvAlg,
                                &matstruct->cuSpMV[opA].spmvBufferSize);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc(&matstruct->cuSpMV[opA].spmvBuffer,matstruct->cuSpMV[opA].spmvBufferSize);CHKERRCUDA(cerr);

        matstruct->cuSpMV[opA].initialized = PETSC_TRUE;
      } else {
        /* x, y's value pointers might change between calls, but their shape is kept, so we just update pointers */
        stat = cusparseDnVecSetValues(matstruct->cuSpMV[opA].vecXDescr,xptr);CHKERRCUSPARSE(stat);
        stat = cusparseDnVecSetValues(matstruct->cuSpMV[opA].vecYDescr,dptr);CHKERRCUSPARSE(stat);
      }

      stat = cusparseSpMV(cusparsestruct->handle, opA,
                               matstruct->alpha_one,
                               matstruct->matDescr, /* built in MatSeqAIJCUSPARSECopyToGPU() or MatSeqAIJCUSPARSEFormExplicitTranspose() */
                               matstruct->cuSpMV[opA].vecXDescr,
                               beta,
                               matstruct->cuSpMV[opA].vecYDescr,
                               cusparse_scalartype,
                               cusparsestruct->spmvAlg,
                               matstruct->cuSpMV[opA].spmvBuffer);CHKERRCUSPARSE(stat);
     #else
      CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
      stat = cusparse_csr_spmv(cusparsestruct->handle, opA,
                               mat->num_rows, mat->num_cols,
                               mat->num_entries, matstruct->alpha_one, matstruct->descr,
                               mat->values->data().get(), mat->row_offsets->data().get(),
                               mat->column_indices->data().get(), xptr, beta,
                               dptr);CHKERRCUSPARSE(stat);
     #endif
    } else {
      if (cusparsestruct->nrows) {
       #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
       #else
        cusparseHybMat_t hybMat = (cusparseHybMat_t)matstruct->mat;
        stat = cusparse_hyb_spmv(cusparsestruct->handle, opA,
                                 matstruct->alpha_one, matstruct->descr, hybMat,
                                 xptr, beta,
                                 dptr);CHKERRCUSPARSE(stat);
       #endif
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    if (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      if (yy) { /* MatMultAdd: zz = A*xx + yy */
        if (compressed) { /* A is compressed. We first copy yy to zz, then ScatterAdd the work vector to zz */
          ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr); /* zz = yy */
        } else if (zz != yy) { /* A is not compressed. zz already contains A*xx, and we just need to add yy */
          ierr = VecAXPY_SeqCUDA(zz,1.0,yy);CHKERRQ(ierr); /* zz += yy */
        }
      } else if (compressed) { /* MatMult: zz = A*xx. A is compressed, so we zero zz first, then ScatterAdd the work vector to zz */
        ierr = VecSet_SeqCUDA(zz,0);CHKERRQ(ierr);
      }

      /* ScatterAdd the result from work vector into the full vector when A is compressed */
      if (compressed) {
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        /* I wanted to make this for_each asynchronous but failed. thrust::async::for_each() returns an event (internally registerred)
           and in the destructor of the scope, it will call cudaStreamSynchronize() on this stream. One has to store all events to
           prevent that. So I just add a ScatterAdd kernel.
         */
       #if 0
        thrust::device_ptr<PetscScalar> zptr = thrust::device_pointer_cast(zarray);
        thrust::async::for_each(thrust::cuda::par.on(cusparsestruct->stream),
                         thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecCUDAPlusEquals());
       #else
        PetscInt n = matstruct->cprowIndices->size();
        ScatterAdd<<<(n+255)/256,256,0,PetscDefaultCudaStream>>>(n,matstruct->cprowIndices->data().get(),cusparsestruct->workVector->data().get(),zarray);
       #endif
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      }
    } else {
      if (yy && yy != zz) {
        ierr = VecAXPY_SeqCUDA(zz,1.0,yy);CHKERRQ(ierr); /* zz += yy */
      }
    }
    ierr = VecCUDARestoreArrayRead(xx,(const PetscScalar**)&xarray);CHKERRQ(ierr);
    if (yy == zz) {ierr = VecCUDARestoreArray(zz,&zarray);CHKERRQ(ierr);}
    else {ierr = VecCUDARestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);}
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  if (yy) {
    ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  } else {
    ierr = PetscLogGpuFlops(2.0*a->nz-a->nonzerorowcnt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode     ierr;
  PetscObjectState   onnz = A->nonzerostate;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (onnz != A->nonzerostate && cusp->deviceMat) {
    cudaError_t cerr;

    ierr = PetscInfo(A,"Destroy device mat since nonzerostate changed\n");CHKERRQ(ierr);
    cerr = cudaFree(cusp->deviceMat);CHKERRCUDA(cerr);
    cusp->deviceMat = NULL;
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@
   MatCreateSeqAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSEQAIJCUSPARSE, MATAIJCUSPARSE
@*/
PetscErrorCode  MatCreateSeqAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = MatSeqAIJCUSPARSE_Destroy((Mat_SeqAIJCUSPARSE**)&A->spptr);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJCUSPARSETriFactors_Destroy((Mat_SeqAIJCUSPARSETriFactors**)&A->spptr);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqaijcusparse_hypre_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUSPARSE(Mat,MatType,MatReuse,Mat*);
static PetscErrorCode MatBindToCPU_SeqAIJCUSPARSE(Mat,PetscBool);
static PetscErrorCode MatDuplicate_SeqAIJCUSPARSE(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCUSPARSE(*B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqAIJCUSPARSE(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;
  Mat_SeqAIJCUSPARSE *cy;
  Mat_SeqAIJCUSPARSE *cx;
  PetscScalar        *ay;
  const PetscScalar  *ax;
  CsrMatrix          *csry,*csrx;

  PetscFunctionBegin;
  cy = (Mat_SeqAIJCUSPARSE*)Y->spptr;
  cx = (Mat_SeqAIJCUSPARSE*)X->spptr;
  if (X->ops->axpy != Y->ops->axpy) {
    ierr = MatSeqAIJCUSPARSEInvalidateTranspose(Y,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* if we are here, it means both matrices are bound to GPU */
  ierr = MatSeqAIJCUSPARSECopyToGPU(Y);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSECopyToGPU(X);CHKERRQ(ierr);
  if (cy->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_GPU,"only MAT_CUSPARSE_CSR supported");
  if (cx->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_GPU,"only MAT_CUSPARSE_CSR supported");
  csry = (CsrMatrix*)cy->mat->mat;
  csrx = (CsrMatrix*)cx->mat->mat;
  /* see if we can turn this into a cublas axpy */
  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz && !x->compressedrow.use && !y->compressedrow.use) {
    bool eq = thrust::equal(thrust::device,csry->row_offsets->begin(),csry->row_offsets->end(),csrx->row_offsets->begin());
    if (eq) {
      eq = thrust::equal(thrust::device,csry->column_indices->begin(),csry->column_indices->end(),csrx->column_indices->begin());
    }
    if (eq) str = SAME_NONZERO_PATTERN;
  }
  /* spgeam is buggy with one column */
  if (Y->cmap->n == 1 && str != SAME_NONZERO_PATTERN) str = DIFFERENT_NONZERO_PATTERN;

  if (str == SUBSET_NONZERO_PATTERN) {
    cusparseStatus_t stat;
    PetscScalar      b = 1.0;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    size_t           bufferSize;
    void             *buffer;
    cudaError_t      cerr;
#endif

    ierr = MatSeqAIJCUSPARSEGetArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
    stat = cusparseSetPointerMode(cy->handle, CUSPARSE_POINTER_MODE_HOST);CHKERRCUSPARSE(stat);
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    stat = cusparse_csr_spgeam_bufferSize(cy->handle,Y->rmap->n,Y->cmap->n,
                                          &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                                          &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                             cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),&bufferSize);CHKERRCUSPARSE(stat);
    cerr = cudaMalloc(&buffer,bufferSize);CHKERRCUDA(cerr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    stat = cusparse_csr_spgeam(cy->handle,Y->rmap->n,Y->cmap->n,
                               &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                               &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                  cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),buffer);CHKERRCUSPARSE(stat);
    ierr = PetscLogGpuFlops(x->nz + y->nz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    cerr = cudaFree(buffer);CHKERRCUDA(cerr);
#else
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    stat = cusparse_csr_spgeam(cy->handle,Y->rmap->n,Y->cmap->n,
                               &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                               &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                  cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get());CHKERRCUSPARSE(stat);
    ierr = PetscLogGpuFlops(x->nz + y->nz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    stat = cusparseSetPointerMode(cy->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);
    ierr = MatSeqAIJCUSPARSERestoreArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
    ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  } else if (str == SAME_NONZERO_PATTERN) {
    cublasHandle_t cublasv2handle;
    cublasStatus_t berr;
    PetscBLASInt   one = 1, bnz = 1;

    ierr = MatSeqAIJCUSPARSEGetArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(x->nz,&bnz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    berr = cublasXaxpy(cublasv2handle,bnz,&a,ax,one,ay,one);CHKERRCUBLAS(berr);
    ierr = PetscLogGpuFlops(2.0*bnz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSERestoreArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
    ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJCUSPARSEInvalidateTranspose(Y,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJCUSPARSE(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *y = (Mat_SeqAIJ*)Y->data;
  PetscScalar    *ay;
  cublasHandle_t cublasv2handle;
  cublasStatus_t berr;
  PetscBLASInt   one = 1, bnz = 1;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(y->nz,&bnz);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXscal(cublasv2handle,bnz,&a,ay,one);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuFlops(bnz);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;
  PetscBool      both = PETSC_FALSE;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    if (spptr->mat) {
      CsrMatrix* matrix = (CsrMatrix*)spptr->mat->mat;
      if (matrix->values) {
        both = PETSC_TRUE;
        thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
      }
    }
    if (spptr->matTranspose) {
      CsrMatrix* matrix = (CsrMatrix*)spptr->matTranspose->mat;
      if (matrix->values) {
        thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
      }
    }
  }
  //ierr = MatZeroEntries_SeqAIJ(A);CHKERRQ(ierr);
  ierr = PetscArrayzero(a->a,a->i[A->rmap->n]);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqAIJCUSPARSE(Mat A,PetscBool flg)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) PetscFunctionReturn(0);
  if (flg) {
    ierr = MatSeqAIJCUSPARSECopyFromGPU(A);CHKERRQ(ierr);

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
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJ);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",NULL);CHKERRQ(ierr);
  } else {
    A->ops->scale                     = MatScale_SeqAIJCUSPARSE;
    A->ops->axpy                      = MatAXPY_SeqAIJCUSPARSE;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJCUSPARSE;
    A->ops->mult                      = MatMult_SeqAIJCUSPARSE;
    A->ops->multadd                   = MatMultAdd_SeqAIJCUSPARSE;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJCUSPARSE;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJCUSPARSE;
    A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJCUSPARSE;
    A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJCUSPARSE;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJCUSPARSE;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",MatSeqAIJCopySubArray_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",MatProductSetFromOptions_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",MatProductSetFromOptions_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",MatProductSetFromOptions_SeqAIJCUSPARSE);CHKERRQ(ierr);
  }
  A->boundtocpu = flg;
  a->inode.use = flg;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUSPARSE(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode   ierr;
  cusparseStatus_t stat;
  Mat              B;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr); /* first use of CUSPARSE may be via MatConvert */
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  B = *newmat;

  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);

  if (reuse != MAT_REUSE_MATRIX && !B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      Mat_SeqAIJCUSPARSE *spptr;
      ierr = PetscNew(&spptr);CHKERRQ(ierr);
      stat = cusparseCreate(&spptr->handle);CHKERRCUSPARSE(stat);
      stat = cusparseSetStream(spptr->handle,PetscDefaultCudaStream);CHKERRCUSPARSE(stat);
      spptr->format     = MAT_CUSPARSE_CSR;
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
     #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
      spptr->spmvAlg    = CUSPARSE_SPMV_CSR_ALG1; /* default, since we only support csr */
     #else
      spptr->spmvAlg    = CUSPARSE_CSRMV_ALG1;    /* default, since we only support csr */
     #endif
      spptr->spmmAlg    = CUSPARSE_SPMM_CSR_ALG1; /* default, only support column-major dense matrix B */
      spptr->csr2cscAlg = CUSPARSE_CSR2CSC_ALG1;
     #endif
      B->spptr = spptr;
    } else {
      Mat_SeqAIJCUSPARSETriFactors *spptr;

      ierr = PetscNew(&spptr);CHKERRQ(ierr);
      stat = cusparseCreate(&spptr->handle);CHKERRCUSPARSE(stat);
      stat = cusparseSetStream(spptr->handle,PetscDefaultCudaStream);CHKERRCUSPARSE(stat);
      B->spptr = spptr;
    }
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSPARSE;
  B->ops->destroy        = MatDestroy_SeqAIJCUSPARSE;
  B->ops->setoption      = MatSetOption_SeqAIJCUSPARSE;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJCUSPARSE;
  B->ops->bindtocpu      = MatBindToCPU_SeqAIJCUSPARSE;
  B->ops->duplicate      = MatDuplicate_SeqAIJCUSPARSE;

  ierr = MatBindToCPU_SeqAIJCUSPARSE(B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_SeqAIJCUSPARSE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijcusparse_hypre_C",MatConvert_AIJ_HYPRE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCUSPARSE(B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJCUSPARSE - MATAIJCUSPARSE = "(seq)aijcusparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the CUSPARSE library.

   Options Database Keys:
+  -mat_type aijcusparse - sets the matrix type to "seqaijcusparse" during a call to MatSetFromOptions()
.  -mat_cusparse_storage_format csr - sets the storage format of matrices (for MatMult and factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_storage_format csr - sets the storage format of matrices (for MatMult) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

.seealso: MatCreateSeqAIJCUSPARSE(), MATAIJCUSPARSE, MatCreateAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijcusparse_cusparse_band(Mat,MatFactorType,Mat*);

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_CUSPARSE(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERCUSPARSEBAND, MATSEQAIJ, MAT_FACTOR_LU,MatGetFactor_seqaijcusparse_cusparse_band);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_LU,MatGetFactor_seqaijcusparse_cusparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_CHOLESKY,MatGetFactor_seqaijcusparse_cusparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_ILU,MatGetFactor_seqaijcusparse_cusparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_ICC,MatGetFactor_seqaijcusparse_cusparse);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSE_Destroy(Mat_SeqAIJCUSPARSE **cusparsestruct)
{
  PetscErrorCode   ierr;
  cusparseStatus_t stat;

  PetscFunctionBegin;
  if (*cusparsestruct) {
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*cusparsestruct)->mat,(*cusparsestruct)->format);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*cusparsestruct)->matTranspose,(*cusparsestruct)->format);CHKERRQ(ierr);
    delete (*cusparsestruct)->workVector;
    delete (*cusparsestruct)->rowoffsets_gpu;
    delete (*cusparsestruct)->cooPerm;
    delete (*cusparsestruct)->cooPerm_a;
    delete (*cusparsestruct)->csr2csc_i;
    if ((*cusparsestruct)->handle) {stat = cusparseDestroy((*cusparsestruct)->handle);CHKERRCUSPARSE(stat);}
    ierr = PetscFree(*cusparsestruct);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEMultStruct_Destroy(Mat_SeqAIJCUSPARSETriFactorStruct **trifactor)
{
  cusparseStatus_t stat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (*trifactor) {
    if ((*trifactor)->descr) { stat = cusparseDestroyMatDescr((*trifactor)->descr);CHKERRCUSPARSE(stat); }
    if ((*trifactor)->solveInfo) { stat = cusparse_destroy_analysis_info((*trifactor)->solveInfo);CHKERRCUSPARSE(stat); }
    ierr = CsrMatrix_Destroy(&(*trifactor)->csrMat);CHKERRQ(ierr);
    if ((*trifactor)->solveBuffer)   {cudaError_t cerr = cudaFree((*trifactor)->solveBuffer);CHKERRCUDA(cerr);}
    if ((*trifactor)->AA_h)   {cudaError_t cerr = cudaFreeHost((*trifactor)->AA_h);CHKERRCUDA(cerr);}
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    if ((*trifactor)->csr2cscBuffer) {cudaError_t cerr = cudaFree((*trifactor)->csr2cscBuffer);CHKERRCUDA(cerr);}
   #endif
    ierr = PetscFree(*trifactor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEMultStruct_Destroy(Mat_SeqAIJCUSPARSEMultStruct **matstruct,MatCUSPARSEStorageFormat format)
{
  CsrMatrix        *mat;
  cusparseStatus_t stat;
  cudaError_t      err;

  PetscFunctionBegin;
  if (*matstruct) {
    if ((*matstruct)->mat) {
      if (format==MAT_CUSPARSE_ELL || format==MAT_CUSPARSE_HYB) {
       #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
       #else
        cusparseHybMat_t hybMat = (cusparseHybMat_t)(*matstruct)->mat;
        stat = cusparseDestroyHybMat(hybMat);CHKERRCUSPARSE(stat);
       #endif
      } else {
        mat = (CsrMatrix*)(*matstruct)->mat;
        CsrMatrix_Destroy(&mat);
      }
    }
    if ((*matstruct)->descr) { stat = cusparseDestroyMatDescr((*matstruct)->descr);CHKERRCUSPARSE(stat); }
    delete (*matstruct)->cprowIndices;
    if ((*matstruct)->alpha_one) { err=cudaFree((*matstruct)->alpha_one);CHKERRCUDA(err); }
    if ((*matstruct)->beta_zero) { err=cudaFree((*matstruct)->beta_zero);CHKERRCUDA(err); }
    if ((*matstruct)->beta_one)  { err=cudaFree((*matstruct)->beta_one);CHKERRCUDA(err); }

   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    Mat_SeqAIJCUSPARSEMultStruct *mdata = *matstruct;
    if (mdata->matDescr) {stat = cusparseDestroySpMat(mdata->matDescr);CHKERRCUSPARSE(stat);}
    for (int i=0; i<3; i++) {
      if (mdata->cuSpMV[i].initialized) {
        err  = cudaFree(mdata->cuSpMV[i].spmvBuffer);CHKERRCUDA(err);
        stat = cusparseDestroyDnVec(mdata->cuSpMV[i].vecXDescr);CHKERRCUSPARSE(stat);
        stat = cusparseDestroyDnVec(mdata->cuSpMV[i].vecYDescr);CHKERRCUSPARSE(stat);
      }
    }
   #endif
    delete *matstruct;
    *matstruct = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJCUSPARSETriFactors_Reset(Mat_SeqAIJCUSPARSETriFactors_p* trifactors)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*trifactors) {
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtr);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtr);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtrTranspose);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtrTranspose);CHKERRQ(ierr);
    delete (*trifactors)->rpermIndices;
    delete (*trifactors)->cpermIndices;
    delete (*trifactors)->workVector;
    (*trifactors)->rpermIndices = NULL;
    (*trifactors)->cpermIndices = NULL;
    (*trifactors)->workVector = NULL;
    if ((*trifactors)->a_band_d)   {cudaError_t cerr = cudaFree((*trifactors)->a_band_d);CHKERRCUDA(cerr);}
    if ((*trifactors)->i_band_d)   {cudaError_t cerr = cudaFree((*trifactors)->i_band_d);CHKERRCUDA(cerr);}
    (*trifactors)->init_dev_prop = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Destroy(Mat_SeqAIJCUSPARSETriFactors** trifactors)
{
  PetscErrorCode   ierr;
  cusparseHandle_t handle;
  cusparseStatus_t stat;

  PetscFunctionBegin;
  if (*trifactors) {
    ierr = MatSeqAIJCUSPARSETriFactors_Reset(trifactors);CHKERRQ(ierr);
    if (handle = (*trifactors)->handle) {
      stat = cusparseDestroy(handle);CHKERRCUSPARSE(stat);
    }
    ierr = PetscFree(*trifactors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct IJCompare
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct IJEqual
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() != t2.get<0>() || t1.get<1>() != t2.get<1>()) return false;
    return true;
  }
};

struct IJDiff
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1 == t2 ? 0 : 1;
  }
};

struct IJSum
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1||t2;
  }
};

#include <thrust/iterator/discard_iterator.h>
PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJCUSPARSE                    *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ                            *a = (Mat_SeqAIJ*)A->data;
  THRUSTARRAY                           *cooPerm_v = NULL;
  thrust::device_ptr<const PetscScalar> d_v;
  CsrMatrix                             *matrix;
  PetscErrorCode                        ierr;
  PetscInt                              n;

  PetscFunctionBegin;
  if (!cusp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE struct");
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE CsrMatrix");
  if (!cusp->cooPerm) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  matrix = (CsrMatrix*)cusp->mat->mat;
  if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  if (!v) {
    if (imode == INSERT_VALUES) thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
    goto finalize;
  }
  n = cusp->cooPerm->size();
  if (isCudaMem(v)) {
    d_v = thrust::device_pointer_cast(v);
  } else {
    cooPerm_v = new THRUSTARRAY(n);
    cooPerm_v->assign(v,v+n);
    d_v = cooPerm_v->data();
    ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (imode == ADD_VALUES) { /* ADD VALUES means add to existing ones */
    if (cusp->cooPerm_a) { /* there are repeated entries in d_v[], and we need to add these them */
      THRUSTARRAY *cooPerm_w = new THRUSTARRAY(matrix->values->size());
      auto vbit = thrust::make_permutation_iterator(d_v,cusp->cooPerm->begin());
      /* thrust::reduce_by_key(keys_first,keys_last,values_first,keys_output,values_output)
        cooPerm_a = [0,0,1,2,3,4]. The length is n, number of nonozeros in d_v[].
        cooPerm_a is ordered. d_v[i] is the cooPerm_a[i]-th unique nonzero.
      */
      thrust::reduce_by_key(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),vbit,thrust::make_discard_iterator(),cooPerm_w->begin(),thrust::equal_to<PetscInt>(),thrust::plus<PetscScalar>());
      thrust::transform(cooPerm_w->begin(),cooPerm_w->end(),matrix->values->begin(),matrix->values->begin(),thrust::plus<PetscScalar>());
      delete cooPerm_w;
    } else {
      /* all nonzeros in d_v[] are unique entries */
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecCUDAPlusEquals()); /* values[i] += d_v[cooPerm[i]]  */
    }
  } else {
    if (cusp->cooPerm_a) { /* repeated entries in COO, with INSERT_VALUES -> reduce */
      auto vbit = thrust::make_permutation_iterator(d_v,cusp->cooPerm->begin());
      thrust::reduce_by_key(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),vbit,thrust::make_discard_iterator(),matrix->values->begin(),thrust::equal_to<PetscInt>(),thrust::plus<PetscScalar>());
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecCUDAEquals());
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
finalize:
  delete cooPerm_v;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  ierr = PetscInfo3(A,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",A->rmap->n,A->cmap->n,a->nz);CHKERRQ(ierr);
  ierr = PetscInfo(A,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",a->rmax);CHKERRQ(ierr);
  a->reallocs         = 0;
  A->info.mallocs    += 0;
  A->info.nz_unneeded = 0;
  A->assembled = A->was_assembled = PETSC_TRUE;
  A->num_ass++;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJCUSPARSEInvalidateTranspose(Mat A, PetscBool destroy)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (!cusp) PetscFunctionReturn(0);
  if (destroy) {
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusp->matTranspose,cusp->format);CHKERRQ(ierr);
    delete cusp->csr2csc_i;
    cusp->csr2csc_i = NULL;
  }
  A->transupdated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#include <thrust/binary_search.h>
PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat A, PetscInt n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  PetscErrorCode     ierr;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           cooPerm_n, nzr = 0;
  cudaError_t        cerr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  cooPerm_n = cusp->cooPerm ? cusp->cooPerm->size() : 0;
  if (n != cooPerm_n) {
    delete cusp->cooPerm;
    delete cusp->cooPerm_a;
    cusp->cooPerm = NULL;
    cusp->cooPerm_a = NULL;
  }
  if (n) {
    THRUSTINTARRAY d_i(n);
    THRUSTINTARRAY d_j(n);
    THRUSTINTARRAY ii(A->rmap->n);

    if (!cusp->cooPerm)   { cusp->cooPerm   = new THRUSTINTARRAY(n); }
    if (!cusp->cooPerm_a) { cusp->cooPerm_a = new THRUSTINTARRAY(n); }

    ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
    d_i.assign(coo_i,coo_i+n);
    d_j.assign(coo_j,coo_j+n);

    /* Ex.
      n = 6
      coo_i = [3,3,1,4,1,4]
      coo_j = [3,2,2,5,2,6]
    */
    auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin()));
    auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end()));

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    thrust::sequence(thrust::device, cusp->cooPerm->begin(), cusp->cooPerm->end(), 0);
    thrust::sort_by_key(fkey, ekey, cusp->cooPerm->begin(), IJCompare()); /* sort by row, then by col */
    *cusp->cooPerm_a = d_i; /* copy the sorted array */
    THRUSTINTARRAY w = d_j;

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
      adjacent_difference(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),cusp->cooPerm_a->begin(),IJDiff()); /* cooPerm_a: [1,1,3,3,4,4] => [1,0,1,0,1,0]*/
      adjacent_difference(w.begin(),w.end(),w.begin(),IJDiff());                                              /* w:         [2,2,2,3,5,6] => [2,0,0,1,1,1]*/
      (*cusp->cooPerm_a)[0] = 0; /* clear the first entry, though accessing an entry on device implies a cudaMemcpy */
      w[0] = 0;
      thrust::transform(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),w.begin(),cusp->cooPerm_a->begin(),IJSum()); /* cooPerm_a =          [0,0,1,1,1,1]*/
      thrust::inclusive_scan(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),cusp->cooPerm_a->begin(),thrust::plus<PetscInt>()); /*cooPerm_a=[0,0,1,2,3,4]*/
    }
    thrust::counting_iterator<PetscInt> search_begin(0);
    thrust::upper_bound(d_i.begin(), nekey.get_iterator_tuple().get<0>(), /* binary search entries of [0,1,2,3,4,5,6) in ordered array d_i = [1,3,3,4,4], supposing A->rmap->n = 6. */
                        search_begin, search_begin + A->rmap->n,  /* return in ii[] the index of last position in d_i[] where value could be inserted without violating the ordering */
                        ii.begin()); /* ii = [0,1,1,3,5,5]. A leading 0 will be added later */
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
    a->singlemalloc = PETSC_FALSE;
    a->free_a       = PETSC_TRUE;
    a->free_ij      = PETSC_TRUE;
    ierr = PetscMalloc1(A->rmap->n+1,&a->i);CHKERRQ(ierr);
    a->i[0] = 0; /* a->i = [0,0,1,1,3,5,5] */
    cerr = cudaMemcpy(a->i+1,ii.data().get(),A->rmap->n*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    a->nz = a->maxnz = a->i[A->rmap->n];
    a->rmax = 0;
    ierr = PetscMalloc1(a->nz,&a->a);CHKERRQ(ierr);
    ierr = PetscMalloc1(a->nz,&a->j);CHKERRQ(ierr);
    cerr = cudaMemcpy(a->j,d_j.data().get(),a->nz*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    if (!a->ilen) { ierr = PetscMalloc1(A->rmap->n,&a->ilen);CHKERRQ(ierr); }
    if (!a->imax) { ierr = PetscMalloc1(A->rmap->n,&a->imax);CHKERRQ(ierr); }
    for (PetscInt i = 0; i < A->rmap->n; i++) {
      const PetscInt nnzr = a->i[i+1] - a->i[i];
      nzr += (PetscInt)!!(nnzr);
      a->ilen[i] = a->imax[i] = nnzr;
      a->rmax = PetscMax(a->rmax,nnzr);
    }
    a->nonzerorowcnt = nzr;
    A->preallocated = PETSC_TRUE;
    ierr = PetscLogGpuToCpu((A->rmap->n+a->nz)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJSetPreallocation(A,0,NULL);CHKERRQ(ierr);
  }
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  /* We want to allocate the CUSPARSE struct for matvec now.
     The code is so convoluted now that I prefer to copy zeros */
  ierr = PetscArrayzero(a->a,a->nz);CHKERRQ(ierr);
  ierr = MatCheckCompressedRow(A,nzr,&a->compressedrow,a->i,A->rmap->n,0.6);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_CPU;
  A->nonzerostate++;
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);

  A->assembled = PETSC_FALSE;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
    MatSeqAIJCUSPARSEGetIJ - returns the device row storage i and j indices for MATSEQAIJCUSPARSE matrices.

   Not collective

    Input Parameters:
+   A - the matrix
-   compressed - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be always returned in compressed form

    Output Parameters:
+   ia - the CSR row pointers
-   ja - the CSR column indices

    Level: developer

    Notes:
      When compressed is true, the CSR structure does not contain empty rows

.seealso: MatSeqAIJCUSPARSERestoreIJ(), MatSeqAIJCUSPARSEGetArrayRead()
@*/
PetscErrorCode MatSeqAIJCUSPARSEGetIJ(Mat A, PetscBool compressed, const int** i, const int **j)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!i || !j) PetscFunctionReturn(0);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  if (i) {
    if (!compressed && a->compressedrow.use) { /* need full row offset */
      if (!cusp->rowoffsets_gpu) {
        cusp->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
        cusp->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
        ierr = PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
      }
      *i = cusp->rowoffsets_gpu->data().get();
    } else *i = csr->row_offsets->data().get();
  }
  if (j) *j = csr->column_indices->data().get();
  PetscFunctionReturn(0);
}

/*@C
    MatSeqAIJCUSPARSERestoreIJ - restore the device row storage i and j indices obtained with MatSeqAIJCUSPARSEGetIJ()

   Not collective

    Input Parameters:
+   A - the matrix
-   compressed - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be always returned in compressed form

    Output Parameters:
+   ia - the CSR row pointers
-   ja - the CSR column indices

    Level: developer

.seealso: MatSeqAIJCUSPARSEGetIJ()
@*/
PetscErrorCode MatSeqAIJCUSPARSERestoreIJ(Mat A, PetscBool compressed, const int** i, const int **j)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (i) *i = NULL;
  if (j) *j = NULL;
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJCUSPARSEGetArrayRead - gives read-only access to the array where the device data for a MATSEQAIJCUSPARSE matrix is stored

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Notes: may trigger host-device copies if up-to-date matrix data is on host

.seealso: MatSeqAIJCUSPARSEGetArray(), MatSeqAIJCUSPARSEGetArrayWrite(), MatSeqAIJCUSPARSERestoreArrayRead()
@*/
PetscErrorCode MatSeqAIJCUSPARSEGetArrayRead(Mat A, const PetscScalar** a)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  *a = csr->values->data().get();
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJCUSPARSERestoreArrayRead - restore the read-only access array obtained from MatSeqAIJCUSPARSEGetArrayRead()

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

.seealso: MatSeqAIJCUSPARSEGetArrayRead()
@*/
PetscErrorCode MatSeqAIJCUSPARSERestoreArrayRead(Mat A, const PetscScalar** a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJCUSPARSEGetArray - gives read-write access to the array where the device data for a MATSEQAIJCUSPARSE matrix is stored

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Notes: may trigger host-device copies if up-to-date matrix data is on host

.seealso: MatSeqAIJCUSPARSEGetArrayRead(), MatSeqAIJCUSPARSEGetArrayWrite(), MatSeqAIJCUSPARSERestoreArray()
@*/
PetscErrorCode MatSeqAIJCUSPARSEGetArray(Mat A, PetscScalar** a)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*@C
   MatSeqAIJCUSPARSERestoreArray - restore the read-write access array obtained from MatSeqAIJCUSPARSEGetArray()

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

.seealso: MatSeqAIJCUSPARSEGetArray()
@*/
PetscErrorCode MatSeqAIJCUSPARSERestoreArray(Mat A, PetscScalar** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJCUSPARSEGetArrayWrite - gives write access to the array where the device data for a MATSEQAIJCUSPARSE matrix is stored

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

   Notes: does not trigger host-device copies and flags data validity on the GPU

.seealso: MatSeqAIJCUSPARSEGetArray(), MatSeqAIJCUSPARSEGetArrayRead(), MatSeqAIJCUSPARSERestoreArrayWrite()
@*/
PetscErrorCode MatSeqAIJCUSPARSEGetArrayWrite(Mat A, PetscScalar** a)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJCUSPARSERestoreArrayWrite - restore the write-only access array obtained from MatSeqAIJCUSPARSEGetArrayWrite()

   Not Collective

   Input Parameter:
.   A - a MATSEQAIJCUSPARSE matrix

   Output Parameter:
.   a - pointer to the device data

   Level: developer

.seealso: MatSeqAIJCUSPARSEGetArrayWrite()
@*/
PetscErrorCode MatSeqAIJCUSPARSERestoreArrayWrite(Mat A, PetscScalar** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  *a = NULL;
  PetscFunctionReturn(0);
}

struct IJCompare4
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<int, int, PetscScalar, int> &t1, const thrust::tuple<int, int, PetscScalar, int> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct Shift
{
  int _shift;

  Shift(int shift) : _shift(shift) {}
  __host__ __device__
  inline int operator() (const int &c)
  {
    return c + _shift;
  }
};

/* merges two SeqAIJCUSPARSE matrices A, B by concatenating their rows. [A';B']' operation in matlab notation */
PetscErrorCode MatSeqAIJCUSPARSEMergeMats(Mat A,Mat B,MatReuse reuse,Mat* C)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data, *b = (Mat_SeqAIJ*)B->data, *c;
  Mat_SeqAIJCUSPARSE           *Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr, *Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr, *Ccusp;
  Mat_SeqAIJCUSPARSEMultStruct *Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscInt                     Annz,Bnnz;
  cusparseStatus_t             stat;
  PetscInt                     i,m,n,zero = 0;
  cudaError_t                  cerr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidPointer(C,4);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckTypeName(B,MATSEQAIJCUSPARSE);
  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %D != %D",A->rmap->n,B->rmap->n);
  if (reuse == MAT_INPLACE_MATRIX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_INPLACE_MATRIX not supported");
  if (Acusp->format == MAT_CUSPARSE_ELL || Acusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (Bcusp->format == MAT_CUSPARSE_ELL || Bcusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (reuse == MAT_INITIAL_MATRIX) {
    m     = A->rmap->n;
    n     = A->cmap->n + B->cmap->n;
    ierr  = MatCreate(PETSC_COMM_SELF,C);CHKERRQ(ierr);
    ierr  = MatSetSizes(*C,m,n,m,n);CHKERRQ(ierr);
    ierr  = MatSetType(*C,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    c     = (Mat_SeqAIJ*)(*C)->data;
    Ccusp = (Mat_SeqAIJCUSPARSE*)(*C)->spptr;
    Cmat  = new Mat_SeqAIJCUSPARSEMultStruct;
    Ccsr  = new CsrMatrix;
    Cmat->cprowIndices      = NULL;
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Ccusp->workVector       = NULL;
    Ccusp->nrows    = m;
    Ccusp->mat      = Cmat;
    Ccusp->mat->mat = Ccsr;
    Ccsr->num_rows  = m;
    Ccsr->num_cols  = n;
    stat = cusparseCreateMatDescr(&Cmat->descr);CHKERRCUSPARSE(stat);
    stat = cusparseSetMatIndexBase(Cmat->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
    stat = cusparseSetMatType(Cmat->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
    cerr = cudaMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMemcpy(Cmat->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(Cmat->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(Cmat->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSECopyToGPU(B);CHKERRQ(ierr);
    if (!Acusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
    if (!Bcusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");

    Acsr = (CsrMatrix*)Acusp->mat->mat;
    Bcsr = (CsrMatrix*)Bcusp->mat->mat;
    Annz = (PetscInt)Acsr->column_indices->size();
    Bnnz = (PetscInt)Bcsr->column_indices->size();
    c->nz = Annz + Bnnz;
    Ccsr->row_offsets = new THRUSTINTARRAY32(m+1);
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values = new THRUSTARRAY(c->nz);
    Ccsr->num_entries = c->nz;
    Ccusp->cooPerm = new THRUSTINTARRAY(c->nz);
    if (c->nz) {
      auto Acoo = new THRUSTINTARRAY32(Annz);
      auto Bcoo = new THRUSTINTARRAY32(Bnnz);
      auto Ccoo = new THRUSTINTARRAY32(c->nz);
      THRUSTINTARRAY32 *Aroff,*Broff;

      if (a->compressedrow.use) { /* need full row offset */
        if (!Acusp->rowoffsets_gpu) {
          Acusp->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
          Acusp->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
          ierr = PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
        }
        Aroff = Acusp->rowoffsets_gpu;
      } else Aroff = Acsr->row_offsets;
      if (b->compressedrow.use) { /* need full row offset */
        if (!Bcusp->rowoffsets_gpu) {
          Bcusp->rowoffsets_gpu  = new THRUSTINTARRAY32(B->rmap->n + 1);
          Bcusp->rowoffsets_gpu->assign(b->i,b->i + B->rmap->n + 1);
          ierr = PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
        }
        Broff = Bcusp->rowoffsets_gpu;
      } else Broff = Bcsr->row_offsets;
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      stat = cusparseXcsr2coo(Acusp->handle,
                              Aroff->data().get(),
                              Annz,
                              m,
                              Acoo->data().get(),
                              CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
      stat = cusparseXcsr2coo(Bcusp->handle,
                              Broff->data().get(),
                              Bnnz,
                              m,
                              Bcoo->data().get(),
                              CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
      /* Issues when using bool with large matrices on SUMMIT 10.2.89 */
      auto Aperm = thrust::make_constant_iterator(1);
      auto Bperm = thrust::make_constant_iterator(0);
#if PETSC_PKG_CUDA_VERSION_GE(10,0,0)
      auto Bcib = thrust::make_transform_iterator(Bcsr->column_indices->begin(),Shift(A->cmap->n));
      auto Bcie = thrust::make_transform_iterator(Bcsr->column_indices->end(),Shift(A->cmap->n));
#else
      /* there are issues instantiating the merge operation using a transform iterator for the columns of B */
      auto Bcib = Bcsr->column_indices->begin();
      auto Bcie = Bcsr->column_indices->end();
      thrust::transform(Bcib,Bcie,Bcib,Shift(A->cmap->n));
#endif
      auto wPerm = new THRUSTINTARRAY32(Annz+Bnnz);
      auto Azb = thrust::make_zip_iterator(thrust::make_tuple(Acoo->begin(),Acsr->column_indices->begin(),Acsr->values->begin(),Aperm));
      auto Aze = thrust::make_zip_iterator(thrust::make_tuple(Acoo->end(),Acsr->column_indices->end(),Acsr->values->end(),Aperm));
      auto Bzb = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->begin(),Bcib,Bcsr->values->begin(),Bperm));
      auto Bze = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->end(),Bcie,Bcsr->values->end(),Bperm));
      auto Czb = thrust::make_zip_iterator(thrust::make_tuple(Ccoo->begin(),Ccsr->column_indices->begin(),Ccsr->values->begin(),wPerm->begin()));
      auto p1 = Ccusp->cooPerm->begin();
      auto p2 = Ccusp->cooPerm->begin();
      thrust::advance(p2,Annz);
      PetscStackCallThrust(thrust::merge(thrust::device,Azb,Aze,Bzb,Bze,Czb,IJCompare4()));
#if PETSC_PKG_CUDA_VERSION_LT(10,0,0)
      thrust::transform(Bcib,Bcie,Bcib,Shift(-A->cmap->n));
#endif
      auto cci = thrust::make_counting_iterator(zero);
      auto cce = thrust::make_counting_iterator(c->nz);
#if 0 //Errors on SUMMIT cuda 11.1.0
      PetscStackCallThrust(thrust::partition_copy(thrust::device,cci,cce,wPerm->begin(),p1,p2,thrust::identity<int>()));
#else
      auto pred = thrust::identity<int>();
      PetscStackCallThrust(thrust::copy_if(thrust::device,cci,cce,wPerm->begin(),p1,pred));
      PetscStackCallThrust(thrust::remove_copy_if(thrust::device,cci,cce,wPerm->begin(),p2,pred));
#endif
      stat = cusparseXcoo2csr(Ccusp->handle,
                              Ccoo->data().get(),
                              c->nz,
                              m,
                              Ccsr->row_offsets->data().get(),
                              CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      delete wPerm;
      delete Acoo;
      delete Bcoo;
      delete Ccoo;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      stat = cusparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, Ccsr->num_entries,
                               Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get(),
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);CHKERRCUSPARSE(stat);
#endif
      if (A->form_explicit_transpose && B->form_explicit_transpose) { /* if A and B have the transpose, generate C transpose too */
        ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(A);CHKERRQ(ierr);
        ierr = MatSeqAIJCUSPARSEFormExplicitTranspose(B);CHKERRQ(ierr);
        PetscBool AT = Acusp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bcusp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        Mat_SeqAIJCUSPARSEMultStruct *CmatT = new Mat_SeqAIJCUSPARSEMultStruct;
        CsrMatrix *CcsrT = new CsrMatrix;
        CsrMatrix *AcsrT = AT ? (CsrMatrix*)Acusp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix*)Bcusp->matTranspose->mat : NULL;

        (*C)->form_explicit_transpose = PETSC_TRUE;
        (*C)->transupdated = PETSC_TRUE;
        Ccusp->rowoffsets_gpu = NULL;
        CmatT->cprowIndices = NULL;
        CmatT->mat = CcsrT;
        CcsrT->num_rows = n;
        CcsrT->num_cols = m;
        CcsrT->num_entries = c->nz;

        CcsrT->row_offsets = new THRUSTINTARRAY32(n+1);
        CcsrT->column_indices = new THRUSTINTARRAY32(c->nz);
        CcsrT->values = new THRUSTARRAY(c->nz);

        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        auto rT = CcsrT->row_offsets->begin();
        if (AT) {
          rT = thrust::copy(AcsrT->row_offsets->begin(),AcsrT->row_offsets->end(),rT);
          thrust::advance(rT,-1);
        }
        if (BT) {
          auto titb = thrust::make_transform_iterator(BcsrT->row_offsets->begin(),Shift(a->nz));
          auto tite = thrust::make_transform_iterator(BcsrT->row_offsets->end(),Shift(a->nz));
          thrust::copy(titb,tite,rT);
        }
        auto cT = CcsrT->column_indices->begin();
        if (AT) cT = thrust::copy(AcsrT->column_indices->begin(),AcsrT->column_indices->end(),cT);
        if (BT) thrust::copy(BcsrT->column_indices->begin(),BcsrT->column_indices->end(),cT);
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(),AcsrT->values->end(),vT);
        if (BT) thrust::copy(BcsrT->values->begin(),BcsrT->values->end(),vT);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        stat = cusparseCreateMatDescr(&CmatT->descr);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatIndexBase(CmatT->descr, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSPARSE(stat);
        stat = cusparseSetMatType(CmatT->descr, CUSPARSE_MATRIX_TYPE_GENERAL);CHKERRCUSPARSE(stat);
        cerr = cudaMalloc((void **)&(CmatT->alpha_one),sizeof(PetscScalar));CHKERRCUDA(cerr);
        cerr = cudaMalloc((void **)&(CmatT->beta_zero),sizeof(PetscScalar));CHKERRCUDA(cerr);
        cerr = cudaMalloc((void **)&(CmatT->beta_one), sizeof(PetscScalar));CHKERRCUDA(cerr);
        cerr = cudaMemcpy(CmatT->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        cerr = cudaMemcpy(CmatT->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        cerr = cudaMemcpy(CmatT->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        stat = cusparseCreateCsr(&CmatT->matDescr, CcsrT->num_rows, CcsrT->num_cols, CcsrT->num_entries,
                                 CcsrT->row_offsets->data().get(), CcsrT->column_indices->data().get(), CcsrT->values->data().get(),
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);CHKERRCUSPARSE(stat);
#endif
        Ccusp->matTranspose = CmatT;
      }
    }

    c->singlemalloc = PETSC_FALSE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
    ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
    ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
    if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
      THRUSTINTARRAY ii(Ccsr->row_offsets->size());
      THRUSTINTARRAY jj(Ccsr->column_indices->size());
      ii   = *Ccsr->row_offsets;
      jj   = *Ccsr->column_indices;
      cerr = cudaMemcpy(c->i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      cerr = cudaMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    } else {
      cerr = cudaMemcpy(c->i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      cerr = cudaMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    }
    ierr = PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
    ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
    c->maxnz = c->nz;
    c->nonzerorowcnt = 0;
    c->rmax = 0;
    for (i = 0; i < m; i++) {
      const PetscInt nn = c->i[i+1] - c->i[i];
      c->ilen[i] = c->imax[i] = nn;
      c->nonzerorowcnt += (PetscInt)!!nn;
      c->rmax = PetscMax(c->rmax,nn);
    }
    ierr = MatMarkDiagonal_SeqAIJ(*C);CHKERRQ(ierr);
    ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
    (*C)->nonzerostate++;
    ierr = PetscLayoutSetUp((*C)->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*C)->cmap);CHKERRQ(ierr);
    Ccusp->nonzerostate = (*C)->nonzerostate;
    (*C)->preallocated  = PETSC_TRUE;
  } else {
    if ((*C)->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %D != %D",(*C)->rmap->n,B->rmap->n);
    c = (Mat_SeqAIJ*)(*C)->data;
    if (c->nz) {
      Ccusp = (Mat_SeqAIJCUSPARSE*)(*C)->spptr;
      if (!Ccusp->cooPerm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cooPerm");
      if (Ccusp->format == MAT_CUSPARSE_ELL || Ccusp->format == MAT_CUSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
      if (Ccusp->nonzerostate != (*C)->nonzerostate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Wrong nonzerostate");
      ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSECopyToGPU(B);CHKERRQ(ierr);
      if (!Acusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
      if (!Bcusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
      Acsr = (CsrMatrix*)Acusp->mat->mat;
      Bcsr = (CsrMatrix*)Bcusp->mat->mat;
      Ccsr = (CsrMatrix*)Ccusp->mat->mat;
      if (Acsr->num_entries != (PetscInt)Acsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"A nnz %D != %D",Acsr->num_entries,(PetscInt)Acsr->values->size());
      if (Bcsr->num_entries != (PetscInt)Bcsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"B nnz %D != %D",Bcsr->num_entries,(PetscInt)Bcsr->values->size());
      if (Ccsr->num_entries != (PetscInt)Ccsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %D != %D",Ccsr->num_entries,(PetscInt)Ccsr->values->size());
      if (Ccsr->num_entries != Acsr->num_entries + Bcsr->num_entries) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %D != %D + %D",Ccsr->num_entries,Acsr->num_entries,Bcsr->num_entries);
      if (Ccusp->cooPerm->size() != Ccsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"permSize %D != %D",(PetscInt)Ccusp->cooPerm->size(),(PetscInt)Ccsr->values->size());
      auto pmid = Ccusp->cooPerm->begin();
      thrust::advance(pmid,Acsr->num_entries);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      auto zibait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->begin(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),Ccusp->cooPerm->begin())));
      auto zieait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->end(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),pmid)));
      thrust::for_each(zibait,zieait,VecCUDAEquals());
      auto zibbit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->begin(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),pmid)));
      auto ziebit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->end(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),Ccusp->cooPerm->end())));
      thrust::for_each(zibbit,ziebit,VecCUDAEquals());
      ierr = MatSeqAIJCUSPARSEInvalidateTranspose(*C,PETSC_FALSE);CHKERRQ(ierr);
      if (A->form_explicit_transpose && B->form_explicit_transpose && (*C)->form_explicit_transpose) {
        if (!Ccusp->matTranspose) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing transpose Mat_SeqAIJCUSPARSEMultStruct");
        PetscBool AT = Acusp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bcusp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        CsrMatrix *AcsrT = AT ? (CsrMatrix*)Acusp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix*)Bcusp->matTranspose->mat : NULL;
        CsrMatrix *CcsrT = (CsrMatrix*)Ccusp->matTranspose->mat;
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(),AcsrT->values->end(),vT);
        if (BT) thrust::copy(BcsrT->values->begin(),BcsrT->values->end(),vT);
        (*C)->transupdated = PETSC_TRUE;
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectStateIncrease((PetscObject)*C);CHKERRQ(ierr);
  (*C)->assembled     = PETSC_TRUE;
  (*C)->was_assembled = PETSC_FALSE;
  (*C)->offloadmask   = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJCUSPARSE(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[])
{
  PetscErrorCode    ierr;
  bool              dmem;
  const PetscScalar *av;
  cudaError_t       cerr;

  PetscFunctionBegin;
  dmem = isCudaMem(v);
  ierr = MatSeqAIJCUSPARSEGetArrayRead(A,&av);CHKERRQ(ierr);
  if (n && idx) {
    THRUSTINTARRAY widx(n);
    widx.assign(idx,idx+n);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);

    THRUSTARRAY *w = NULL;
    thrust::device_ptr<PetscScalar> dv;
    if (dmem) {
      dv = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      dv = w->data();
    }
    thrust::device_ptr<const PetscScalar> dav = thrust::device_pointer_cast(av);

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav,widx.begin()),dv));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav,widx.end()),dv+n));
    thrust::for_each(zibit,zieit,VecCUDAEquals());
    if (w) {
      cerr = cudaMemcpy(v,w->data().get(),n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    }
    delete w;
  } else {
    cerr = cudaMemcpy(v,av,n*sizeof(PetscScalar),dmem ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  }
  if (!dmem) { ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr); }
  ierr = MatSeqAIJCUSPARSERestoreArrayRead(A,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
