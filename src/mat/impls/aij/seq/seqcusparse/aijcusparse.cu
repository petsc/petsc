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
#include <thrust/adjacent_difference.h>
#include <thrust/async/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

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

static PetscErrorCode MatSeqAIJCUSPARSECopyFromGPU(Mat);
static PetscErrorCode MatSeqAIJCUSPARSEInvalidateTranspose(Mat,PetscBool);

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJCUSPARSE(Mat,PetscInt,const PetscInt[],PetscScalar[]);
static PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat,PetscCount,const PetscInt[],const PetscInt[]);
static PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat,const PetscScalar[],InsertMode);

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
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,n,n,n,n));
  (*B)->factortype = ftype;
  PetscCall(MatSetType(*B,MATSEQAIJCUSPARSE));

  if (A->boundtocpu && A->bindingpropagates) PetscCall(MatBindToCPU(*B,PETSC_TRUE));
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    PetscCall(MatSetBlockSizesFromMats(*B,A,A));
    if (!A->boundtocpu) {
      (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSPARSE;
      (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSE;
    } else {
      (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJ;
      (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJ;
    }
    PetscCall(PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILUDT]));
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    if (!A->boundtocpu) {
      (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJCUSPARSE;
      (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJCUSPARSE;
    } else {
      (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJ;
      (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ;
    }
    PetscCall(PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSE Matrix Types");

  PetscCall(MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL));
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_cusparse));
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. MAT_CUSPARSE_MULT and MAT_CUSPARSE_ALL are currently supported.",op);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  PetscCall(PetscTryMethod(A,"MatCUSPARSESetFormat_C",(Mat,MatCUSPARSEFormatOperation,MatCUSPARSEStorageFormat),(A,op,format)));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatCUSPARSESetUseCPUSolve_SeqAIJCUSPARSE(Mat A,PetscBool use_cpu)
{
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  cusparsestruct->use_cpu_solve = use_cpu;
  PetscFunctionReturn(0);
}

/*@
   MatCUSPARSESetUseCPUSolve - Sets use CPU MatSolve.

   Input Parameters:
+  A - Matrix of type SEQAIJCUSPARSE
-  use_cpu - set flag for using the built-in CPU MatSolve

   Output Parameter:

   Notes:
   The cuSparse LU solver currently computes the factors with the built-in CPU method
   and moves the factors to the GPU for the solve. We have observed better performance keeping the data on the CPU and computing the solve there.
   This method to specify if the solve is done on the CPU or GPU (GPU is the default).

   Level: intermediate

.seealso: MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
@*/
PetscErrorCode MatCUSPARSESetUseCPUSolve(Mat A,PetscBool use_cpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  PetscCall(PetscTryMethod(A,"MatCUSPARSESetUseCPUSolve_C",(Mat,PetscBool),(A,use_cpu)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqAIJCUSPARSE(Mat A,MatOption op,PetscBool flg)
{
  PetscFunctionBegin;
  switch (op) {
    case MAT_FORM_EXPLICIT_TRANSPOSE:
      /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
      if (A->form_explicit_transpose && !flg) PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE));
      A->form_explicit_transpose = flg;
      break;
    default:
      PetscCall(MatSetOption_SeqAIJ(A,op,flg));
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
  Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyFromGPU(A));
  PetscCall(MatLUFactorNumeric_SeqAIJ(B,A,info));
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  PetscCall(ISIdentity(isrow,&row_identity));
  PetscCall(ISIdentity(iscol,&col_identity));
  if (row_identity && col_identity) {
    if (!cusparsestruct->use_cpu_solve) {
      B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;
      B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering;
    }
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    if (!cusparsestruct->use_cpu_solve) {
      B->ops->solve = MatSolve_SeqAIJCUSPARSE;
      B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE;
    }
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  if (!cusparsestruct->use_cpu_solve) {
    PetscCall(MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_SeqAIJCUSPARSE       *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"SeqAIJCUSPARSE options"));
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);PetscCall(ierr);
    if (flg) PetscCall(MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT,format));

    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV and TriSolve",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);PetscCall(ierr);
    if (flg) PetscCall(MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format));
    PetscCall(PetscOptionsBool("-mat_cusparse_use_cpu_solve","Use CPU (I)LU solve","MatCUSPARSESetUseCPUSolve",cusparsestruct->use_cpu_solve,&cusparsestruct->use_cpu_solve,&flg));
    if (flg) PetscCall(MatCUSPARSESetUseCPUSolve(A,cusparsestruct->use_cpu_solve));
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    ierr = PetscOptionsEnum("-mat_cusparse_spmv_alg","sets cuSPARSE algorithm used in sparse-mat dense-vector multiplication (SpMV)",
                            "cusparseSpMVAlg_t",MatCUSPARSESpMVAlgorithms,(PetscEnum)cusparsestruct->spmvAlg,(PetscEnum*)&cusparsestruct->spmvAlg,&flg);PetscCall(ierr);
    /* If user did use this option, check its consistency with cuSPARSE, since PetscOptionsEnum() sets enum values based on their position in MatCUSPARSESpMVAlgorithms[] */
#if PETSC_PKG_CUDA_VERSION_GE(11,2,0)
    PetscCheckFalse(flg && CUSPARSE_SPMV_CSR_ALG1 != 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");
#else
    PetscCheckFalse(flg && CUSPARSE_CSRMV_ALG1 != 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");
#endif
    ierr = PetscOptionsEnum("-mat_cusparse_spmm_alg","sets cuSPARSE algorithm used in sparse-mat dense-mat multiplication (SpMM)",
                            "cusparseSpMMAlg_t",MatCUSPARSESpMMAlgorithms,(PetscEnum)cusparsestruct->spmmAlg,(PetscEnum*)&cusparsestruct->spmmAlg,&flg);PetscCall(ierr);
    PetscCheckFalse(flg && CUSPARSE_SPMM_CSR_ALG1 != 4,PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMMAlg_t has been changed but PETSc has not been updated accordingly");

    ierr = PetscOptionsEnum("-mat_cusparse_csr2csc_alg","sets cuSPARSE algorithm used in converting CSR matrices to CSC matrices",
                            "cusparseCsr2CscAlg_t",MatCUSPARSECsr2CscAlgorithms,(PetscEnum)cusparsestruct->csr2cscAlg,(PetscEnum*)&cusparsestruct->csr2cscAlg,&flg);PetscCall(ierr);
    PetscCheckFalse(flg && CUSPARSE_CSR2CSC_ALG1 != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseCsr2CscAlg_t has been changed but PETSc has not been updated accordingly");
   #endif
  }
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors));
  PetscCall(MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors));
  PetscCall(MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors));
  PetscCall(MatICCFactorSymbolic_SeqAIJ(B,A,perm,info));
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors));
  PetscCall(MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info));
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  const PetscInt                    *ai = a->i,*aj = a->j,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiLo, *AjLo;
  PetscInt                          i,nz, nzLower, offset, rowOffset;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];
      if (!loTriFactor) {
        PetscScalar                       *AALo;

        PetscCallCUDA(cudaMallocHost((void**) &AALo, nzLower*sizeof(PetscScalar)));

        /* Allocate Space for the lower triangular matrix */
        PetscCallCUDA(cudaMallocHost((void**) &AiLo, (n+1)*sizeof(PetscInt)));
        PetscCallCUDA(cudaMallocHost((void**) &AjLo, nzLower*sizeof(PetscInt)));

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

          PetscCall(PetscArraycpy(&(AjLo[offset]), vi, nz));
          PetscCall(PetscArraycpy(&(AALo[offset]), v, nz));

          offset      += nz;
          AjLo[offset] = (PetscInt) i;
          AALo[offset] = (MatScalar) 1.0;
          offset      += 1;

          v  += nz;
          vi += nz;
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&loTriFactor));
        loTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
        /* Create the matrix description */
        PetscCallCUSPARSE(cusparseCreateMatDescr(&loTriFactor->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(loTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO));
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
       #else
        PetscCallCUSPARSE(cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
       #endif
        PetscCallCUSPARSE(cusparseSetMatFillMode(loTriFactor->descr, CUSPARSE_FILL_MODE_LOWER));
        PetscCallCUSPARSE(cusparseSetMatDiagType(loTriFactor->descr, CUSPARSE_DIAG_TYPE_UNIT));

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
        PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
        PetscCallCUSPARSE(cusparse_create_analysis_info(&loTriFactor->solveInfo));
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactor->solveOp,
                                               loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                               loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                               loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                               &loTriFactor->solveBufferSize));
        PetscCallCUDA(cudaMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize));
      #endif

        /* perform the solve analysis */
        PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, loTriFactor->solveOp,
                                         loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                         loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                         loTriFactor->csrMat->column_indices->data().get(),
                                         #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                         loTriFactor->solveInfo,
                                         loTriFactor->solvePolicy, loTriFactor->solveBuffer));
                                         #else
                                         loTriFactor->solveInfo));
                                         #endif
        PetscCallCUDA(WaitForCUDA());
        PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;
        loTriFactor->AA_h = AALo;
        PetscCallCUDA(cudaFreeHost(AiLo));
        PetscCallCUDA(cudaFreeHost(AjLo));
        PetscCall(PetscLogCpuToGpu((n+1+nzLower)*sizeof(int)+nzLower*sizeof(PetscScalar)));
      } else { /* update values only */
        if (!loTriFactor->AA_h) {
          PetscCallCUDA(cudaMallocHost((void**) &loTriFactor->AA_h, nzLower*sizeof(PetscScalar)));
        }
        /* Fill the lower triangular matrix */
        loTriFactor->AA_h[0]  = 1.0;
        v        = aa;
        vi       = aj;
        offset   = 1;
        for (i=1; i<n; i++) {
          nz = ai[i+1] - ai[i];
          PetscCall(PetscArraycpy(&(loTriFactor->AA_h[offset]), v, nz));
          offset      += nz;
          loTriFactor->AA_h[offset] = 1.0;
          offset      += 1;
          v  += nz;
        }
        loTriFactor->csrMat->values->assign(loTriFactor->AA_h, loTriFactor->AA_h+nzLower);
        PetscCall(PetscLogCpuToGpu(nzLower*sizeof(PetscScalar)));
      }
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
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
  const PetscInt                    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiUp, *AjUp;
  PetscInt                          i,nz, nzUpper, offset;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];
      if (!upTriFactor) {
        PetscScalar *AAUp;

        PetscCallCUDA(cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar)));

        /* Allocate Space for the upper triangular matrix */
        PetscCallCUDA(cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt)));
        PetscCallCUDA(cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt)));

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

          PetscCall(PetscArraycpy(&(AjUp[offset+1]), vi, nz));
          PetscCall(PetscArraycpy(&(AAUp[offset+1]), v, nz));
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&upTriFactor));
        upTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallCUSPARSE(cusparseCreateMatDescr(&upTriFactor->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(upTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO));
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
       #else
        PetscCallCUSPARSE(cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
       #endif
        PetscCallCUSPARSE(cusparseSetMatFillMode(upTriFactor->descr, CUSPARSE_FILL_MODE_UPPER));
        PetscCallCUSPARSE(cusparseSetMatDiagType(upTriFactor->descr, CUSPARSE_DIAG_TYPE_NON_UNIT));

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
        PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
        PetscCallCUSPARSE(cusparse_create_analysis_info(&upTriFactor->solveInfo));
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactor->solveOp,
                                               upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                               upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                               upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                               &upTriFactor->solveBufferSize));
        PetscCallCUDA(cudaMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize));
      #endif

        /* perform the solve analysis */
        PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, upTriFactor->solveOp,
                                         upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                         upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                         upTriFactor->csrMat->column_indices->data().get(),
                                         #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                         upTriFactor->solveInfo,
                                         upTriFactor->solvePolicy, upTriFactor->solveBuffer));
                                         #else
                                         upTriFactor->solveInfo));
                                         #endif
        PetscCallCUDA(WaitForCUDA());
        PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;
        upTriFactor->AA_h = AAUp;
        PetscCallCUDA(cudaFreeHost(AiUp));
        PetscCallCUDA(cudaFreeHost(AjUp));
        PetscCall(PetscLogCpuToGpu((n+1+nzUpper)*sizeof(int)+nzUpper*sizeof(PetscScalar)));
      } else {
        if (!upTriFactor->AA_h) {
          PetscCallCUDA(cudaMallocHost((void**) &upTriFactor->AA_h, nzUpper*sizeof(PetscScalar)));
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
          PetscCall(PetscArraycpy(&(upTriFactor->AA_h[offset+1]), v, nz));
        }
        upTriFactor->csrMat->values->assign(upTriFactor->AA_h, upTriFactor->AA_h+nzUpper);
        PetscCall(PetscLogCpuToGpu(nzUpper*sizeof(PetscScalar)));
      }
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           isrow = a->row,iscol = a->icol;
  PetscBool                    row_identity,col_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(cusparseTriFactors,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
  PetscCall(MatSeqAIJCUSPARSEBuildILULowerTriMatrix(A));
  PetscCall(MatSeqAIJCUSPARSEBuildILUUpperTriMatrix(A));

  if (!cusparseTriFactors->workVector) { cusparseTriFactors->workVector = new THRUSTARRAY(n); }
  cusparseTriFactors->nnz=a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH;
  /* lower triangular indices */
  PetscCall(ISIdentity(isrow,&row_identity));
  if (!row_identity && !cusparseTriFactors->rpermIndices) {
    const PetscInt *r;

    PetscCall(ISGetIndices(isrow,&r));
    cusparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->rpermIndices->assign(r, r+n);
    PetscCall(ISRestoreIndices(isrow,&r));
    PetscCall(PetscLogCpuToGpu(n*sizeof(PetscInt)));
  }

  /* upper triangular indices */
  PetscCall(ISIdentity(iscol,&col_identity));
  if (!col_identity && !cusparseTriFactors->cpermIndices) {
    const PetscInt *c;

    PetscCall(ISGetIndices(iscol,&c));
    cusparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->cpermIndices->assign(c, c+n);
    PetscCall(ISRestoreIndices(iscol,&c));
    PetscCall(PetscLogCpuToGpu(n*sizeof(PetscInt)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
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
      PetscCallCUDA(cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar)));
      PetscCallCUDA(cudaMallocHost((void**) &AALo, nzUpper*sizeof(PetscScalar)));
      if (!upTriFactor && !loTriFactor) {
        /* Allocate Space for the upper triangular matrix */
        PetscCallCUDA(cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt)));
        PetscCallCUDA(cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt)));

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
            PetscCall(PetscArraycpy(&(AjUp[offset]), vj, nz));
            PetscCall(PetscArraycpy(&(AAUp[offset]), v, nz));
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&upTriFactor));
        upTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallCUSPARSE(cusparseCreateMatDescr(&upTriFactor->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(upTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO));
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
       #else
        PetscCallCUSPARSE(cusparseSetMatType(upTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
       #endif
        PetscCallCUSPARSE(cusparseSetMatFillMode(upTriFactor->descr, CUSPARSE_FILL_MODE_UPPER));
        PetscCallCUSPARSE(cusparseSetMatDiagType(upTriFactor->descr, CUSPARSE_DIAG_TYPE_UNIT));

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
        PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
        PetscCallCUSPARSE(cusparse_create_analysis_info(&upTriFactor->solveInfo));
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactor->solveOp,
                                               upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                               upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                               upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                               &upTriFactor->solveBufferSize));
        PetscCallCUDA(cudaMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize));
      #endif

        /* perform the solve analysis */
        PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, upTriFactor->solveOp,
                                         upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                         upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                         upTriFactor->csrMat->column_indices->data().get(),
                                         #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                         upTriFactor->solveInfo,
                                         upTriFactor->solvePolicy, upTriFactor->solveBuffer));
                                         #else
                                         upTriFactor->solveInfo));
                                         #endif
        PetscCallCUDA(WaitForCUDA());
        PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;

        /* allocate space for the triangular factor information */
        PetscCall(PetscNew(&loTriFactor));
        loTriFactor->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        PetscCallCUSPARSE(cusparseCreateMatDescr(&loTriFactor->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(loTriFactor->descr, CUSPARSE_INDEX_BASE_ZERO));
       #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
       #else
        PetscCallCUSPARSE(cusparseSetMatType(loTriFactor->descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
       #endif
        PetscCallCUSPARSE(cusparseSetMatFillMode(loTriFactor->descr, CUSPARSE_FILL_MODE_UPPER));
        PetscCallCUSPARSE(cusparseSetMatDiagType(loTriFactor->descr, CUSPARSE_DIAG_TYPE_NON_UNIT));

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
        PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
        PetscCallCUSPARSE(cusparse_create_analysis_info(&loTriFactor->solveInfo));
      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
        PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactor->solveOp,
                                               loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                               loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                               loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                               &loTriFactor->solveBufferSize));
        PetscCallCUDA(cudaMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize));
      #endif

        /* perform the solve analysis */
        PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, loTriFactor->solveOp,
                                         loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                         loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                         loTriFactor->csrMat->column_indices->data().get(),
                                         #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                         loTriFactor->solveInfo,
                                         loTriFactor->solvePolicy, loTriFactor->solveBuffer));
                                         #else
                                         loTriFactor->solveInfo));
                                         #endif
        PetscCallCUDA(WaitForCUDA());
        PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

        /* assign the pointer */
        ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;

        PetscCall(PetscLogCpuToGpu(2*(((A->rmap->n+1)+(a->nz))*sizeof(int)+(a->nz)*sizeof(PetscScalar))));
        PetscCallCUDA(cudaFreeHost(AiUp));
        PetscCallCUDA(cudaFreeHost(AjUp));
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
            PetscCall(PetscArraycpy(&(AAUp[offset]), v, nz));
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }
        PetscCheck(upTriFactor,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
        PetscCheck(loTriFactor,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
        upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);
        PetscCall(PetscLogCpuToGpu(2*(a->nz)*sizeof(PetscScalar)));
      }
      PetscCallCUDA(cudaFreeHost(AAUp));
      PetscCallCUDA(cudaFreeHost(AALo));
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           ip = a->row;
  PetscBool                    perm_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(cusparseTriFactors,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
  PetscCall(MatSeqAIJCUSPARSEBuildICCTriMatrices(A));
  if (!cusparseTriFactors->workVector) { cusparseTriFactors->workVector = new THRUSTARRAY(n); }
  cusparseTriFactors->nnz=(a->nz-n)*2 + n;

  A->offloadmask = PETSC_OFFLOAD_BOTH;

  /* lower triangular indices */
  PetscCall(ISIdentity(ip,&perm_identity));
  if (!perm_identity) {
    IS             iip;
    const PetscInt *irip,*rip;

    PetscCall(ISInvertPermutation(ip,PETSC_DECIDE,&iip));
    PetscCall(ISGetIndices(iip,&irip));
    PetscCall(ISGetIndices(ip,&rip));
    cusparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->rpermIndices->assign(rip, rip+n);
    cusparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->cpermIndices->assign(irip, irip+n);
    PetscCall(ISRestoreIndices(iip,&irip));
    PetscCall(ISDestroy(&iip));
    PetscCall(ISRestoreIndices(ip,&rip));
    PetscCall(PetscLogCpuToGpu(2.*n*sizeof(PetscInt)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyFromGPU(A));
  PetscCall(MatCholeskyFactorNumeric_SeqAIJ(B,A,info));
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  PetscCall(ISIdentity(ip,&perm_identity));
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
  PetscCall(MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJCUSPARSETriFactors      *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactorStruct *loTriFactorT;
  Mat_SeqAIJCUSPARSETriFactorStruct *upTriFactorT;
  cusparseIndexBase_t               indexBase;
  cusparseMatrixType_t              matrixType;
  cusparseFillMode_t                fillMode;
  cusparseDiagType_t                diagType;

  PetscFunctionBegin;
  /* allocate space for the transpose of the lower triangular factor */
  PetscCall(PetscNew(&loTriFactorT));
  loTriFactorT->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the lower triangular factor */
  matrixType = cusparseGetMatType(loTriFactor->descr);
  indexBase = cusparseGetMatIndexBase(loTriFactor->descr);
  fillMode = cusparseGetMatFillMode(loTriFactor->descr)==CUSPARSE_FILL_MODE_UPPER ?
    CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
  diagType = cusparseGetMatDiagType(loTriFactor->descr);

  /* Create the matrix description */
  PetscCallCUSPARSE(cusparseCreateMatDescr(&loTriFactorT->descr));
  PetscCallCUSPARSE(cusparseSetMatIndexBase(loTriFactorT->descr, indexBase));
  PetscCallCUSPARSE(cusparseSetMatType(loTriFactorT->descr, matrixType));
  PetscCallCUSPARSE(cusparseSetMatFillMode(loTriFactorT->descr, fillMode));
  PetscCallCUSPARSE(cusparseSetMatDiagType(loTriFactorT->descr, diagType));

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
  PetscCallCUSPARSE(cusparseCsr2cscEx2_bufferSize(cusparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                                               loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                                               loTriFactor->csrMat->values->data().get(),
                                               loTriFactor->csrMat->row_offsets->data().get(),
                                               loTriFactor->csrMat->column_indices->data().get(),
                                               loTriFactorT->csrMat->values->data().get(),
                                               loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                               CUSPARSE_ACTION_NUMERIC,indexBase,
                                               CUSPARSE_CSR2CSC_ALG1, &loTriFactor->csr2cscBufferSize));
  PetscCallCUDA(cudaMalloc(&loTriFactor->csr2cscBuffer,loTriFactor->csr2cscBufferSize));
#endif

  PetscCall(PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0));
  PetscCallCUSPARSE(cusparse_csr2csc(cusparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                                  loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                                  loTriFactor->csrMat->values->data().get(),
                                  loTriFactor->csrMat->row_offsets->data().get(),
                                  loTriFactor->csrMat->column_indices->data().get(),
                                  loTriFactorT->csrMat->values->data().get(),
                                  #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                                  loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                  CUSPARSE_ACTION_NUMERIC, indexBase,
                                  CUSPARSE_CSR2CSC_ALG1, loTriFactor->csr2cscBuffer));
                                  #else
                                  loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                  CUSPARSE_ACTION_NUMERIC, indexBase));
                                  #endif
  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0));

  /* Create the solve analysis information */
  PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
  PetscCallCUSPARSE(cusparse_create_analysis_info(&loTriFactorT->solveInfo));
#if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, loTriFactorT->solveOp,
                                         loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                                         loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                         loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo,
                                         &loTriFactorT->solveBufferSize));
  PetscCallCUDA(cudaMalloc(&loTriFactorT->solveBuffer,loTriFactorT->solveBufferSize));
#endif

  /* perform the solve analysis */
  PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, loTriFactorT->solveOp,
                                   loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                                   loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                   loTriFactorT->csrMat->column_indices->data().get(),
                                   #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                   loTriFactorT->solveInfo,
                                   loTriFactorT->solvePolicy, loTriFactorT->solveBuffer));
                                   #else
                                   loTriFactorT->solveInfo));
                                   #endif
  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtrTranspose = loTriFactorT;

  /*********************************************/
  /* Now the Transpose of the Upper Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the upper triangular factor */
  PetscCall(PetscNew(&upTriFactorT));
  upTriFactorT->solvePolicy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the upper triangular factor */
  matrixType = cusparseGetMatType(upTriFactor->descr);
  indexBase = cusparseGetMatIndexBase(upTriFactor->descr);
  fillMode = cusparseGetMatFillMode(upTriFactor->descr)==CUSPARSE_FILL_MODE_UPPER ?
    CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
  diagType = cusparseGetMatDiagType(upTriFactor->descr);

  /* Create the matrix description */
  PetscCallCUSPARSE(cusparseCreateMatDescr(&upTriFactorT->descr));
  PetscCallCUSPARSE(cusparseSetMatIndexBase(upTriFactorT->descr, indexBase));
  PetscCallCUSPARSE(cusparseSetMatType(upTriFactorT->descr, matrixType));
  PetscCallCUSPARSE(cusparseSetMatFillMode(upTriFactorT->descr, fillMode));
  PetscCallCUSPARSE(cusparseSetMatDiagType(upTriFactorT->descr, diagType));

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
  PetscCallCUSPARSE(cusparseCsr2cscEx2_bufferSize(cusparseTriFactors->handle,upTriFactor->csrMat->num_rows,
                                               upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                                               upTriFactor->csrMat->values->data().get(),
                                               upTriFactor->csrMat->row_offsets->data().get(),
                                               upTriFactor->csrMat->column_indices->data().get(),
                                               upTriFactorT->csrMat->values->data().get(),
                                               upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                               CUSPARSE_ACTION_NUMERIC,indexBase,
                                               CUSPARSE_CSR2CSC_ALG1, &upTriFactor->csr2cscBufferSize));
  PetscCallCUDA(cudaMalloc(&upTriFactor->csr2cscBuffer,upTriFactor->csr2cscBufferSize));
#endif

  PetscCall(PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0));
  PetscCallCUSPARSE(cusparse_csr2csc(cusparseTriFactors->handle, upTriFactor->csrMat->num_rows,
                                  upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                                  upTriFactor->csrMat->values->data().get(),
                                  upTriFactor->csrMat->row_offsets->data().get(),
                                  upTriFactor->csrMat->column_indices->data().get(),
                                  upTriFactorT->csrMat->values->data().get(),
                                  #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                                  upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(), cusparse_scalartype,
                                  CUSPARSE_ACTION_NUMERIC, indexBase,
                                  CUSPARSE_CSR2CSC_ALG1, upTriFactor->csr2cscBuffer));
                                  #else
                                  upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                 CUSPARSE_ACTION_NUMERIC, indexBase));
                                 #endif

  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0));

  /* Create the solve analysis information */
  PetscCall(PetscLogEventBegin(MAT_CUSPARSESolveAnalysis,A,0,0,0));
  PetscCallCUSPARSE(cusparse_create_analysis_info(&upTriFactorT->solveInfo));
  #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
  PetscCallCUSPARSE(cusparse_get_svbuffsize(cusparseTriFactors->handle, upTriFactorT->solveOp,
                                         upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                                         upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                         upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo,
                                         &upTriFactorT->solveBufferSize));
  PetscCallCUDA(cudaMalloc(&upTriFactorT->solveBuffer,upTriFactorT->solveBufferSize));
  #endif

  /* perform the solve analysis */
  /* christ, would it have killed you to put this stuff in a function????????? */
  PetscCallCUSPARSE(cusparse_analysis(cusparseTriFactors->handle, upTriFactorT->solveOp,
                                   upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                                   upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                   upTriFactorT->csrMat->column_indices->data().get(),
                                   #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                   upTriFactorT->solveInfo,
                                   upTriFactorT->solvePolicy, upTriFactorT->solveBuffer));
                                   #else
                                   upTriFactorT->solveInfo));
                                   #endif

  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0));

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

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
  PetscCheck(matstruct,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing mat struct");
  matstructT = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->matTranspose;
  PetscCheckFalse(A->transupdated && !matstructT,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing matTranspose struct");
  if (A->transupdated) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0));
  PetscCall(PetscLogGpuTimeBegin());
  if (cusparsestruct->format != MAT_CUSPARSE_CSR) {
    PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE));
  }
  if (!cusparsestruct->matTranspose) { /* create cusparse matrix */
    matstructT = new Mat_SeqAIJCUSPARSEMultStruct;
    PetscCallCUSPARSE(cusparseCreateMatDescr(&matstructT->descr));
    indexBase = cusparseGetMatIndexBase(matstruct->descr);
    PetscCallCUSPARSE(cusparseSetMatIndexBase(matstructT->descr, indexBase));
    PetscCallCUSPARSE(cusparseSetMatType(matstructT->descr, CUSPARSE_MATRIX_TYPE_GENERAL));

    /* set alpha and beta */
    PetscCallCUDA(cudaMalloc((void **)&(matstructT->alpha_one),sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void **)&(matstructT->beta_zero),sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void **)&(matstructT->beta_one), sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(matstructT->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(matstructT->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(matstructT->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));

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
                               indexBase,cusparse_scalartype);PetscCallCUSPARSE(stat);
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
                                 indexBase,cusparse_scalartype);PetscCallCUSPARSE(stat);

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
                              temp->column_indices->data().get());PetscCallCUSPARSE(stat);

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
                              CUSPARSE_ACTION_NUMERIC, indexBase);PetscCallCUSPARSE(stat);

      /* Last, convert CSC to HYB */
      cusparseHybMat_t hybMat;
      PetscCallCUSPARSE(cusparseCreateHybMat(&hybMat));
      cusparseHybPartition_t partition = cusparsestruct->format==MAT_CUSPARSE_ELL ?
        CUSPARSE_HYB_PARTITION_MAX : CUSPARSE_HYB_PARTITION_AUTO;
      stat = cusparse_csr2hyb(cusparsestruct->handle, A->rmap->n, A->cmap->n,
                              matstructT->descr, tempT->values->data().get(),
                              tempT->row_offsets->data().get(),
                              tempT->column_indices->data().get(),
                              hybMat, 0, partition);PetscCallCUSPARSE(stat);

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
    PetscCheck(matrix,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix");
    PetscCheck(matrix->row_offsets,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix rows");
    PetscCheck(matrix->column_indices,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix cols");
    PetscCheck(matrix->values,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrix values");
    PetscCheck(matrixT,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT");
    PetscCheck(matrixT->row_offsets,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT rows");
    PetscCheck(matrixT->column_indices,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT cols");
    PetscCheck(matrixT->values,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CsrMatrixT values");
    if (!cusparsestruct->rowoffsets_gpu) { /* this may be absent when we did not construct the transpose with csr2csc */
      cusparsestruct->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
      cusparsestruct->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
      PetscCall(PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt)));
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
                                           cusparsestruct->csr2cscAlg, &csr2cscBufferSize);PetscCallCUSPARSE(stat);
      PetscCallCUDA(cudaMalloc(&csr2cscBuffer,csr2cscBufferSize));
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
                              cusparsestruct->csr2cscAlg, csr2cscBuffer);PetscCallCUSPARSE(stat);
                             #else
                              matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(),
                              CUSPARSE_ACTION_NUMERIC, indexBase);PetscCallCUSPARSE(stat);
                             #endif
      } else {
        matrixT->row_offsets->assign(matrixT->row_offsets->size(),indexBase);
      }

      cusparsestruct->csr2csc_i = new THRUSTINTARRAY(matrix->num_entries);
      PetscStackCallThrust(thrust::transform(thrust::device,matrixT->values->begin(),matrixT->values->end(),cusparsestruct->csr2csc_i->begin(),PetscScalarToPetscInt()));
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      PetscCallCUDA(cudaFree(csr2cscBuffer));
     #endif
    }
    PetscStackCallThrust(thrust::copy(thrust::device,thrust::make_permutation_iterator(matrix->values->begin(), cusparsestruct->csr2csc_i->begin()),
                                                     thrust::make_permutation_iterator(matrix->values->begin(), cusparsestruct->csr2csc_i->end()),
                                                     matrixT->values->begin()));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(MAT_CUSPARSEGenerateTranspose,A,0,0,0));
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

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    PetscCall(MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A));
    loTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  PetscCall(VecCUDAGetArrayWrite(xx,&xarray));
  PetscCall(VecCUDAGetArrayRead(bb,&barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  PetscCall(PetscLogGpuTimeBegin());
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
                        upTriFactorT->solvePolicy, upTriFactorT->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        tempGPU->data().get());PetscCallCUSPARSE(stat);
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
                        loTriFactorT->solvePolicy, loTriFactorT->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                         xarray);PetscCallCUSPARSE(stat);
                      #endif

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(xGPU, cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(xGPU+n, cusparseTriFactors->cpermIndices->end()),
               tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  PetscCall(VecCUDARestoreArrayRead(bb,&barray));
  PetscCall(VecCUDARestoreArrayWrite(xx,&xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n));
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

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    PetscCall(MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A));
    loTriFactorT       = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT       = (Mat_SeqAIJCUSPARSETriFactorStruct*)cusparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  PetscCall(VecCUDAGetArrayWrite(xx,&xarray));
  PetscCall(VecCUDAGetArrayRead(bb,&barray));

  PetscCall(PetscLogGpuTimeBegin());
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
                        upTriFactorT->solvePolicy, upTriFactorT->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        tempGPU->data().get());PetscCallCUSPARSE(stat);
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
                        loTriFactorT->solvePolicy, loTriFactorT->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        xarray);PetscCallCUSPARSE(stat);
                      #endif

  /* restore */
  PetscCall(VecCUDARestoreArrayRead(bb,&barray));
  PetscCall(VecCUDARestoreArrayWrite(xx,&xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n));
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

  PetscFunctionBegin;

  /* Get the GPU pointers */
  PetscCall(VecCUDAGetArrayWrite(xx,&xarray));
  PetscCall(VecCUDAGetArrayRead(bb,&barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  PetscCall(PetscLogGpuTimeBegin());
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
                         loTriFactor->solvePolicy, loTriFactor->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                         xarray);PetscCallCUSPARSE(stat);
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
                        upTriFactor->solvePolicy, upTriFactor->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        tempGPU->data().get());PetscCallCUSPARSE(stat);
                      #endif

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->end()),
               xGPU);

  PetscCall(VecCUDARestoreArrayRead(bb,&barray));
  PetscCall(VecCUDARestoreArrayWrite(xx,&xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n));
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

  PetscFunctionBegin;
  /* Get the GPU pointers */
  PetscCall(VecCUDAGetArrayWrite(xx,&xarray));
  PetscCall(VecCUDAGetArrayRead(bb,&barray));

  PetscCall(PetscLogGpuTimeBegin());
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
                        loTriFactor->solvePolicy,loTriFactor->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        tempGPU->data().get());PetscCallCUSPARSE(stat);
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
                        upTriFactor->solvePolicy, upTriFactor->solveBuffer);PetscCallCUSPARSE(stat);
                      #else
                        xarray);PetscCallCUSPARSE(stat);
                      #endif

  PetscCall(VecCUDARestoreArrayRead(bb,&barray));
  PetscCall(VecCUDARestoreArrayWrite(xx,&xarray));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSECopyFromGPU(Mat A)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    CsrMatrix *matrix = (CsrMatrix*)cusp->mat->mat;

    PetscCall(PetscLogEventBegin(MAT_CUSPARSECopyFromGPU,A,0,0,0));
    PetscCallCUDA(cudaMemcpy(a->a, matrix->values->data().get(), a->nz*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscCallCUDA(WaitForCUDA());
    PetscCall(PetscLogGpuToCpu(a->nz*sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(MAT_CUSPARSECopyFromGPU,A,0,0,0));
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJCUSPARSE(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyFromGPU(A));
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJCUSPARSE(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJCUSPARSE(Mat A,const PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyFromGPU(A));
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJCUSPARSE(Mat A,const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJCUSPARSE(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJCUSPARSE(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetCSRAndMemType_SeqAIJCUSPARSE(Mat A,const PetscInt **i,const PetscInt **j,PetscScalar **a,PetscMemType *mtype)
{
  Mat_SeqAIJCUSPARSE           *cusp;
  CsrMatrix                    *matrix;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  cusp = static_cast<Mat_SeqAIJCUSPARSE*>(A->spptr);
  PetscCheck(cusp != NULL,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"cusp is NULL");
  matrix = (CsrMatrix*)cusp->mat->mat;

  if (i) {
   #if !defined(PETSC_USE_64BIT_INDICES)
    *i = matrix->row_offsets->data().get();
   #else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSparse does not supported 64-bit indices");
   #endif
  }
  if (j) {
   #if !defined(PETSC_USE_64BIT_INDICES)
    *j = matrix->column_indices->data().get();
   #else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSparse does not supported 64-bit indices");
   #endif
  }
  if (a) *a = matrix->values->data().get();
  if (mtype) *mtype = PETSC_MEMTYPE_CUDA;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct = cusparsestruct->mat;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  PetscInt                     m = A->rmap->n,*ii,*ridx,tmp;
  cusparseStatus_t             stat;
  PetscBool                    both = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCheck(!A->boundtocpu,PETSC_COMM_SELF,PETSC_ERR_GPU,"Cannot copy to GPU");
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (A->nonzerostate == cusparsestruct->nonzerostate && cusparsestruct->format == MAT_CUSPARSE_CSR) { /* Copy values only */
      CsrMatrix *matrix;
      matrix = (CsrMatrix*)cusparsestruct->mat->mat;

      PetscCheckFalse(a->nz && !a->a,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR values");
      PetscCall(PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0));
      matrix->values->assign(a->a, a->a+a->nz);
      PetscCallCUDA(WaitForCUDA());
      PetscCall(PetscLogCpuToGpu((a->nz)*sizeof(PetscScalar)));
      PetscCall(PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0));
      PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE));
    } else {
      PetscInt nnz;
      PetscCall(PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0));
      PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&cusparsestruct->mat,cusparsestruct->format));
      PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE));
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
        PetscCheckFalse(!ii,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR row data");
        if (!a->a) { nnz = ii[m]; both = PETSC_FALSE; }
        else nnz = a->nz;
        PetscCheckFalse(nnz && !a->j,PETSC_COMM_SELF,PETSC_ERR_GPU,"Missing CSR column data");

        /* create cusparse matrix */
        cusparsestruct->nrows = m;
        matstruct = new Mat_SeqAIJCUSPARSEMultStruct;
        PetscCallCUSPARSE(cusparseCreateMatDescr(&matstruct->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(matstruct->descr, CUSPARSE_INDEX_BASE_ZERO));
        PetscCallCUSPARSE(cusparseSetMatType(matstruct->descr, CUSPARSE_MATRIX_TYPE_GENERAL));

        PetscCallCUDA(cudaMalloc((void **)&(matstruct->alpha_one),sizeof(PetscScalar)));
        PetscCallCUDA(cudaMalloc((void **)&(matstruct->beta_zero),sizeof(PetscScalar)));
        PetscCallCUDA(cudaMalloc((void **)&(matstruct->beta_one), sizeof(PetscScalar)));
        PetscCallCUDA(cudaMemcpy(matstruct->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
        PetscCallCUDA(cudaMemcpy(matstruct->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice));
        PetscCallCUDA(cudaMemcpy(matstruct->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
        PetscCallCUSPARSE(cusparseSetPointerMode(cusparsestruct->handle, CUSPARSE_POINTER_MODE_DEVICE));

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
                                    CUSPARSE_INDEX_BASE_ZERO,cusparse_scalartype);PetscCallCUSPARSE(stat);
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
          PetscCallCUSPARSE(cusparseCreateHybMat(&hybMat));
          cusparseHybPartition_t partition = cusparsestruct->format==MAT_CUSPARSE_ELL ?
            CUSPARSE_HYB_PARTITION_MAX : CUSPARSE_HYB_PARTITION_AUTO;
          stat = cusparse_csr2hyb(cusparsestruct->handle, mat->num_rows, mat->num_cols,
              matstruct->descr, mat->values->data().get(),
              mat->row_offsets->data().get(),
              mat->column_indices->data().get(),
              hybMat, 0, partition);PetscCallCUSPARSE(stat);
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
        PetscCall(PetscLogCpuToGpu(((m+1)+(a->nz))*sizeof(int)+tmp*sizeof(PetscInt)+(3+(a->nz))*sizeof(PetscScalar)));

        /* assign the pointer */
        cusparsestruct->mat = matstruct;
      } catch(char *ex) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
      }
      PetscCallCUDA(WaitForCUDA());
      PetscCall(PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0));
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
  MatMatCusparse *mmdata = (MatMatCusparse *)data;

  PetscFunctionBegin;
  PetscCallCUDA(cudaFree(mmdata->Bt));
  delete mmdata->Bcsr;
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  if (mmdata->matSpBDescr) PetscCallCUSPARSE(cusparseDestroySpMat(mmdata->matSpBDescr));
  if (mmdata->matBDescr)   PetscCallCUSPARSE(cusparseDestroyDnMat(mmdata->matBDescr));
  if (mmdata->matCDescr)   PetscCallCUSPARSE(cusparseDestroyDnMat(mmdata->matCDescr));
  if (mmdata->spgemmDesc)  PetscCallCUSPARSE(cusparseSpGEMM_destroyDescr(mmdata->spgemmDesc));
 #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
  if (mmdata->dBuffer4)  PetscCallCUDA(cudaFree(mmdata->dBuffer4));
  if (mmdata->dBuffer5)  PetscCallCUDA(cudaFree(mmdata->dBuffer5));
 #endif
  if (mmdata->mmBuffer)  PetscCallCUDA(cudaFree(mmdata->mmBuffer));
  if (mmdata->mmBuffer2) PetscCallCUDA(cudaFree(mmdata->mmBuffer2));
 #endif
  PetscCall(MatDestroy(&mmdata->X));
  PetscCall(PetscFree(data));
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
  MatMatCusparse               *mmdata;
  Mat_SeqAIJCUSPARSEMultStruct *mat;
  CsrMatrix                    *csrmat;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data empty");
  mmdata = (MatMatCusparse*)product->data;
  A    = product->A;
  B    = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  /* currently CopyToGpu does not copy if the matrix is bound to CPU
     Instead of silently accepting the wrong answer, I prefer to raise the error */
  PetscCheck(!A->boundtocpu,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
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
      PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(A));
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
    SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  PetscCheck(mat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csrmat = (CsrMatrix*)mat->mat;
  /* if the user passed a CPU matrix, copy the data to the GPU */
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQDENSECUDA,&biscuda));
  if (!biscuda) PetscCall(MatConvert(B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&B));
  PetscCall(MatDenseCUDAGetArrayRead(B,&barray));

  PetscCall(MatDenseGetLDA(B,&blda));
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    PetscCall(MatDenseCUDAGetArrayWrite(mmdata->X,&carray));
    PetscCall(MatDenseGetLDA(mmdata->X,&clda));
  } else {
    PetscCall(MatDenseCUDAGetArrayWrite(C,&carray));
    PetscCall(MatDenseGetLDA(C,&clda));
  }

  PetscCall(PetscLogGpuTimeBegin());
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseOperation_t opB = (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  /* (re)allocate mmBuffer if not initialized or LDAs are different */
  if (!mmdata->initialized || mmdata->Blda != blda || mmdata->Clda != clda) {
    size_t mmBufferSize;
    if (mmdata->initialized && mmdata->Blda != blda) {PetscCallCUSPARSE(cusparseDestroyDnMat(mmdata->matBDescr)); mmdata->matBDescr = NULL;}
    if (!mmdata->matBDescr) {
      PetscCallCUSPARSE(cusparseCreateDnMat(&mmdata->matBDescr,B->rmap->n,B->cmap->n,blda,(void*)barray,cusparse_scalartype,CUSPARSE_ORDER_COL));
      mmdata->Blda = blda;
    }

    if (mmdata->initialized && mmdata->Clda != clda) {PetscCallCUSPARSE(cusparseDestroyDnMat(mmdata->matCDescr)); mmdata->matCDescr = NULL;}
    if (!mmdata->matCDescr) { /* matCDescr is for C or mmdata->X */
      PetscCallCUSPARSE(cusparseCreateDnMat(&mmdata->matCDescr,m,n,clda,(void*)carray,cusparse_scalartype,CUSPARSE_ORDER_COL));
      mmdata->Clda = clda;
    }

    if (!mat->matDescr) {
      stat = cusparseCreateCsr(&mat->matDescr,
                               csrmat->num_rows, csrmat->num_cols, csrmat->num_entries,
                               csrmat->row_offsets->data().get(), csrmat->column_indices->data().get(),
                               csrmat->values->data().get(),
                               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, /* row offset, col idx types due to THRUSTINTARRAY32 */
                               CUSPARSE_INDEX_BASE_ZERO,cusparse_scalartype);PetscCallCUSPARSE(stat);
    }
    stat = cusparseSpMM_bufferSize(cusp->handle,opA,opB,mat->alpha_one,
                                   mat->matDescr,mmdata->matBDescr,mat->beta_zero,
                                   mmdata->matCDescr,cusparse_scalartype,
                                   cusp->spmmAlg,&mmBufferSize);PetscCallCUSPARSE(stat);
    if ((mmdata->mmBuffer && mmdata->mmBufferSize < mmBufferSize) || !mmdata->mmBuffer) {
      PetscCallCUDA(cudaFree(mmdata->mmBuffer));
      PetscCallCUDA(cudaMalloc(&mmdata->mmBuffer,mmBufferSize));
      mmdata->mmBufferSize = mmBufferSize;
    }
    mmdata->initialized = PETSC_TRUE;
  } else {
    /* to be safe, always update pointers of the mats */
    PetscCallCUSPARSE(cusparseSpMatSetValues(mat->matDescr,csrmat->values->data().get()));
    PetscCallCUSPARSE(cusparseDnMatSetValues(mmdata->matBDescr,(void*)barray));
    PetscCallCUSPARSE(cusparseDnMatSetValues(mmdata->matCDescr,(void*)carray));
  }

  /* do cusparseSpMM, which supports transpose on B */
  stat = cusparseSpMM(cusp->handle,opA,opB,mat->alpha_one,
                      mat->matDescr,mmdata->matBDescr,mat->beta_zero,
                      mmdata->matCDescr,cusparse_scalartype,
                      cusp->spmmAlg,mmdata->mmBuffer);PetscCallCUSPARSE(stat);
 #else
  PetscInt k;
  /* cusparseXcsrmm does not support transpose on B */
  if (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) {
    cublasHandle_t cublasv2handle;
    cublasStatus_t cerr;

    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    cerr = cublasXgeam(cublasv2handle,CUBLAS_OP_T,CUBLAS_OP_T,
                       B->cmap->n,B->rmap->n,
                       &PETSC_CUSPARSE_ONE ,barray,blda,
                       &PETSC_CUSPARSE_ZERO,barray,blda,
                       mmdata->Bt,B->cmap->n);PetscCallCUBLAS(cerr);
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
                           carray,clda);PetscCallCUSPARSE(stat);
 #endif
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(n*2.0*csrmat->num_entries));
  PetscCall(MatDenseCUDARestoreArrayRead(B,&barray));
  if (product->type == MATPRODUCT_RARt) {
    PetscCall(MatDenseCUDARestoreArrayWrite(mmdata->X,&carray));
    PetscCall(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(B,mmdata->X,C,PETSC_FALSE,PETSC_FALSE));
  } else if (product->type == MATPRODUCT_PtAP) {
    PetscCall(MatDenseCUDARestoreArrayWrite(mmdata->X,&carray));
    PetscCall(MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(B,mmdata->X,C,PETSC_TRUE,PETSC_FALSE));
  } else {
    PetscCall(MatDenseCUDARestoreArrayWrite(C,&carray));
  }
  if (mmdata->cisdense) {
    PetscCall(MatConvert(C,MATSEQDENSE,MAT_INPLACE_MATRIX,&C));
  }
  if (!biscuda) {
    PetscCall(MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJCUSPARSE_SeqDENSECUDA(Mat C)
{
  Mat_Product        *product = C->product;
  Mat                A,B;
  PetscInt           m,n;
  PetscBool          cisdense,flg;
  MatMatCusparse     *mmdata;
  Mat_SeqAIJCUSPARSE *cusp;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data not empty");
  A    = product->A;
  B    = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  PetscCheckFalse(cusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
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
    SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  PetscCall(MatSetSizes(C,m,n,m,n));
  /* if C is of type MATSEQDENSE (CPU), perform the operation on the GPU and then copy on the CPU */
  PetscCall(PetscObjectTypeCompare((PetscObject)C,MATSEQDENSE,&cisdense));
  PetscCall(MatSetType(C,MATSEQDENSECUDA));

  /* product data */
  PetscCall(PetscNew(&mmdata));
  mmdata->cisdense = cisdense;
 #if PETSC_PKG_CUDA_VERSION_LT(11,0,0)
  /* cusparseXcsrmm does not support transpose on B, so we allocate buffer to store B^T */
  if (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) {
    PetscCallCUDA(cudaMalloc((void**)&mmdata->Bt,(size_t)B->rmap->n*(size_t)B->cmap->n*sizeof(PetscScalar)));
  }
 #endif
  /* for these products we need intermediate storage */
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)C),&mmdata->X));
    PetscCall(MatSetType(mmdata->X,MATSEQDENSECUDA));
    if (product->type == MATPRODUCT_RARt) { /* do not preallocate, since the first call to MatDenseCUDAGetArray will preallocate on the GPU for us */
      PetscCall(MatSetSizes(mmdata->X,A->rmap->n,B->rmap->n,A->rmap->n,B->rmap->n));
    } else {
      PetscCall(MatSetSizes(mmdata->X,A->rmap->n,B->cmap->n,A->rmap->n,B->cmap->n));
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
  cusparseStatus_t             stat;
  MatProductType               ptype;
  MatMatCusparse               *mmdata;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseSpMatDescr_t         BmatSpDescr;
#endif
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE,opB = CUSPARSE_OPERATION_NON_TRANSPOSE; /* cuSPARSE spgemm doesn't support transpose yet */

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data empty");
  PetscCall(PetscObjectTypeCompare((PetscObject)C,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for C of type %s",((PetscObject)C)->type_name);
  mmdata = (MatMatCusparse*)C->product->data;
  A = product->A;
  B = product->B;
  if (mmdata->reusesym) { /* this happens when api_user is true, meaning that the matrix values have been already computed in the MatProductSymbolic phase */
    mmdata->reusesym = PETSC_FALSE;
    Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
    PetscCheckFalse(Ccusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
    Cmat = Ccusp->mat;
    PetscCheck(Cmat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C mult struct for product type %s",MatProductTypes[C->product->type]);
    Ccsr = (CsrMatrix*)Cmat->mat;
    PetscCheck(Ccsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C CSR struct");
    goto finalize;
  }
  if (!c->nz) goto finalize;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for B of type %s",((PetscObject)B)->type_name);
  PetscCheck(!A->boundtocpu,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  PetscCheck(!B->boundtocpu,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr;
  Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
  PetscCheckFalse(Acusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  PetscCheckFalse(Bcusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  PetscCheckFalse(Ccusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(B));

  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) {
    ptype = MATPRODUCT_AB;
    PetscCheck(product->symbolic_used_the_fact_A_is_symmetric,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Symbolic should have been built using the fact that A is symmetric");
  }
  if (B->symmetric && ptype == MATPRODUCT_ABt) {
    ptype = MATPRODUCT_AB;
    PetscCheck(product->symbolic_used_the_fact_B_is_symmetric,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Symbolic should have been built using the fact that B is symmetric");
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
    SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  Cmat = Ccusp->mat;
  PetscCheck(Amat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  PetscCheck(Bmat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
  PetscCheck(Cmat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C mult struct for product type %s",MatProductTypes[ptype]);
  Acsr = (CsrMatrix*)Amat->mat;
  Bcsr = mmdata->Bcsr ? mmdata->Bcsr : (CsrMatrix*)Bmat->mat; /* B may be in compressed row storage */
  Ccsr = (CsrMatrix*)Cmat->mat;
  PetscCheck(Acsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A CSR struct");
  PetscCheck(Bcsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B CSR struct");
  PetscCheck(Ccsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing C CSR struct");
  PetscCall(PetscLogGpuTimeBegin());
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  BmatSpDescr = mmdata->Bcsr ? mmdata->matSpBDescr : Bmat->matDescr; /* B may be in compressed row storage */
  PetscCallCUSPARSE(cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE));
  #if PETSC_PKG_CUDA_VERSION_GE(11,4,0)
    stat = cusparseSpGEMMreuse_compute(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                               mmdata->spgemmDesc);PetscCallCUSPARSE(stat);
  #else
    stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                               mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer);PetscCallCUSPARSE(stat);
    stat = cusparseSpGEMM_copy(Ccusp->handle, opA, opB,
                               Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                               cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc);PetscCallCUSPARSE(stat);
  #endif
#else
  stat = cusparse_csr_spgemm(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get());PetscCallCUSPARSE(stat);
#endif
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogGpuTimeEnd());
  C->offloadmask = PETSC_OFFLOAD_GPU;
finalize:
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  PetscCall(PetscInfo(C,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n",C->rmap->n,C->cmap->n,c->nz));
  PetscCall(PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(C,"Maximum nonzeros in any row is %" PetscInt_FMT "\n",c->rmax));
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
  cusparseStatus_t             stat;
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
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Product data not empty");
  A    = product->A;
  B    = product->B;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for type %s",((PetscObject)A)->type_name);
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJCUSPARSE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Not for B of type %s",((PetscObject)B)->type_name);
  a = (Mat_SeqAIJ*)A->data;
  b = (Mat_SeqAIJ*)B->data;
  /* product data */
  PetscCall(PetscNew(&mmdata));
  C->product->data    = mmdata;
  C->product->destroy = MatDestroy_MatMatCusparse;

  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(B));
  Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr; /* Access spptr after MatSeqAIJCUSPARSECopyToGPU, not before */
  Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr;
  PetscCheckFalse(Acusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");
  PetscCheckFalse(Bcusp->format != MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Only for MAT_CUSPARSE_CSR format");

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
    PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(A));
    Amat = Acusp->matTranspose;
    Bmat = Bcusp->mat;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    k = A->cmap->n;
    PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(B));
    Amat = Acusp->mat;
    Bmat = Bcusp->matTranspose;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  /* create cusparse matrix */
  PetscCall(MatSetSizes(C,m,n,m,n));
  PetscCall(MatSetType(C,MATSEQAIJCUSPARSE));
  c     = (Mat_SeqAIJ*)C->data;
  Ccusp = (Mat_SeqAIJCUSPARSE*)C->spptr;
  Cmat  = new Mat_SeqAIJCUSPARSEMultStruct;
  Ccsr  = new CsrMatrix;

  c->compressedrow.use = ciscompressed;
  if (c->compressedrow.use) { /* if a is in compressed row, than c will be in compressed row format */
    c->compressedrow.nrows = a->compressedrow.nrows;
    PetscCall(PetscMalloc2(c->compressedrow.nrows+1,&c->compressedrow.i,c->compressedrow.nrows,&c->compressedrow.rindex));
    PetscCall(PetscArraycpy(c->compressedrow.rindex,a->compressedrow.rindex,c->compressedrow.nrows));
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
  PetscCallCUSPARSE(cusparseCreateMatDescr(&Cmat->descr));
  PetscCallCUSPARSE(cusparseSetMatIndexBase(Cmat->descr, CUSPARSE_INDEX_BASE_ZERO));
  PetscCallCUSPARSE(cusparseSetMatType(Cmat->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  PetscCallCUDA(cudaMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar)));
  PetscCallCUDA(cudaMemcpy(Cmat->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(Cmat->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(Cmat->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
  if (!Ccsr->num_rows || !Ccsr->num_cols || !a->nz || !b->nz) { /* cusparse raise errors in different calls when matrices have zero rows/columns! */
    thrust::fill(thrust::device,Ccsr->row_offsets->begin(),Ccsr->row_offsets->end(),0);
    c->nz = 0;
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values = new THRUSTARRAY(c->nz);
    goto finalizesym;
  }

  PetscCheck(Amat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  PetscCheck(Bmat,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
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
      PetscCall(PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt)));
    }
    Bcsr->row_offsets = Bcusp->rowoffsets_gpu;
    mmdata->Bcsr = Bcsr;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    if (Bcsr->num_rows && Bcsr->num_cols) {
      stat = cusparseCreateCsr(&mmdata->matSpBDescr, Bcsr->num_rows, Bcsr->num_cols, Bcsr->num_entries,
                               Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                               Bcsr->values->data().get(),
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);PetscCallCUSPARSE(stat);
    }
    BmatSpDescr = mmdata->matSpBDescr;
#endif
  }
  PetscCheck(Acsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing A CSR struct");
  PetscCheck(Bcsr,PetscObjectComm((PetscObject)C),PETSC_ERR_GPU,"Missing B CSR struct");
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
  PetscCall(PetscLogGpuTimeBegin());

#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  PetscCallCUSPARSE(cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE));
  stat = cusparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, 0,
                          NULL, NULL, NULL,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);PetscCallCUSPARSE(stat);
  PetscCallCUSPARSE(cusparseSpGEMM_createDescr(&mmdata->spgemmDesc));
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
                                            &bufferSize1, NULL);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaMalloc((void**) &dBuffer1, bufferSize1));
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  stat = cusparseSpGEMMreuse_workEstimation(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                            CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                            &bufferSize1, dBuffer1);PetscCallCUSPARSE(stat);

  /*----------------------------------------------------------------------*/
  stat = cusparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                 CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                 &bufferSize2, NULL, &bufferSize3, NULL, &bufferSize4, NULL);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaMalloc((void**) &dBuffer2, bufferSize2));
  PetscCallCUDA(cudaMalloc((void**) &dBuffer3, bufferSize3));
  PetscCallCUDA(cudaMalloc((void**) &mmdata->dBuffer4, bufferSize4));
  stat = cusparseSpGEMMreuse_nnz(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                 CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                 &bufferSize2, dBuffer2, &bufferSize3, dBuffer3, &bufferSize4, mmdata->dBuffer4);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaFree(dBuffer1));
  PetscCallCUDA(cudaFree(dBuffer2));

  /*----------------------------------------------------------------------*/
  /* get matrix C non-zero entries C_nnz1 */
  PetscCallCUSPARSE(cusparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1));
  c->nz = (PetscInt) C_nnz1;
  /* allocate matrix C */
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values         = new THRUSTARRAY(c->nz);PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  /* update matC with the new pointers */
  stat = cusparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(),
                                Ccsr->values->data().get());PetscCallCUSPARSE(stat);

  /*----------------------------------------------------------------------*/
  stat = cusparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                  CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                  &bufferSize5, NULL);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaMalloc((void**) &mmdata->dBuffer5, bufferSize5));
  stat = cusparseSpGEMMreuse_copy(Ccusp->handle, opA, opB, Amat->matDescr, BmatSpDescr, Cmat->matDescr,
                                  CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc,
                                  &bufferSize5, mmdata->dBuffer5);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaFree(dBuffer3));
  stat = cusparseSpGEMMreuse_compute(Ccusp->handle, opA, opB,
                                     Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                     cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                     mmdata->spgemmDesc);PetscCallCUSPARSE(stat);
  PetscCall(PetscInfo(C,"Buffer sizes for type %s, result %" PetscInt_FMT " x %" PetscInt_FMT " (k %" PetscInt_FMT ", nzA %" PetscInt_FMT ", nzB %" PetscInt_FMT ", nzC %" PetscInt_FMT ") are: %ldKB %ldKB\n",MatProductTypes[ptype],m,n,k,a->nz,b->nz,c->nz,bufferSize4/1024,bufferSize5/1024));
 }
 #else
  size_t bufSize2;
  /* ask bufferSize bytes for external memory */
  stat = cusparseSpGEMM_workEstimation(Ccusp->handle, opA, opB,
                                       Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                       cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                       mmdata->spgemmDesc, &bufSize2, NULL);PetscCallCUSPARSE(stat);
  PetscCallCUDA(cudaMalloc((void**) &mmdata->mmBuffer2, bufSize2));
  /* inspect the matrices A and B to understand the memory requirement for the next step */
  stat = cusparseSpGEMM_workEstimation(Ccusp->handle, opA, opB,
                                       Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                       cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                       mmdata->spgemmDesc, &bufSize2, mmdata->mmBuffer2);PetscCallCUSPARSE(stat);
  /* ask bufferSize again bytes for external memory */
  stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                                Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                mmdata->spgemmDesc, &mmdata->mmBufferSize, NULL);PetscCallCUSPARSE(stat);
  /* The CUSPARSE documentation is not clear, nor the API
     We need both buffers to perform the operations properly!
     mmdata->mmBuffer2 does not appear anywhere in the compute/copy API
     it only appears for the workEstimation stuff, but it seems it is needed in compute, so probably the address
     is stored in the descriptor! What a messy API... */
  PetscCallCUDA(cudaMalloc((void**) &mmdata->mmBuffer, mmdata->mmBufferSize));
  /* compute the intermediate product of A * B */
  stat = cusparseSpGEMM_compute(Ccusp->handle, opA, opB,
                                Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                                cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT,
                                mmdata->spgemmDesc, &mmdata->mmBufferSize, mmdata->mmBuffer);PetscCallCUSPARSE(stat);
  /* get matrix C non-zero entries C_nnz1 */
  PetscCallCUSPARSE(cusparseSpMatGetSize(Cmat->matDescr, &C_num_rows1, &C_num_cols1, &C_nnz1));
  c->nz = (PetscInt) C_nnz1;
  PetscCall(PetscInfo(C,"Buffer sizes for type %s, result %" PetscInt_FMT " x %" PetscInt_FMT " (k %" PetscInt_FMT ", nzA %" PetscInt_FMT ", nzB %" PetscInt_FMT ", nzC %" PetscInt_FMT ") are: %ldKB %ldKB\n",MatProductTypes[ptype],m,n,k,a->nz,b->nz,c->nz,bufSize2/1024,mmdata->mmBufferSize/1024));
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  stat = cusparseCsrSetPointers(Cmat->matDescr, Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(),
                                Ccsr->values->data().get());PetscCallCUSPARSE(stat);
  stat = cusparseSpGEMM_copy(Ccusp->handle, opA, opB,
                             Cmat->alpha_one, Amat->matDescr, BmatSpDescr, Cmat->beta_zero, Cmat->matDescr,
                             cusparse_scalartype, CUSPARSE_SPGEMM_DEFAULT, mmdata->spgemmDesc);PetscCallCUSPARSE(stat);
 #endif // PETSC_PKG_CUDA_VERSION_GE(11,4,0)
#else
  PetscCallCUSPARSE(cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_HOST));
  stat = cusparseXcsrgemmNnz(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->row_offsets->data().get(), &cnz);PetscCallCUSPARSE(stat);
  c->nz = cnz;
  Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
  PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */
  Ccsr->values = new THRUSTARRAY(c->nz);
  PetscCallCUDA(cudaPeekAtLastError()); /* catch out of memory errors */

  PetscCallCUSPARSE(cusparseSetPointerMode(Ccusp->handle, CUSPARSE_POINTER_MODE_DEVICE));
  /* with the old gemm interface (removed from 11.0 on) we cannot compute the symbolic factorization only.
     I have tried using the gemm2 interface (alpha * A * B + beta * D), which allows to do symbolic by passing NULL for values, but it seems quite buggy when
     D is NULL, despite the fact that CUSPARSE documentation claims it is supported! */
  stat = cusparse_csr_spgemm(Ccusp->handle, opA, opB,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get());PetscCallCUSPARSE(stat);
#endif
  PetscCall(PetscLogGpuFlops(mmdata->flops));
  PetscCall(PetscLogGpuTimeEnd());
finalizesym:
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  PetscCall(PetscMalloc1(m+1,&c->i));
  PetscCall(PetscMalloc1(c->nz,&c->j));
  if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
    PetscInt *d_i = c->i;
    THRUSTINTARRAY ii(Ccsr->row_offsets->size());
    THRUSTINTARRAY jj(Ccsr->column_indices->size());
    ii   = *Ccsr->row_offsets;
    jj   = *Ccsr->column_indices;
    if (ciscompressed) d_i = c->compressedrow.i;
    PetscCallCUDA(cudaMemcpy(d_i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
  } else {
    PetscInt *d_i = c->i;
    if (ciscompressed) d_i = c->compressedrow.i;
    PetscCallCUDA(cudaMemcpy(d_i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
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
  PetscCall(PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt)));
  PetscCall(PetscMalloc1(m,&c->ilen));
  PetscCall(PetscMalloc1(m,&c->imax));
  c->maxnz = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k+1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax,nn);
  }
  PetscCall(MatMarkDiagonal_SeqAIJ(C));
  PetscCall(PetscMalloc1(c->nz,&c->a));
  Ccsr->num_entries = c->nz;

  C->nonzerostate++;
  PetscCall(PetscLayoutSetUp(C->rmap));
  PetscCall(PetscLayoutSetUp(C->cmap));
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
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)product->B,MATSEQDENSE,&isdense));
  if (!product->A->boundtocpu && !product->B->boundtocpu) {
    PetscCall(PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJCUSPARSE,&Biscusp));
  }
  if (product->type == MATPRODUCT_ABC) {
    Ciscusp = PETSC_FALSE;
    if (!product->C->boundtocpu) {
      PetscCall(PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJCUSPARSE,&Ciscusp));
    }
  }
  if (Biscusp && Ciscusp) { /* we can always select the CPU backend */
    PetscBool usecpu = PETSC_FALSE;
    switch (product->type) {
    case MATPRODUCT_AB:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatMatMult","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-matmatmult_backend_cpu","Use CPU code","MatMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_AB","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu","Use CPU code","MatMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      }
      break;
    case MATPRODUCT_AtB:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatTransposeMatMult","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mattransposematmult_backend_cpu","Use CPU code","MatTransposeMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_AtB","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu","Use CPU code","MatTransposeMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      }
      break;
    case MATPRODUCT_PtAP:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatPtAP","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-matptap_backend_cpu","Use CPU code","MatPtAP",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_PtAP","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu","Use CPU code","MatPtAP",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      }
      break;
    case MATPRODUCT_RARt:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatRARt","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-matrart_backend_cpu","Use CPU code","MatRARt",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_RARt","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu","Use CPU code","MatRARt",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      }
      break;
    case MATPRODUCT_ABC:
      if (product->api_user) {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatMatMatMult","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-matmatmatmult_backend_cpu","Use CPU code","MatMatMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
      } else {
        ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"MatProduct_ABC","Mat");PetscCall(ierr);
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu","Use CPU code","MatMatMatMult",usecpu,&usecpu,NULL));
        ierr = PetscOptionsEnd();PetscCall(ierr);
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
        PetscCall(MatProductSetFromOptions_SeqAIJ_SeqDense(mat));
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
    PetscCall(MatProductSetFromOptions_SeqAIJ(mat));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_FALSE));
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
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  PetscBool                    compressed;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  PetscInt                     nx,ny;
#endif

  PetscFunctionBegin;
  PetscCheckFalse(herm && !trans,PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"Hermitian and not transpose not supported");
  if (!a->nz) {
    if (!yy) PetscCall(VecSet_SeqCUDA(zz,0));
    else PetscCall(VecCopy_SeqCUDA(yy,zz));
    PetscFunctionReturn(0);
  }
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  if (!trans) {
    matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    PetscCheck(matstruct,PetscObjectComm((PetscObject)A),PETSC_ERR_GPU,"SeqAIJCUSPARSE does not have a 'mat' (need to fix)");
  } else {
    if (herm || !A->form_explicit_transpose) {
      opA = herm ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
      matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    } else {
      if (!cusparsestruct->matTranspose) PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(A));
      matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->matTranspose;
    }
  }
  /* Does the matrix use compressed rows (i.e., drop zero rows)? */
  compressed = matstruct->cprowIndices ? PETSC_TRUE : PETSC_FALSE;

  try {
    PetscCall(VecCUDAGetArrayRead(xx,(const PetscScalar**)&xarray));
    if (yy == zz) PetscCall(VecCUDAGetArray(zz,&zarray)); /* read & write zz, so need to get uptodate zarray on GPU */
    else PetscCall(VecCUDAGetArrayWrite(zz,&zarray)); /* write zz, so no need to init zarray on GPU */

    PetscCall(PetscLogGpuTimeBegin());
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
      PetscCheck(opA >= 0 && opA <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE ABI on cusparseOperation_t has changed and PETSc has not been updated accordingly");
      if (!matstruct->cuSpMV[opA].initialized) { /* built on demand */
        PetscCallCUSPARSE(cusparseCreateDnVec(&matstruct->cuSpMV[opA].vecXDescr,nx,xptr,cusparse_scalartype));
        PetscCallCUSPARSE(cusparseCreateDnVec(&matstruct->cuSpMV[opA].vecYDescr,ny,dptr,cusparse_scalartype));
        PetscCallCUSPARSE(cusparseSpMV_bufferSize(cusparsestruct->handle, opA, matstruct->alpha_one,
                                               matstruct->matDescr,
                                               matstruct->cuSpMV[opA].vecXDescr, beta,
                                               matstruct->cuSpMV[opA].vecYDescr,
                                               cusparse_scalartype,
                                               cusparsestruct->spmvAlg,
                                               &matstruct->cuSpMV[opA].spmvBufferSize));
        PetscCallCUDA(cudaMalloc(&matstruct->cuSpMV[opA].spmvBuffer,matstruct->cuSpMV[opA].spmvBufferSize));

        matstruct->cuSpMV[opA].initialized = PETSC_TRUE;
      } else {
        /* x, y's value pointers might change between calls, but their shape is kept, so we just update pointers */
        PetscCallCUSPARSE(cusparseDnVecSetValues(matstruct->cuSpMV[opA].vecXDescr,xptr));
        PetscCallCUSPARSE(cusparseDnVecSetValues(matstruct->cuSpMV[opA].vecYDescr,dptr));
      }

      PetscCallCUSPARSE(cusparseSpMV(cusparsestruct->handle, opA,
                                  matstruct->alpha_one,
                                  matstruct->matDescr, /* built in MatSeqAIJCUSPARSECopyToGPU() or MatSeqAIJCUSPARSEFormExplicitTranspose() */
                                  matstruct->cuSpMV[opA].vecXDescr,
                                  beta,
                                  matstruct->cuSpMV[opA].vecYDescr,
                                  cusparse_scalartype,
                                  cusparsestruct->spmvAlg,
                                  matstruct->cuSpMV[opA].spmvBuffer));
     #else
      CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
      PetscCallCUSPARSE(cusparse_csr_spmv(cusparsestruct->handle, opA,
                                       mat->num_rows, mat->num_cols,
                                       mat->num_entries, matstruct->alpha_one, matstruct->descr,
                                       mat->values->data().get(), mat->row_offsets->data().get(),
                                       mat->column_indices->data().get(), xptr, beta,
                                       dptr));
     #endif
    } else {
      if (cusparsestruct->nrows) {
       #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
       #else
        cusparseHybMat_t hybMat = (cusparseHybMat_t)matstruct->mat;
        PetscCallCUSPARSE(cusparse_hyb_spmv(cusparsestruct->handle, opA,
                                         matstruct->alpha_one, matstruct->descr, hybMat,
                                         xptr, beta,
                                         dptr));
       #endif
      }
    }
    PetscCall(PetscLogGpuTimeEnd());

    if (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      if (yy) { /* MatMultAdd: zz = A*xx + yy */
        if (compressed) { /* A is compressed. We first copy yy to zz, then ScatterAdd the work vector to zz */
          PetscCall(VecCopy_SeqCUDA(yy,zz)); /* zz = yy */
        } else if (zz != yy) { /* A is not compressed. zz already contains A*xx, and we just need to add yy */
          PetscCall(VecAXPY_SeqCUDA(zz,1.0,yy)); /* zz += yy */
        }
      } else if (compressed) { /* MatMult: zz = A*xx. A is compressed, so we zero zz first, then ScatterAdd the work vector to zz */
        PetscCall(VecSet_SeqCUDA(zz,0));
      }

      /* ScatterAdd the result from work vector into the full vector when A is compressed */
      if (compressed) {
        PetscCall(PetscLogGpuTimeBegin());
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
        PetscCall(PetscLogGpuTimeEnd());
      }
    } else {
      if (yy && yy != zz) {
        PetscCall(VecAXPY_SeqCUDA(zz,1.0,yy)); /* zz += yy */
      }
    }
    PetscCall(VecCUDARestoreArrayRead(xx,(const PetscScalar**)&xarray));
    if (yy == zz) PetscCall(VecCUDARestoreArray(zz,&zarray));
    else PetscCall(VecCUDARestoreArrayWrite(zz,&zarray));
  } catch(char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  if (yy) {
    PetscCall(PetscLogGpuFlops(2.0*a->nz));
  } else {
    PetscCall(PetscLogGpuFlops(2.0*a->nz-a->nonzerorowcnt));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAddKernel_SeqAIJCUSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscObjectState   onnz = A->nonzerostate;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqAIJ(A,mode));
  if (onnz != A->nonzerostate && cusp->deviceMat) {

    PetscCall(PetscInfo(A,"Destroy device mat since nonzerostate changed\n"));
    PetscCallCUDA(cudaFree(cusp->deviceMat));
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
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQAIJCUSPARSE));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(MatSeqAIJCUSPARSE_Destroy((Mat_SeqAIJCUSPARSE**)&A->spptr));
  } else {
    PetscCall(MatSeqAIJCUSPARSETriFactors_Destroy((Mat_SeqAIJCUSPARSETriFactors**)&A->spptr));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetUseCPUSolve_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqaijcusparse_hypre_C",NULL));
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUSPARSE(Mat,MatType,MatReuse,Mat*);
static PetscErrorCode MatBindToCPU_SeqAIJCUSPARSE(Mat,PetscBool);
static PetscErrorCode MatDuplicate_SeqAIJCUSPARSE(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqAIJ(A,cpvalues,B));
  PetscCall(MatConvert_SeqAIJ_SeqAIJCUSPARSE(*B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqAIJCUSPARSE(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
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
    PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(Y,PETSC_FALSE));
    PetscCall(MatAXPY_SeqAIJ(Y,a,X,str));
    PetscFunctionReturn(0);
  }
  /* if we are here, it means both matrices are bound to GPU */
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(Y));
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(X));
  PetscCheck(cy->format == MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)Y),PETSC_ERR_GPU,"only MAT_CUSPARSE_CSR supported");
  PetscCheck(cx->format == MAT_CUSPARSE_CSR,PetscObjectComm((PetscObject)X),PETSC_ERR_GPU,"only MAT_CUSPARSE_CSR supported");
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
    PetscScalar b = 1.0;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    size_t      bufferSize;
    void        *buffer;
#endif

    PetscCall(MatSeqAIJCUSPARSEGetArrayRead(X,&ax));
    PetscCall(MatSeqAIJCUSPARSEGetArray(Y,&ay));
    PetscCallCUSPARSE(cusparseSetPointerMode(cy->handle, CUSPARSE_POINTER_MODE_HOST));
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    PetscCallCUSPARSE(cusparse_csr_spgeam_bufferSize(cy->handle,Y->rmap->n,Y->cmap->n,
                                                  &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                                                  &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                                  cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),&bufferSize));
    PetscCallCUDA(cudaMalloc(&buffer,bufferSize));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUSPARSE(cusparse_csr_spgeam(cy->handle,Y->rmap->n,Y->cmap->n,
                                       &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                                       &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                       cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),buffer));
    PetscCall(PetscLogGpuFlops(x->nz + y->nz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCallCUDA(cudaFree(buffer));
#else
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUSPARSE(cusparse_csr_spgeam(cy->handle,Y->rmap->n,Y->cmap->n,
                                       &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                                       &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                       cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get()));
    PetscCall(PetscLogGpuFlops(x->nz + y->nz));
    PetscCall(PetscLogGpuTimeEnd());
#endif
    PetscCallCUSPARSE(cusparseSetPointerMode(cy->handle, CUSPARSE_POINTER_MODE_DEVICE));
    PetscCall(MatSeqAIJCUSPARSERestoreArrayRead(X,&ax));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(Y,&ay));
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  } else if (str == SAME_NONZERO_PATTERN) {
    cublasHandle_t cublasv2handle;
    PetscBLASInt   one = 1, bnz = 1;

    PetscCall(MatSeqAIJCUSPARSEGetArrayRead(X,&ax));
    PetscCall(MatSeqAIJCUSPARSEGetArray(Y,&ay));
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(PetscBLASIntCast(x->nz,&bnz));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bnz,&a,ax,one,ay,one));
    PetscCall(PetscLogGpuFlops(2.0*bnz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(MatSeqAIJCUSPARSERestoreArrayRead(X,&ax));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(Y,&ay));
    PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  } else {
    PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(Y,PETSC_FALSE));
    PetscCall(MatAXPY_SeqAIJ(Y,a,X,str));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJCUSPARSE(Mat Y,PetscScalar a)
{
  Mat_SeqAIJ     *y = (Mat_SeqAIJ*)Y->data;
  PetscScalar    *ay;
  cublasHandle_t cublasv2handle;
  PetscBLASInt   one = 1, bnz = 1;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCUSPARSEGetArray(Y,&ay));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(y->nz,&bnz));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXscal(cublasv2handle,bnz,&a,ay,one));
  PetscCall(PetscLogGpuFlops(bnz));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatSeqAIJCUSPARSERestoreArray(Y,&ay));
  PetscCall(MatSeqAIJInvalidateDiagonal(Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJCUSPARSE(Mat A)
{
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
  PetscCall(PetscArrayzero(a->a,a->i[A->rmap->n]));
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqAIJCUSPARSE(Mat A,PetscBool flg)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) {
    A->boundtocpu = flg;
    PetscFunctionReturn(0);
  }
  if (flg) {
    PetscCall(MatSeqAIJCUSPARSECopyFromGPU(A));

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
    PetscCall(PetscMemzero(a->ops,sizeof(Mat_SeqAIJOps)));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJ));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",NULL));
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
    a->ops->getarray                  = MatSeqAIJGetArray_SeqAIJCUSPARSE;
    a->ops->restorearray              = MatSeqAIJRestoreArray_SeqAIJCUSPARSE;
    a->ops->getarrayread              = MatSeqAIJGetArrayRead_SeqAIJCUSPARSE;
    a->ops->restorearrayread          = MatSeqAIJRestoreArrayRead_SeqAIJCUSPARSE;
    a->ops->getarraywrite             = MatSeqAIJGetArrayWrite_SeqAIJCUSPARSE;
    a->ops->restorearraywrite         = MatSeqAIJRestoreArrayWrite_SeqAIJCUSPARSE;
    a->ops->getcsrandmemtype          = MatSeqAIJGetCSRAndMemType_SeqAIJCUSPARSE;

    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",MatSeqAIJCopySubArray_SeqAIJCUSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",MatProductSetFromOptions_SeqAIJCUSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",MatProductSetFromOptions_SeqAIJCUSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_SeqAIJCUSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_SeqAIJCUSPARSE));
    PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqaijcusparse_C",MatProductSetFromOptions_SeqAIJCUSPARSE));
   }
  A->boundtocpu = flg;
  if (flg && a->inode.size) {
    a->inode.use = PETSC_TRUE;
  } else {
    a->inode.use = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUSPARSE(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  Mat              B;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA)); /* first use of CUSPARSE may be via MatConvert */
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(A,*newmat,SAME_NONZERO_PATTERN));
  }
  B = *newmat;

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA,&B->defaultvectype));

  if (reuse != MAT_REUSE_MATRIX && !B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      Mat_SeqAIJCUSPARSE *spptr;
      PetscCall(PetscNew(&spptr));
      PetscCallCUSPARSE(cusparseCreate(&spptr->handle));
      PetscCallCUSPARSE(cusparseSetStream(spptr->handle,PetscDefaultCudaStream));
      spptr->format     = MAT_CUSPARSE_CSR;
     #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
     #if PETSC_PKG_CUDA_VERSION_GE(11,2,0)
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

      PetscCall(PetscNew(&spptr));
      PetscCallCUSPARSE(cusparseCreate(&spptr->handle));
      PetscCallCUSPARSE(cusparseSetStream(spptr->handle,PetscDefaultCudaStream));
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

  PetscCall(MatBindToCPU_SeqAIJCUSPARSE(B,PETSC_FALSE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_SeqAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijcusparse_hypre_C",MatConvert_AIJ_HYPRE));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatCUSPARSESetUseCPUSolve_C",MatCUSPARSESetUseCPUSolve_SeqAIJCUSPARSE));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqAIJ(B));
  PetscCall(MatConvert_SeqAIJ_SeqAIJCUSPARSE(B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,&B));
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
+  -mat_cusparse_use_cpu_solve - Do MatSolve on CPU

  Level: beginner

.seealso: MatCreateSeqAIJCUSPARSE(), MATAIJCUSPARSE, MatCreateAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijcusparse_cusparse_band(Mat,MatFactorType,Mat*);

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_CUSPARSE(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERCUSPARSEBAND,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_seqaijcusparse_cusparse_band));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_LU,MatGetFactor_seqaijcusparse_cusparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_CHOLESKY,MatGetFactor_seqaijcusparse_cusparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_ILU,MatGetFactor_seqaijcusparse_cusparse));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUSPARSE,MATSEQAIJCUSPARSE,MAT_FACTOR_ICC,MatGetFactor_seqaijcusparse_cusparse));

  PetscFunctionReturn(0);
}

static PetscErrorCode MatResetPreallocationCOO_SeqAIJCUSPARSE(Mat mat)
{
  Mat_SeqAIJCUSPARSE* cusp = (Mat_SeqAIJCUSPARSE*)mat->spptr;

  PetscFunctionBegin;
  if (!cusp) PetscFunctionReturn(0);
  delete cusp->cooPerm;
  delete cusp->cooPerm_a;
  cusp->cooPerm = NULL;
  cusp->cooPerm_a = NULL;
  if (cusp->use_extended_coo) {
    PetscCallCUDA(cudaFree(cusp->jmap_d));
    PetscCallCUDA(cudaFree(cusp->perm_d));
  }
  cusp->use_extended_coo = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSE_Destroy(Mat_SeqAIJCUSPARSE **cusparsestruct)
{
  PetscFunctionBegin;
  if (*cusparsestruct) {
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*cusparsestruct)->mat,(*cusparsestruct)->format));
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*cusparsestruct)->matTranspose,(*cusparsestruct)->format));
    delete (*cusparsestruct)->workVector;
    delete (*cusparsestruct)->rowoffsets_gpu;
    delete (*cusparsestruct)->cooPerm;
    delete (*cusparsestruct)->cooPerm_a;
    delete (*cusparsestruct)->csr2csc_i;
    if ((*cusparsestruct)->handle) PetscCallCUSPARSE(cusparseDestroy((*cusparsestruct)->handle));
    if ((*cusparsestruct)->jmap_d) PetscCallCUDA(cudaFree((*cusparsestruct)->jmap_d));
    if ((*cusparsestruct)->perm_d) PetscCallCUDA(cudaFree((*cusparsestruct)->perm_d));
    PetscCall(PetscFree(*cusparsestruct));
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
  PetscFunctionBegin;
  if (*trifactor) {
    if ((*trifactor)->descr) PetscCallCUSPARSE(cusparseDestroyMatDescr((*trifactor)->descr));
    if ((*trifactor)->solveInfo) PetscCallCUSPARSE(cusparse_destroy_analysis_info((*trifactor)->solveInfo));
    PetscCall(CsrMatrix_Destroy(&(*trifactor)->csrMat));
    if ((*trifactor)->solveBuffer)   PetscCallCUDA(cudaFree((*trifactor)->solveBuffer));
    if ((*trifactor)->AA_h)   PetscCallCUDA(cudaFreeHost((*trifactor)->AA_h));
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    if ((*trifactor)->csr2cscBuffer) PetscCallCUDA(cudaFree((*trifactor)->csr2cscBuffer));
   #endif
    PetscCall(PetscFree(*trifactor));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEMultStruct_Destroy(Mat_SeqAIJCUSPARSEMultStruct **matstruct,MatCUSPARSEStorageFormat format)
{
  CsrMatrix        *mat;

  PetscFunctionBegin;
  if (*matstruct) {
    if ((*matstruct)->mat) {
      if (format==MAT_CUSPARSE_ELL || format==MAT_CUSPARSE_HYB) {
       #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_CUSPARSE_ELL and MAT_CUSPARSE_HYB are not supported since CUDA-11.0");
       #else
        cusparseHybMat_t hybMat = (cusparseHybMat_t)(*matstruct)->mat;
        PetscCallCUSPARSE(cusparseDestroyHybMat(hybMat));
       #endif
      } else {
        mat = (CsrMatrix*)(*matstruct)->mat;
        CsrMatrix_Destroy(&mat);
      }
    }
    if ((*matstruct)->descr) PetscCallCUSPARSE(cusparseDestroyMatDescr((*matstruct)->descr));
    delete (*matstruct)->cprowIndices;
    if ((*matstruct)->alpha_one) PetscCallCUDA(cudaFree((*matstruct)->alpha_one));
    if ((*matstruct)->beta_zero) PetscCallCUDA(cudaFree((*matstruct)->beta_zero));
    if ((*matstruct)->beta_one)  PetscCallCUDA(cudaFree((*matstruct)->beta_one));

   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    Mat_SeqAIJCUSPARSEMultStruct *mdata = *matstruct;
    if (mdata->matDescr) PetscCallCUSPARSE(cusparseDestroySpMat(mdata->matDescr));
    for (int i=0; i<3; i++) {
      if (mdata->cuSpMV[i].initialized) {
        PetscCallCUDA(cudaFree(mdata->cuSpMV[i].spmvBuffer));
        PetscCallCUSPARSE(cusparseDestroyDnVec(mdata->cuSpMV[i].vecXDescr));
        PetscCallCUSPARSE(cusparseDestroyDnVec(mdata->cuSpMV[i].vecYDescr));
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
  PetscFunctionBegin;
  if (*trifactors) {
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtr));
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtr));
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtrTranspose));
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtrTranspose));
    delete (*trifactors)->rpermIndices;
    delete (*trifactors)->cpermIndices;
    delete (*trifactors)->workVector;
    (*trifactors)->rpermIndices = NULL;
    (*trifactors)->cpermIndices = NULL;
    (*trifactors)->workVector = NULL;
    if ((*trifactors)->a_band_d)   PetscCallCUDA(cudaFree((*trifactors)->a_band_d));
    if ((*trifactors)->i_band_d)   PetscCallCUDA(cudaFree((*trifactors)->i_band_d));
    (*trifactors)->init_dev_prop = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Destroy(Mat_SeqAIJCUSPARSETriFactors** trifactors)
{
  cusparseHandle_t handle;

  PetscFunctionBegin;
  if (*trifactors) {
    PetscCall(MatSeqAIJCUSPARSETriFactors_Reset(trifactors));
    if (handle = (*trifactors)->handle) {
      PetscCallCUSPARSE(cusparseDestroy(handle));
    }
    PetscCall(PetscFree(*trifactors));
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
/* Associated with MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic() */
PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE_Basic(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJCUSPARSE                    *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ                            *a = (Mat_SeqAIJ*)A->data;
  THRUSTARRAY                           *cooPerm_v = NULL;
  thrust::device_ptr<const PetscScalar> d_v;
  CsrMatrix                             *matrix;
  PetscInt                              n;

  PetscFunctionBegin;
  PetscCheck(cusp,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE struct");
  PetscCheck(cusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE CsrMatrix");
  if (!cusp->cooPerm) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }
  matrix = (CsrMatrix*)cusp->mat->mat;
  PetscCheck(matrix->values,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
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
    PetscCall(PetscLogCpuToGpu(n*sizeof(PetscScalar)));
  }
  PetscCall(PetscLogGpuTimeBegin());
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
  PetscCall(PetscLogGpuTimeEnd());
finalize:
  delete cooPerm_v;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  PetscCall(PetscInfo(A,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n",A->rmap->n,A->cmap->n,a->nz));
  PetscCall(PetscInfo(A,"Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(A,"Maximum nonzeros in any row is %" PetscInt_FMT "\n",a->rmax));
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

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  if (!cusp) PetscFunctionReturn(0);
  if (destroy) {
    PetscCall(MatSeqAIJCUSPARSEMultStruct_Destroy(&cusp->matTranspose,cusp->format));
    delete cusp->csr2csc_i;
    cusp->csr2csc_i = NULL;
  }
  A->transupdated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#include <thrust/binary_search.h>
/* 'Basic' means it only works when coo_i[] and coo_j[] do not contain negative indices */
PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(Mat A, PetscCount n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           cooPerm_n, nzr = 0;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
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

    PetscCall(PetscLogCpuToGpu(2.*n*sizeof(PetscInt)));
    d_i.assign(coo_i,coo_i+n);
    d_j.assign(coo_j,coo_j+n);

    /* Ex.
      n = 6
      coo_i = [3,3,1,4,1,4]
      coo_j = [3,2,2,5,2,6]
    */
    auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin()));
    auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end()));

    PetscCall(PetscLogGpuTimeBegin());
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
    PetscCall(PetscLogGpuTimeEnd());

    PetscCall(MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i));
    a->singlemalloc = PETSC_FALSE;
    a->free_a       = PETSC_TRUE;
    a->free_ij      = PETSC_TRUE;
    PetscCall(PetscMalloc1(A->rmap->n+1,&a->i));
    a->i[0] = 0; /* a->i = [0,0,1,1,3,5,5] */
    PetscCallCUDA(cudaMemcpy(a->i+1,ii.data().get(),A->rmap->n*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    a->nz = a->maxnz = a->i[A->rmap->n];
    a->rmax = 0;
    PetscCall(PetscMalloc1(a->nz,&a->a));
    PetscCall(PetscMalloc1(a->nz,&a->j));
    PetscCallCUDA(cudaMemcpy(a->j,d_j.data().get(),a->nz*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    if (!a->ilen) PetscCall(PetscMalloc1(A->rmap->n,&a->ilen));
    if (!a->imax) PetscCall(PetscMalloc1(A->rmap->n,&a->imax));
    for (PetscInt i = 0; i < A->rmap->n; i++) {
      const PetscInt nnzr = a->i[i+1] - a->i[i];
      nzr += (PetscInt)!!(nnzr);
      a->ilen[i] = a->imax[i] = nnzr;
      a->rmax = PetscMax(a->rmax,nnzr);
    }
    a->nonzerorowcnt = nzr;
    A->preallocated = PETSC_TRUE;
    PetscCall(PetscLogGpuToCpu((A->rmap->n+a->nz)*sizeof(PetscInt)));
    PetscCall(MatMarkDiagonal_SeqAIJ(A));
  } else {
    PetscCall(MatSeqAIJSetPreallocation(A,0,NULL));
  }
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  /* We want to allocate the CUSPARSE struct for matvec now.
     The code is so convoluted now that I prefer to copy zeros */
  PetscCall(PetscArrayzero(a->a,a->nz));
  PetscCall(MatCheckCompressedRow(A,nzr,&a->compressedrow,a->i,A->rmap->n,0.6));
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_SeqAIJ         *seq;
  Mat_SeqAIJCUSPARSE *dev;
  PetscBool          coo_basic = PETSC_TRUE;
  PetscMemType       mtype = PETSC_MEMTYPE_DEVICE;

  PetscFunctionBegin;
  PetscCall(MatResetPreallocationCOO_SeqAIJ(mat));
  PetscCall(MatResetPreallocationCOO_SeqAIJCUSPARSE(mat));
  if (coo_i) {
    PetscCall(PetscGetMemType(coo_i,&mtype));
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k=0; k<coo_n; k++) {
        if (coo_i[k] < 0 || coo_j[k] < 0) {coo_basic = PETSC_FALSE; break;}
      }
    }
  }

  if (coo_basic) { /* i,j are on device or do not contain negative indices */
    PetscCall(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(mat,coo_n,coo_i,coo_j));
  } else {
    PetscCall(MatSetPreallocationCOO_SeqAIJ(mat,coo_n,coo_i,coo_j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(mat));
    seq  = static_cast<Mat_SeqAIJ*>(mat->data);
    dev  = static_cast<Mat_SeqAIJCUSPARSE*>(mat->spptr);
    PetscCallCUDA(cudaMalloc((void**)&dev->jmap_d,(seq->nz+1)*sizeof(PetscCount)));
    PetscCallCUDA(cudaMemcpy(dev->jmap_d,seq->jmap,(seq->nz+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMalloc((void**)&dev->perm_d,seq->Atot*sizeof(PetscCount)));
    PetscCallCUDA(cudaMemcpy(dev->perm_d,seq->perm,seq->Atot*sizeof(PetscCount),cudaMemcpyHostToDevice));
    dev->use_extended_coo = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

__global__ void MatAddCOOValues(const PetscScalar kv[],PetscCount nnz,const PetscCount jmap[],const PetscCount perm[],InsertMode imode,PetscScalar a[])
{
  PetscCount        i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount  grid_size = gridDim.x * blockDim.x;
  for (; i<nnz; i+= grid_size) {
    PetscScalar sum = 0.0;
    for (PetscCount k=jmap[i]; k<jmap[i+1]; k++) sum += kv[perm[k]];
    a[i] = (imode == INSERT_VALUES? 0.0 : a[i]) + sum;
  }
}

PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJ          *seq = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE  *dev = (Mat_SeqAIJCUSPARSE*)A->spptr;
  PetscCount          Annz = seq->nz;
  PetscMemType        memtype;
  const PetscScalar   *v1 = v;
  PetscScalar         *Aa;

  PetscFunctionBegin;
  if (dev->use_extended_coo) {
    PetscCall(PetscGetMemType(v,&memtype));
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
      PetscCallCUDA(cudaMalloc((void**)&v1,seq->coo_n*sizeof(PetscScalar)));
      PetscCallCUDA(cudaMemcpy((void*)v1,v,seq->coo_n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(A,&Aa));
    else PetscCall(MatSeqAIJCUSPARSEGetArray(A,&Aa));

    if (Annz) {
      MatAddCOOValues<<<(Annz+255)/256,256>>>(v1,Annz,dev->jmap_d,dev->perm_d,imode,Aa);
      PetscCallCUDA(cudaPeekAtLastError());
    }

    if (imode == INSERT_VALUES) PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(A,&Aa));
    else PetscCall(MatSeqAIJCUSPARSERestoreArray(A,&Aa));

    if (PetscMemTypeHost(memtype)) PetscCallCUDA(cudaFree((void*)v1));
  } else {
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(A,v,imode));
  }
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
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!i || !j) PetscFunctionReturn(0);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckFalse(cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCheck(cusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  if (i) {
    if (!compressed && a->compressedrow.use) { /* need full row offset */
      if (!cusp->rowoffsets_gpu) {
        cusp->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
        cusp->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
        PetscCall(PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt)));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckFalse(cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCheck(cusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  PetscCheck(csr->values,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckFalse(cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
  PetscCheck(cusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  PetscCheck(csr->values,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckFalse(cusp->format == MAT_CUSPARSE_ELL || cusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  PetscCheck(cusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
  csr = (CsrMatrix*)cusp->mat->mat;
  PetscCheck(csr->values,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(A,PETSC_FALSE));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data, *b = (Mat_SeqAIJ*)B->data, *c;
  Mat_SeqAIJCUSPARSE           *Acusp = (Mat_SeqAIJCUSPARSE*)A->spptr, *Bcusp = (Mat_SeqAIJCUSPARSE*)B->spptr, *Ccusp;
  Mat_SeqAIJCUSPARSEMultStruct *Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscInt                     Annz,Bnnz;
  cusparseStatus_t             stat;
  PetscInt                     i,m,n,zero = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidPointer(C,4);
  PetscCheckTypeName(A,MATSEQAIJCUSPARSE);
  PetscCheckTypeName(B,MATSEQAIJCUSPARSE);
  PetscCheck(A->rmap->n == B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT,A->rmap->n,B->rmap->n);
  PetscCheckFalse(reuse == MAT_INPLACE_MATRIX,PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_INPLACE_MATRIX not supported");
  PetscCheckFalse(Acusp->format == MAT_CUSPARSE_ELL || Acusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  PetscCheckFalse(Bcusp->format == MAT_CUSPARSE_ELL || Bcusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (reuse == MAT_INITIAL_MATRIX) {
    m     = A->rmap->n;
    n     = A->cmap->n + B->cmap->n;
    PetscCall(MatCreate(PETSC_COMM_SELF,C));
    PetscCall(MatSetSizes(*C,m,n,m,n));
    PetscCall(MatSetType(*C,MATSEQAIJCUSPARSE));
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
    PetscCallCUSPARSE(cusparseCreateMatDescr(&Cmat->descr));
    PetscCallCUSPARSE(cusparseSetMatIndexBase(Cmat->descr, CUSPARSE_INDEX_BASE_ZERO));
    PetscCallCUSPARSE(cusparseSetMatType(Cmat->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    PetscCallCUDA(cudaMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy(Cmat->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(Cmat->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(Cmat->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(B));
    PetscCheck(Acusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
    PetscCheck(Bcusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");

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
          PetscCall(PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt)));
        }
        Aroff = Acusp->rowoffsets_gpu;
      } else Aroff = Acsr->row_offsets;
      if (b->compressedrow.use) { /* need full row offset */
        if (!Bcusp->rowoffsets_gpu) {
          Bcusp->rowoffsets_gpu  = new THRUSTINTARRAY32(B->rmap->n + 1);
          Bcusp->rowoffsets_gpu->assign(b->i,b->i + B->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt)));
        }
        Broff = Bcusp->rowoffsets_gpu;
      } else Broff = Bcsr->row_offsets;
      PetscCall(PetscLogGpuTimeBegin());
      stat = cusparseXcsr2coo(Acusp->handle,
                              Aroff->data().get(),
                              Annz,
                              m,
                              Acoo->data().get(),
                              CUSPARSE_INDEX_BASE_ZERO);PetscCallCUSPARSE(stat);
      stat = cusparseXcsr2coo(Bcusp->handle,
                              Broff->data().get(),
                              Bnnz,
                              m,
                              Bcoo->data().get(),
                              CUSPARSE_INDEX_BASE_ZERO);PetscCallCUSPARSE(stat);
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
                              CUSPARSE_INDEX_BASE_ZERO);PetscCallCUSPARSE(stat);
      PetscCall(PetscLogGpuTimeEnd());
      delete wPerm;
      delete Acoo;
      delete Bcoo;
      delete Ccoo;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      stat = cusparseCreateCsr(&Cmat->matDescr, Ccsr->num_rows, Ccsr->num_cols, Ccsr->num_entries,
                               Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get(), Ccsr->values->data().get(),
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);PetscCallCUSPARSE(stat);
#endif
      if (A->form_explicit_transpose && B->form_explicit_transpose) { /* if A and B have the transpose, generate C transpose too */
        PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(A));
        PetscCall(MatSeqAIJCUSPARSEFormExplicitTranspose(B));
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

        PetscCall(PetscLogGpuTimeBegin());
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
        PetscCall(PetscLogGpuTimeEnd());

        PetscCallCUSPARSE(cusparseCreateMatDescr(&CmatT->descr));
        PetscCallCUSPARSE(cusparseSetMatIndexBase(CmatT->descr, CUSPARSE_INDEX_BASE_ZERO));
        PetscCallCUSPARSE(cusparseSetMatType(CmatT->descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        PetscCallCUDA(cudaMalloc((void **)&(CmatT->alpha_one),sizeof(PetscScalar)));
        PetscCallCUDA(cudaMalloc((void **)&(CmatT->beta_zero),sizeof(PetscScalar)));
        PetscCallCUDA(cudaMalloc((void **)&(CmatT->beta_one), sizeof(PetscScalar)));
        PetscCallCUDA(cudaMemcpy(CmatT->alpha_one,&PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
        PetscCallCUDA(cudaMemcpy(CmatT->beta_zero,&PETSC_CUSPARSE_ZERO,sizeof(PetscScalar),cudaMemcpyHostToDevice));
        PetscCallCUDA(cudaMemcpy(CmatT->beta_one, &PETSC_CUSPARSE_ONE, sizeof(PetscScalar),cudaMemcpyHostToDevice));
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
        stat = cusparseCreateCsr(&CmatT->matDescr, CcsrT->num_rows, CcsrT->num_cols, CcsrT->num_entries,
                                 CcsrT->row_offsets->data().get(), CcsrT->column_indices->data().get(), CcsrT->values->data().get(),
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, cusparse_scalartype);PetscCallCUSPARSE(stat);
#endif
        Ccusp->matTranspose = CmatT;
      }
    }

    c->singlemalloc = PETSC_FALSE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
    PetscCall(PetscMalloc1(m+1,&c->i));
    PetscCall(PetscMalloc1(c->nz,&c->j));
    if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
      THRUSTINTARRAY ii(Ccsr->row_offsets->size());
      THRUSTINTARRAY jj(Ccsr->column_indices->size());
      ii   = *Ccsr->row_offsets;
      jj   = *Ccsr->column_indices;
      PetscCallCUDA(cudaMemcpy(c->i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
      PetscCallCUDA(cudaMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    } else {
      PetscCallCUDA(cudaMemcpy(c->i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
      PetscCallCUDA(cudaMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),cudaMemcpyDeviceToHost));
    }
    PetscCall(PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt)));
    PetscCall(PetscMalloc1(m,&c->ilen));
    PetscCall(PetscMalloc1(m,&c->imax));
    c->maxnz = c->nz;
    c->nonzerorowcnt = 0;
    c->rmax = 0;
    for (i = 0; i < m; i++) {
      const PetscInt nn = c->i[i+1] - c->i[i];
      c->ilen[i] = c->imax[i] = nn;
      c->nonzerorowcnt += (PetscInt)!!nn;
      c->rmax = PetscMax(c->rmax,nn);
    }
    PetscCall(MatMarkDiagonal_SeqAIJ(*C));
    PetscCall(PetscMalloc1(c->nz,&c->a));
    (*C)->nonzerostate++;
    PetscCall(PetscLayoutSetUp((*C)->rmap));
    PetscCall(PetscLayoutSetUp((*C)->cmap));
    Ccusp->nonzerostate = (*C)->nonzerostate;
    (*C)->preallocated  = PETSC_TRUE;
  } else {
    PetscCheckFalse((*C)->rmap->n != B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT,(*C)->rmap->n,B->rmap->n);
    c = (Mat_SeqAIJ*)(*C)->data;
    if (c->nz) {
      Ccusp = (Mat_SeqAIJCUSPARSE*)(*C)->spptr;
      PetscCheck(Ccusp->cooPerm,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cooPerm");
      PetscCheckFalse(Ccusp->format == MAT_CUSPARSE_ELL || Ccusp->format == MAT_CUSPARSE_HYB,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
      PetscCheckFalse(Ccusp->nonzerostate != (*C)->nonzerostate,PETSC_COMM_SELF,PETSC_ERR_COR,"Wrong nonzerostate");
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(B));
      PetscCheck(Acusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
      PetscCheck(Bcusp->mat,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJCUSPARSEMultStruct");
      Acsr = (CsrMatrix*)Acusp->mat->mat;
      Bcsr = (CsrMatrix*)Bcusp->mat->mat;
      Ccsr = (CsrMatrix*)Ccusp->mat->mat;
      PetscCheckFalse(Acsr->num_entries != (PetscInt)Acsr->values->size(),PETSC_COMM_SELF,PETSC_ERR_COR,"A nnz %" PetscInt_FMT " != %" PetscInt_FMT,Acsr->num_entries,(PetscInt)Acsr->values->size());
      PetscCheckFalse(Bcsr->num_entries != (PetscInt)Bcsr->values->size(),PETSC_COMM_SELF,PETSC_ERR_COR,"B nnz %" PetscInt_FMT " != %" PetscInt_FMT,Bcsr->num_entries,(PetscInt)Bcsr->values->size());
      PetscCheckFalse(Ccsr->num_entries != (PetscInt)Ccsr->values->size(),PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %" PetscInt_FMT " != %" PetscInt_FMT,Ccsr->num_entries,(PetscInt)Ccsr->values->size());
      PetscCheckFalse(Ccsr->num_entries != Acsr->num_entries + Bcsr->num_entries,PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %" PetscInt_FMT " != %" PetscInt_FMT " + %" PetscInt_FMT,Ccsr->num_entries,Acsr->num_entries,Bcsr->num_entries);
      PetscCheck(Ccusp->cooPerm->size() == Ccsr->values->size(),PETSC_COMM_SELF,PETSC_ERR_COR,"permSize %" PetscInt_FMT " != %" PetscInt_FMT,(PetscInt)Ccusp->cooPerm->size(),(PetscInt)Ccsr->values->size());
      auto pmid = Ccusp->cooPerm->begin();
      thrust::advance(pmid,Acsr->num_entries);
      PetscCall(PetscLogGpuTimeBegin());
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
      PetscCall(MatSeqAIJCUSPARSEInvalidateTranspose(*C,PETSC_FALSE));
      if (A->form_explicit_transpose && B->form_explicit_transpose && (*C)->form_explicit_transpose) {
        PetscCheck(Ccusp->matTranspose,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing transpose Mat_SeqAIJCUSPARSEMultStruct");
        PetscBool AT = Acusp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bcusp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        CsrMatrix *AcsrT = AT ? (CsrMatrix*)Acusp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix*)Bcusp->matTranspose->mat : NULL;
        CsrMatrix *CcsrT = (CsrMatrix*)Ccusp->matTranspose->mat;
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(),AcsrT->values->end(),vT);
        if (BT) thrust::copy(BcsrT->values->begin(),BcsrT->values->end(),vT);
        (*C)->transupdated = PETSC_TRUE;
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)*C));
  (*C)->assembled     = PETSC_TRUE;
  (*C)->was_assembled = PETSC_FALSE;
  (*C)->offloadmask   = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJCUSPARSE(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[])
{
  bool              dmem;
  const PetscScalar *av;

  PetscFunctionBegin;
  dmem = isCudaMem(v);
  PetscCall(MatSeqAIJCUSPARSEGetArrayRead(A,&av));
  if (n && idx) {
    THRUSTINTARRAY widx(n);
    widx.assign(idx,idx+n);
    PetscCall(PetscLogCpuToGpu(n*sizeof(PetscInt)));

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
      PetscCallCUDA(cudaMemcpy(v,w->data().get(),n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    }
    delete w;
  } else {
    PetscCallCUDA(cudaMemcpy(v,av,n*sizeof(PetscScalar),dmem ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));
  }
  if (!dmem) PetscCall(PetscLogCpuToGpu(n*sizeof(PetscScalar)));
  PetscCall(MatSeqAIJCUSPARSERestoreArrayRead(A,&av));
  PetscFunctionReturn(0);
}
