/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the CUSPARSE library,
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>

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
static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Reset(Mat_SeqAIJCUSPARSETriFactors**);
static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Destroy(Mat_SeqAIJCUSPARSETriFactors**);
static PetscErrorCode MatSeqAIJCUSPARSE_Destroy(Mat_SeqAIJCUSPARSE**);

PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat,const PetscScalar[],InsertMode);

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
  (*B)->useordering = PETSC_TRUE;
  ierr = MatSetType(*B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSE;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJCUSPARSE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSE Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
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

/*@
   MatSeqAIJCUSPARSESetGenerateTranspose - Sets the flag to explicitly generate the tranpose matrix before calling MatMultTranspose

   Collective on mat

   Input Parameters:
+  A - Matrix of type SEQAIJCUSPARSE
-  transgen - the boolean flag

   Level: intermediate

.seealso: MATSEQAIJCUSPARSE
@*/
PetscErrorCode MatSeqAIJCUSPARSESetGenerateTranspose(Mat A,PetscBool transgen)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare(((PetscObject)A),MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (flg) {
    Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;

    if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    cusp->transgen = transgen;
    if (!transgen) { /* need to destroy the transpose matrix if present to prevent from logic errors if transgen is set to true later */
      ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusp->matTranspose,cusp->format);CHKERRQ(ierr);
    }
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
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJCUSPARSE options");CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscBool transgen = cusparsestruct->transgen;

    ierr = PetscOptionsBool("-mat_cusparse_transgen","Generate explicit transpose for MatMultTranspose","MatSeqAIJCUSPARSESetGenerateTranspose",transgen,&transgen,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatSeqAIJCUSPARSESetGenerateTranspose(A,transgen);CHKERRQ(ierr);}

    ierr = PetscOptionsEnum("-mat_cusparse_mult_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT,format);CHKERRQ(ierr);}

    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV and TriSolve",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);}
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    cusparsestruct->spmvAlg = CUSPARSE_CSRMV_ALG1; /* default, since we only support csr */
    ierr = PetscOptionsEnum("-mat_cusparse_spmv_alg","sets cuSPARSE algorithm used in sparse-mat dense-vector multiplication (SpMV)",
                            "cusparseSpMVAlg_t",MatCUSPARSESpMVAlgorithms,(PetscEnum)cusparsestruct->spmvAlg,(PetscEnum*)&cusparsestruct->spmvAlg,&flg);CHKERRQ(ierr);
    /* If user did use this option, check its consistency with cuSPARSE, since PetscOptionsEnum() sets enum values based on their position in MatCUSPARSESpMVAlgorithms[] */
    if (flg && CUSPARSE_CSRMV_ALG1 != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMVAlg_t has been changed but PETSc has not been updated accordingly");

    cusparsestruct->spmmAlg = CUSPARSE_SPMM_CSR_ALG1; /* default, only support column-major dense matrix B */
    ierr = PetscOptionsEnum("-mat_cusparse_spmm_alg","sets cuSPARSE algorithm used in sparse-mat dense-mat multiplication (SpMM)",
                            "cusparseSpMMAlg_t",MatCUSPARSESpMMAlgorithms,(PetscEnum)cusparsestruct->spmmAlg,(PetscEnum*)&cusparsestruct->spmmAlg,&flg);CHKERRQ(ierr);
    if (flg && CUSPARSE_SPMM_CSR_ALG1 != 4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cuSPARSE enum cusparseSpMMAlg_t has been changed but PETSc has not been updated accordingly");

    cusparsestruct->csr2cscAlg = CUSPARSE_CSR2CSC_ALG1;
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
                                 loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo
                               #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
                               #endif
);CHKERRCUSPARSE(stat);
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
                                 upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo
                               #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
                               #endif
);CHKERRCUSPARSE(stat);
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
                                 upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo
                                #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
                                #endif
);CHKERRCUSPARSE(stat);
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
                                 loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo
                                #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                                 ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
                                #endif
);CHKERRCUSPARSE(stat);
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

static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row,iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
                          CUSPARSE_CSR2CSC_ALG1, loTriFactor->csr2cscBuffer
                        #else
                          loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                          CUSPARSE_ACTION_NUMERIC, indexBase
                        #endif
);CHKERRCUSPARSE(stat);
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
                           loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo
                          #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                           ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
                          #endif
);CHKERRCUSPARSE(stat);
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
                          CUSPARSE_CSR2CSC_ALG1, upTriFactor->csr2cscBuffer
                        #else
                          upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                          CUSPARSE_ACTION_NUMERIC, indexBase
                        #endif
);CHKERRCUSPARSE(stat);
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
                           upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo
                          #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                           ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
                          #endif
);CHKERRCUSPARSE(stat);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(MAT_CUSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtrTranspose = upTriFactorT;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCUSPARSEGenerateTransposeForMult(Mat A)
{
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
  Mat_SeqAIJCUSPARSEMultStruct *matstructT = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->matTranspose;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  cusparseStatus_t             stat;
  cusparseIndexBase_t          indexBase;
  cudaError_t                  err;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  if (!cusparsestruct->transgen || cusparsestruct->matTranspose) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* create cusparse matrix */
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
  stat = cusparseSetPointerMode(cusparsestruct->handle, CUSPARSE_POINTER_MODE_DEVICE);CHKERRCUSPARSE(stat);

  if (cusparsestruct->format==MAT_CUSPARSE_CSR) {
    CsrMatrix *matrix = (CsrMatrix*)matstruct->mat;
    CsrMatrix *matrixT= new CsrMatrix;
    matrixT->num_rows = A->cmap->n;
    matrixT->num_cols = A->rmap->n;
    matrixT->num_entries = a->nz;
    matrixT->row_offsets = new THRUSTINTARRAY32(matrixT->num_rows+1);
    matrixT->column_indices = new THRUSTINTARRAY32(a->nz);
    matrixT->values = new THRUSTARRAY(a->nz);

    cusparsestruct->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1);
    cusparsestruct->rowoffsets_gpu->assign(a->i,a->i+A->rmap->n+1);

    /* compute the transpose, i.e. the CSC */
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    stat = cusparseCsr2cscEx2_bufferSize(cusparsestruct->handle, A->rmap->n,
                                  A->cmap->n, matrix->num_entries,
                                  matrix->values->data().get(),
                                  cusparsestruct->rowoffsets_gpu->data().get(),
                                  matrix->column_indices->data().get(),
                                  matrixT->values->data().get(),
                                  matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), cusparse_scalartype,
                                  CUSPARSE_ACTION_NUMERIC,indexBase,
                                  cusparsestruct->csr2cscAlg, &cusparsestruct->csr2cscBufferSize);CHKERRCUSPARSE(stat);
    err = cudaMalloc(&cusparsestruct->csr2cscBuffer,cusparsestruct->csr2cscBufferSize);CHKERRCUDA(err);
   #endif

    stat = cusparse_csr2csc(cusparsestruct->handle, A->rmap->n,
                            A->cmap->n, matrix->num_entries,
                            matrix->values->data().get(),
                            cusparsestruct->rowoffsets_gpu->data().get(),
                            matrix->column_indices->data().get(),
                            matrixT->values->data().get(),
                          #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                            matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), cusparse_scalartype,
                            CUSPARSE_ACTION_NUMERIC,indexBase,
                            cusparsestruct->csr2cscAlg, cusparsestruct->csr2cscBuffer
                          #else
                            matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(),
                            CUSPARSE_ACTION_NUMERIC, indexBase
                          #endif
);CHKERRCUSPARSE(stat);
    matstructT->mat = matrixT;

   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    stat = cusparseCreateCsr(&matstructT->matDescr,
                             matrixT->num_rows, matrixT->num_cols, matrixT->num_entries,
                             matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(),
                             matrixT->values->data().get(),
                             CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, /* row offset, col idx type due to THRUSTINTARRAY32 */
                             indexBase,cusparse_scalartype);CHKERRCUSPARSE(stat);
   #endif
  } else if (cusparsestruct->format==MAT_CUSPARSE_ELL || cusparsestruct->format==MAT_CUSPARSE_HYB) {
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
  err  = WaitForCUDA();CHKERRCUDA(err);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  /* the compressed row indices is not used for matTranspose */
  matstructT->cprowIndices = NULL;
  /* assign the pointer */
  ((Mat_SeqAIJCUSPARSE*)A->spptr)->matTranspose = matstructT;
  PetscFunctionReturn(0);
}

/* Why do we need to analyze the tranposed matrix again? Can't we just use op(A) = CUSPARSE_OPERATION_TRANSPOSE in MatSolve_SeqAIJCUSPARSE? */
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
  cudaError_t                           cerr;

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
  thrust::copy(thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->begin()),
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
                        xarray, tempGPU->data().get()
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

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
                        tempGPU->data().get(), xarray
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::make_permutation_iterator(xGPU, cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(xGPU+n, cusparseTriFactors->cpermIndices->end()),
               tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
  cudaError_t                       cerr;

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
                        barray, tempGPU->data().get()
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

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
                        tempGPU->data().get(), xarray
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
  cudaError_t                           cerr;

  PetscFunctionBegin;

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->begin()),
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
                        tempGPU->data().get(), xarray
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

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
                        upTriFactor->solveInfo,
                        xarray, tempGPU->data().get()
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->end()),
               xGPU);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
  cudaError_t                       cerr;

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
                        barray, tempGPU->data().get()
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

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
                        tempGPU->data().get(), xarray
                      #if PETSC_PKG_CUDA_VERSION_GE(9,0,0)
                        ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
                      #endif
);CHKERRCUSPARSE(stat);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
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

static PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct = cusparsestruct->mat;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  PetscInt                     m = A->rmap->n,*ii,*ridx,tmp;
  PetscErrorCode               ierr;
  cusparseStatus_t             stat;
  cudaError_t                  err;

  PetscFunctionBegin;
  if (A->boundtocpu) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (A->was_assembled && A->nonzerostate == cusparsestruct->nonzerostate && cusparsestruct->format == MAT_CUSPARSE_CSR) {
      /* Copy values only */
      CsrMatrix *matrix,*matrixT;
      matrix = (CsrMatrix*)cusparsestruct->mat->mat;

      ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      matrix->values->assign(a->a, a->a+a->nz);
      err  = WaitForCUDA();CHKERRCUDA(err);
      ierr = PetscLogCpuToGpu((a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);

      /* Update matT when it was built before */
      if (cusparsestruct->matTranspose) {
        cusparseIndexBase_t indexBase = cusparseGetMatIndexBase(cusparsestruct->mat->descr);
        matrixT = (CsrMatrix*)cusparsestruct->matTranspose->mat;
        ierr = PetscLogEventBegin(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
        stat = cusparse_csr2csc(cusparsestruct->handle, A->rmap->n,
                            A->cmap->n, matrix->num_entries,
                            matrix->values->data().get(),
                            cusparsestruct->rowoffsets_gpu->data().get(),
                            matrix->column_indices->data().get(),
                            matrixT->values->data().get(),
                          #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
                            matrixT->row_offsets->data().get(), matrixT->column_indices->data().get(), cusparse_scalartype,
                            CUSPARSE_ACTION_NUMERIC,indexBase,
                            cusparsestruct->csr2cscAlg, cusparsestruct->csr2cscBuffer
                          #else
                            matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(),
                            CUSPARSE_ACTION_NUMERIC, indexBase
                          #endif
);CHKERRCUSPARSE(stat);
        err  = WaitForCUDA();CHKERRCUDA(err);
        ierr = PetscLogEventEnd(MAT_CUSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusparsestruct->mat,cusparsestruct->format);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusparsestruct->matTranspose,cusparsestruct->format);CHKERRQ(ierr);
      delete cusparsestruct->workVector;
      delete cusparsestruct->rowoffsets_gpu;
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
        cusparsestruct->nrows = m;

        /* create cusparse matrix */
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
          mat->num_entries = a->nz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(a->nz);
          mat->column_indices->assign(a->j, a->j+a->nz);

          mat->values = new THRUSTARRAY(a->nz);
          mat->values->assign(a->a, a->a+a->nz);

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
          mat->num_entries = a->nz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(a->nz);
          mat->column_indices->assign(a->j, a->j+a->nz);

          mat->values = new THRUSTARRAY(a->nz);
          mat->values->assign(a->a, a->a+a->nz);

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
    A->offloadmask = PETSC_OFFLOAD_BOTH;
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
  PetscBool            cisdense;
  PetscScalar          *Bt;
  Mat                  X;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  PetscBool            initialized;   /* C = alpha op(A) op(B) + beta C */
  cusparseDnMatDescr_t matBDescr;
  cusparseDnMatDescr_t matCDescr;
  size_t               spmmBufferSize;
  void                 *spmmBuffer;
  PetscInt             Blda,Clda; /* Record leading dimensions of B and C here to detect changes*/
#endif
};

static PetscErrorCode MatDestroy_MatMatCusparse(void *data)
{
  PetscErrorCode ierr;
  MatMatCusparse *mmdata = (MatMatCusparse *)data;
  cudaError_t    cerr;

  PetscFunctionBegin;
  cerr = cudaFree(mmdata->Bt);CHKERRCUDA(cerr);
 #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  cusparseStatus_t stat;
  if (mmdata->matBDescr)  {stat = cusparseDestroyDnMat(mmdata->matBDescr);CHKERRCUSPARSE(stat);}
  if (mmdata->matCDescr)  {stat = cusparseDestroyDnMat(mmdata->matCDescr);CHKERRCUSPARSE(stat);}
  if (mmdata->spmmBuffer) {cerr = cudaFree(mmdata->spmmBuffer);CHKERRCUDA(cerr);}
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
  cudaError_t                  cerr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  mmdata = (MatMatCusparse*)product->data;
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  /* currently CopyToGpu does not copy if the matrix is bound to CPU
     Instead of silently accepting the wrong answer, I prefer to raise the error */
  if (A->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a CUSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_PtAP:
    mat = cusp->mat;
    opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    if (!cusp->transgen) {
      mat = cusp->mat;
      opA = CUSPARSE_OPERATION_TRANSPOSE;
    } else {
      ierr = MatSeqAIJCUSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
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
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  if (!mat) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct");
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
  /* (re)allcoate spmmBuffer if not initialized or LDAs are different */
  if (!mmdata->initialized || mmdata->Blda != blda || mmdata->Clda != clda) {
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
                                   cusp->spmmAlg,&mmdata->spmmBufferSize);CHKERRCUSPARSE(stat);
    if (mmdata->spmmBuffer) {cerr = cudaFree(mmdata->spmmBuffer);CHKERRCUDA(cerr);}
    cerr = cudaMalloc(&mmdata->spmmBuffer,mmdata->spmmBufferSize);CHKERRCUDA(cerr);
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
                      cusp->spmmAlg,mmdata->spmmBuffer);CHKERRCUSPARSE(stat);
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
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  if (cusp->format != MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_CUSPARSE_CSR format");
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
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
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

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);

/* handles dense B */
static PetscErrorCode MatProductSetFromOptions_SeqAIJCUSPARSE(Mat C)
{
  Mat_Product    *product = C->product;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!product->A) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing A");
  if (product->A->boundtocpu) {
    ierr = MatProductSetFromOptions_SeqAIJ_SeqDense(C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_AtB:
  case MATPRODUCT_ABt:
  case MATPRODUCT_PtAP:
  case MATPRODUCT_RARt:
    C->ops->productsymbolic = MatProductSymbolic_SeqAIJCUSPARSE_SeqDENSECUDA;
  default:
    break;
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

/* z = op(A) x + y. If trans & !herm, op = ^T; if trans & herm, op = ^H; if !trans, op = no-op */
static PetscErrorCode MatMultAddKernel_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans,PetscBool herm)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstruct;
  PetscScalar                  *xarray,*zarray,*dptr,*beta,*xptr;
  PetscErrorCode               ierr;
  cudaError_t                  cerr;
  cusparseStatus_t             stat;
  cusparseOperation_t          opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  PetscBool                    compressed;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  PetscInt                     nx,ny;
#endif

  PetscFunctionBegin;
  if (herm && !trans) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Hermitian and not transpose not supported");
  if (!a->nonzerorowcnt) {
    if (!yy) {ierr = VecSet_SeqCUDA(zz,0);CHKERRQ(ierr);}
    else {ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!trans) {
    matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    if (!matstruct) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"SeqAIJCUSPARSE does not have a 'mat' (need to fix)");
  } else {
    if (herm || !cusparsestruct->transgen) {
      opA = herm ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
      matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
    } else {
      if (!cusparsestruct->matTranspose) {ierr = MatSeqAIJCUSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);}
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
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))),
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
                               matstruct->matDescr, /* built in MatSeqAIJCUSPARSECopyToGPU() or MatSeqAIJCUSPARSEGenerateTransposeForMult() */
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
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
        thrust::device_ptr<PetscScalar> zptr = thrust::device_pointer_cast(zarray);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(cusparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecCUDAPlusEquals());
        cerr = WaitForCUDA();CHKERRCUDA(cerr);
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
  PetscErrorCode              ierr;
  PetscSplitCSRDataStructure  *d_mat = NULL;
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    d_mat = ((Mat_SeqAIJCUSPARSE*)A->spptr)->deviceMat;
  }
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr); // this does very little if assembled on GPU - call it?
  if (mode == MAT_FLUSH_ASSEMBLY || A->boundtocpu) PetscFunctionReturn(0);
  if (d_mat) {
    A->offloadmask = PETSC_OFFLOAD_GPU;
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
  PetscErrorCode              ierr;
  PetscSplitCSRDataStructure  *d_mat = NULL;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    d_mat = ((Mat_SeqAIJCUSPARSE*)A->spptr)->deviceMat;
    ((Mat_SeqAIJCUSPARSE*)A->spptr)->deviceMat = NULL;
    ierr = MatSeqAIJCUSPARSE_Destroy((Mat_SeqAIJCUSPARSE**)&A->spptr);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJCUSPARSETriFactors_Destroy((Mat_SeqAIJCUSPARSETriFactors**)&A->spptr);CHKERRQ(ierr);
  }
  if (d_mat) {
    Mat_SeqAIJ                 *a = (Mat_SeqAIJ*)A->data;
    cudaError_t                err;
    PetscSplitCSRDataStructure h_mat;
    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    err = cudaMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    if (a->compressedrow.use) {
      err = cudaFree(h_mat.diag.i);CHKERRCUDA(err);
    }
    err = cudaFree(d_mat);CHKERRCUDA(err);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
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

static PetscErrorCode MatAXPY_SeqAIJCUSPARSE(Mat Y,PetscScalar a,Mat X,MatStructure str) // put axpy in aijcusparse, etc.
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;
  PetscBool      flgx,flgy;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  ierr = PetscObjectTypeCompare((PetscObject)Y,MATSEQAIJCUSPARSE,&flgy);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQAIJCUSPARSE,&flgx);CHKERRQ(ierr);
  if (!flgx || !flgy) {
    ierr = MatAXPY_SeqAIJ( Y, a, X, str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (Y->factortype != MAT_FACTOR_NONE || X->factortype != MAT_FACTOR_NONE) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_PLIB,"both matrices must be MAT_FACTOR_NONE");
  if (str == DIFFERENT_NONZERO_PATTERN) {
    if (x->nz == y->nz) {
      PetscBool e;
      ierr = PetscArraycmp(x->i,y->i,Y->rmap->n+1,&e);CHKERRQ(ierr);
      if (e) {
        ierr = PetscArraycmp(x->j,y->j,y->nz,&e);CHKERRQ(ierr);
        if (e) {
          str = SAME_NONZERO_PATTERN;
        }
      }
    }
  }
  if (str != SAME_NONZERO_PATTERN) {
    ierr = MatAXPY_SeqAIJ( Y, a, X, str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    Mat_SeqAIJCUSPARSE           *cusparsestruct_y = (Mat_SeqAIJCUSPARSE*)Y->spptr;
    Mat_SeqAIJCUSPARSE           *cusparsestruct_x = (Mat_SeqAIJCUSPARSE*)X->spptr;
    if (cusparsestruct_y->format!=MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_PLIB,"only MAT_CUSPARSE_CSR supported");
    if (cusparsestruct_x->format!=MAT_CUSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_PLIB,"only MAT_CUSPARSE_CSR supported");
    if (!cusparsestruct_y->mat || !cusparsestruct_x->mat) {
      if (Y->offloadmask == PETSC_OFFLOAD_UNALLOCATED || Y->offloadmask == PETSC_OFFLOAD_GPU) {
        ierr = MatSeqAIJCUSPARSECopyFromGPU(Y);CHKERRQ(ierr);
      }
      if (X->offloadmask == PETSC_OFFLOAD_UNALLOCATED || X->offloadmask == PETSC_OFFLOAD_GPU) {
        ierr = MatSeqAIJCUSPARSECopyFromGPU(X);CHKERRQ(ierr);
      }
      ierr = MatAXPY_SeqAIJ(Y,a,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSECopyToGPU(Y);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSECopyToGPU(X);CHKERRQ(ierr);
    } else {
      cublasHandle_t cublasv2handle;
      cublasStatus_t cberr;
      cudaError_t    err;
      PetscScalar    alpha = a;
      PetscBLASInt   one = 1, bnz = 1;
      CsrMatrix      *matrix_y = (CsrMatrix*)cusparsestruct_y->mat->mat;
      CsrMatrix      *matrix_x = (CsrMatrix*)cusparsestruct_x->mat->mat;
      PetscScalar    *aa_y, *aa_x;
      aa_y = matrix_y->values->data().get();
      aa_x = matrix_x->values->data().get();
      ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(x->nz,&bnz);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cberr = cublasXaxpy(cublasv2handle,bnz,&alpha,aa_x,one,aa_y,one);CHKERRCUBLAS(cberr);
      err  = WaitForCUDA();CHKERRCUDA(err);
      ierr = PetscLogGpuFlops(2.0*bnz);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
      ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
      if (Y->offloadmask == PETSC_OFFLOAD_BOTH) Y->offloadmask = PETSC_OFFLOAD_GPU;
      else if (Y->offloadmask != PETSC_OFFLOAD_GPU) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_PLIB,"wrong state");
      ierr = MatSeqAIJCUSPARSECopyFromGPU(Y);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode             ierr;
  PetscBool                  both = PETSC_FALSE;
  Mat_SeqAIJ                 *a = (Mat_SeqAIJ*)A->data;

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

    A->ops->axpy                      = MatAXPY_SeqAIJ;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJ;
    A->ops->mult                      = MatMult_SeqAIJ;
    A->ops->multadd                   = MatMultAdd_SeqAIJ;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJ;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJ;
    A->ops->multhermitiantranspose    = NULL;
    A->ops->multhermitiantransposeadd = NULL;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJ);CHKERRQ(ierr);
  } else {
    A->ops->axpy                      = MatAXPY_SeqAIJCUSPARSE;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJCUSPARSE;
    A->ops->mult                      = MatMult_SeqAIJCUSPARSE;
    A->ops->multadd                   = MatMultAdd_SeqAIJCUSPARSE;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJCUSPARSE;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJCUSPARSE;
    A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJCUSPARSE;
    A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJCUSPARSE;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdensecuda_C",MatProductSetFromOptions_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijcusparse_seqdense_C",MatProductSetFromOptions_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_SeqAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJCUSPARSE);CHKERRQ(ierr);
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
      spptr->format = MAT_CUSPARSE_CSR;
      stat = cusparseCreate(&spptr->handle);CHKERRCUSPARSE(stat);
      B->spptr = spptr;
      spptr->deviceMat = NULL;
    } else {
      Mat_SeqAIJCUSPARSETriFactors *spptr;

      ierr = PetscNew(&spptr);CHKERRQ(ierr);
      stat = cusparseCreate(&spptr->handle);CHKERRCUSPARSE(stat);
      B->spptr = spptr;
    }
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSPARSE;
  B->ops->destroy        = MatDestroy_SeqAIJCUSPARSE;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJCUSPARSE;
  B->ops->bindtocpu      = MatBindToCPU_SeqAIJCUSPARSE;
  B->ops->duplicate      = MatDuplicate_SeqAIJCUSPARSE;

  ierr = MatBindToCPU_SeqAIJCUSPARSE(B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_SeqAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCUSPARSE(B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);
  ierr = MatSetFromOptions_SeqAIJCUSPARSE(PetscOptionsObject,B);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijcusparse_cusparse(Mat,MatFactorType,Mat*);


PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_CUSPARSE(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    delete (*cusparsestruct)->cooPerm_v;
    delete (*cusparsestruct)->cooPerm_w;
    if ((*cusparsestruct)->handle) {stat = cusparseDestroy((*cusparsestruct)->handle);CHKERRCUSPARSE(stat);}
   #if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
    cudaError_t cerr = cudaFree((*cusparsestruct)->csr2cscBuffer);CHKERRCUDA(cerr);
   #endif
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

static PetscErrorCode MatSeqAIJCUSPARSETriFactors_Reset(Mat_SeqAIJCUSPARSETriFactors** trifactors)
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
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJCUSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *matrix;
  PetscErrorCode     ierr;
  cudaError_t        cerr;
  PetscInt           n;

  PetscFunctionBegin;
  if (!cusp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE struct");
  if (!cusp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUSPARSE CsrMatrix");
  if (!cusp->cooPerm) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  n = cusp->cooPerm->size();
  matrix = (CsrMatrix*)cusp->mat->mat;
  if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
  if (!cusp->cooPerm_v) { cusp->cooPerm_v = new THRUSTARRAY(n); }
  ierr = PetscLogEventBegin(MAT_CUSPARSESetVCOO,A,0,0,0);CHKERRQ(ierr);
  if (v) {
    cusp->cooPerm_v->assign(v,v+n);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  else thrust::fill(thrust::device,cusp->cooPerm_v->begin(),cusp->cooPerm_v->end(),0.);
  if (imode == ADD_VALUES) {
    if (cusp->cooPerm_a) {
      if (!cusp->cooPerm_w) cusp->cooPerm_w = new THRUSTARRAY(matrix->values->size());
      auto vbit = thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->begin());
      thrust::reduce_by_key(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),vbit,thrust::make_discard_iterator(),cusp->cooPerm_w->begin(),thrust::equal_to<PetscInt>(),thrust::plus<PetscScalar>());
      thrust::transform(cusp->cooPerm_w->begin(),cusp->cooPerm_w->end(),matrix->values->begin(),matrix->values->begin(),thrust::plus<PetscScalar>());
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecCUDAPlusEquals());
    }
  } else {
    if (cusp->cooPerm_a) { /* non unique values insertion, result is undefined (we cannot guarantee last takes precedence)
                              if we are inserting two different values into the same location */
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->begin()),
                                                                thrust::make_permutation_iterator(matrix->values->begin(),cusp->cooPerm_a->begin())));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->end()),
                                                                thrust::make_permutation_iterator(matrix->values->begin(),cusp->cooPerm_a->end())));
      thrust::for_each(zibit,zieit,VecCUDAEquals());
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(cusp->cooPerm_v->begin(),cusp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecCUDAEquals());
    }
  }
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(MAT_CUSPARSESetVCOO,A,0,0,0);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* we can remove this call when MatSeqAIJGetArray operations are used everywhere! */
  ierr = MatSeqAIJCUSPARSECopyFromGPU(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <thrust/binary_search.h>
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJCUSPARSE(Mat A, PetscInt n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  PetscErrorCode     ierr;
  Mat_SeqAIJCUSPARSE *cusp = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CsrMatrix          *matrix;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           cooPerm_n, nzr = 0;
  cudaError_t        cerr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_CUSPARSEPreallCOO,A,0,0,0);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  cooPerm_n = cusp->cooPerm ? cusp->cooPerm->size() : 0;
  if (n != cooPerm_n) {
    delete cusp->cooPerm;
    delete cusp->cooPerm_v;
    delete cusp->cooPerm_w;
    delete cusp->cooPerm_a;
    cusp->cooPerm = NULL;
    cusp->cooPerm_v = NULL;
    cusp->cooPerm_w = NULL;
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
    auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin()));
    auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end()));

    thrust::sequence(thrust::device, cusp->cooPerm->begin(), cusp->cooPerm->end(), 0);
    thrust::sort_by_key(fkey, ekey, cusp->cooPerm->begin(), IJCompare());
    *cusp->cooPerm_a = d_i;
    THRUSTINTARRAY w = d_j;

    auto nekey = thrust::unique(fkey, ekey, IJEqual());
    if (nekey == ekey) { /* all entries are unique */
      delete cusp->cooPerm_a;
      cusp->cooPerm_a = NULL;
    } else { /* I couldn't come up with a more elegant algorithm */
      adjacent_difference(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),cusp->cooPerm_a->begin(),IJDiff());
      adjacent_difference(w.begin(),w.end(),w.begin(),IJDiff());
      (*cusp->cooPerm_a)[0] = 0;
      w[0] = 0;
      thrust::transform(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),w.begin(),cusp->cooPerm_a->begin(),IJSum());
      thrust::inclusive_scan(cusp->cooPerm_a->begin(),cusp->cooPerm_a->end(),cusp->cooPerm_a->begin(),thrust::plus<PetscInt>());
    }
    thrust::counting_iterator<PetscInt> search_begin(0);
    thrust::upper_bound(d_i.begin(), nekey.get_iterator_tuple().get<0>(),
                        search_begin, search_begin + A->rmap->n,
                        ii.begin());

    ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
    a->singlemalloc = PETSC_FALSE;
    a->free_a       = PETSC_TRUE;
    a->free_ij      = PETSC_TRUE;
    ierr = PetscMalloc1(A->rmap->n+1,&a->i);CHKERRQ(ierr);
    a->i[0] = 0;
    cerr = cudaMemcpy(a->i+1,ii.data().get(),A->rmap->n*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    a->nz = a->maxnz = a->i[A->rmap->n];
    ierr = PetscMalloc1(a->nz,&a->a);CHKERRQ(ierr);
    ierr = PetscMalloc1(a->nz,&a->j);CHKERRQ(ierr);
    cerr = cudaMemcpy(a->j,d_j.data().get(),a->nz*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    if (!a->ilen) { ierr = PetscMalloc1(A->rmap->n,&a->ilen);CHKERRQ(ierr); }
    if (!a->imax) { ierr = PetscMalloc1(A->rmap->n,&a->imax);CHKERRQ(ierr); }
    for (PetscInt i = 0; i < A->rmap->n; i++) {
      const PetscInt nnzr = a->i[i+1] - a->i[i];
      nzr += (PetscInt)!!(nnzr);
      a->ilen[i] = a->imax[i] = nnzr;
    }
    A->preallocated = PETSC_TRUE;
    ierr = PetscLogGpuToCpu((A->rmap->n+a->nz)*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJSetPreallocation(A,0,NULL);CHKERRQ(ierr);
  }
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(MAT_CUSPARSEPreallCOO,A,0,0,0);CHKERRQ(ierr);

  /* We want to allocate the CUSPARSE struct for matvec now.
     The code is so convoluted now that I prefer to copy garbage to the GPU */
  ierr = MatCheckCompressedRow(A,nzr,&a->compressedrow,a->i,A->rmap->n,0.6);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_CPU;
  A->nonzerostate++;
  ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  {
    matrix = (CsrMatrix*)cusp->mat->mat;
    if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing CUDA memory");
    thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
    ierr = MatSeqAIJCUSPARSEMultStruct_Destroy(&cusp->matTranspose,cusp->format);CHKERRQ(ierr);
  }

  A->offloadmask = PETSC_OFFLOAD_CPU;
  A->assembled = PETSC_FALSE;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
