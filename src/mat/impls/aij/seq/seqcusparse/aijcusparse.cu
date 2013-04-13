/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the CUSPARSE library,
*/

#include "petscconf.h"
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
#undef VecType
#include "cusparsematimpl.h"

const char *const MatCUSPARSEStorageFormats[] = {"CSR","ELL","HYB","MatCUSPARSEStorageFormat","MAT_CUSPARSE_",0};

/* this is such a hack ... but I don't know of another way to pass this variable
   from one GPU_Matrix_Ifc class to another. This is necessary for the parallel
   SpMV. Essentially, I need to use the same stream variable in two different
   data structures. I do this by creating a single instance of that stream
   and reuse it. */
cudaStream_t theBodyStream=0;

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
static PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(Mat);
static PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_cusparse"
PetscErrorCode MatFactorGetSolverPackage_seqaij_cusparse(Mat A,const MatSolverPackage *type)
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

  ./configure --download-txpetscgpu to install PETSc to use CUSPARSE

  Consult CUSPARSE documentation for more information about the matrix storage formats
  which correspond to the options database keys below.

   Options Database Keys:
.  -mat_cusparse_solve_storage_format csr - sets the storage format matrices (for factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid). Only available with 'txpetscgpu' package.

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage, MatCreateSeqAIJCUSPARSE(), MATAIJCUSPARSE, MatCreateAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqaij_cusparse"
PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_cusparse(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizes(*B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSE;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJCUSPARSE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSE Matrix Types");

  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_seqaij_cusparse);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPARSESetFormat_SeqAIJCUSPARSE"
PETSC_INTERN PetscErrorCode MatCUSPARSESetFormat_SeqAIJCUSPARSE(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT:
    cusparseMat->format = format;
    break;
  case MAT_CUSPARSE_SOLVE:
    cusparseMatSolveStorageFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseMat->format           = format;
    cusparseMatSolveStorageFormat = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. MAT_CUSPARSE_MULT, MAT_CUSPARSE_SOLVE, and MAT_CUSPARSE_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

/*@
   MatCUSPARSESetFormat - Sets the storage format of CUSPARSE matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for MPIAIJCUSPARSE matrices. This requires the txpetscgpu package. Use --download-txpetscgpu
   to build/install PETSc to use this package.

   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJCUSPARSE
.  op - MatCUSPARSEFormatOperation. SEQAIJCUSPARSE matrices support MAT_CUSPARSE_MULT, MAT_CUSPARSE_SOLVE, and MAT_CUSPARSE_ALL. MPIAIJCUSPARSE matrices support MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_OFFDIAG, and MAT_CUSPARSE_ALL.
-  format - MatCUSPARSEStorageFormat (one of MAT_CUSPARSE_CSR, MAT_CUSPARSE_ELL, MAT_CUSPARSE_HYB)

   Output Parameter:

   Level: intermediate

.seealso: MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
@*/
#undef __FUNCT__
#define __FUNCT__ "MatCUSPARSESetFormat"
PetscErrorCode MatCUSPARSESetFormat(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  ierr = PetscTryMethod(A, "MatCUSPARSESetFormat_C",(Mat,MatCUSPARSEFormatOperation,MatCUSPARSEStorageFormat),(A,op,format));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetFromOptions_SeqAIJCUSPARSE"
static PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode           ierr;
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SeqAIJCUSPARSE options");CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)A);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)MAT_CUSPARSE_CSR,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT,format);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscOptionsEnum("-mat_cusparse_solve_storage_format","sets storage format of (seq)aijcusparse gpu matrices for TriSolve",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)MAT_CUSPARSE_CSR,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_SOLVE,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of (seq)aijcusparse gpu matrices for SpMV and TriSolve",
                          "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)MAT_CUSPARSE_CSR,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJCUSPARSE"
static PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJCUSPARSE"
static PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatICCFactorSymbolic_SeqAIJCUSPARSE"
static PetscErrorCode MatICCFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatICCFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqAIJCUSPARSE"
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEBuildILULowerTriMatrix"
static PetscErrorCode MatSeqAIJCUSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  PetscInt                     n                   = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMat        = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  cusparseStatus_t             stat;
  const PetscInt               *ai = a->i,*aj = a->j,*vi;
  const MatScalar              *aa = a->a,*v;
  PetscInt                     *AiLo, *AjLo;
  PetscScalar                  *AALo;
  PetscInt                     i,nz, nzLower, offset, rowOffset;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];

      /* Allocate Space for the lower triangular matrix */
      ierr = cudaMallocHost((void**) &AiLo, (n+1)*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AjLo, nzLower*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AALo, nzLower*sizeof(PetscScalar));CHKERRCUSP(ierr);

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

        ierr = PetscMemcpy(&(AjLo[offset]), vi, nz*sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscMemcpy(&(AALo[offset]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);

        offset      += nz;
        AjLo[offset] = (PetscInt) i;
        AALo[offset] = (MatScalar) 1.0;
        offset      += 1;

        v  += nz;
        vi += nz;
      }
      cusparseMat = GPU_Matrix_Factory::getNew(MatCUSPARSEStorageFormats[cusparseTriFactors->format]);

      stat = cusparseMat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSP(stat);
      ierr = cusparseMat->setMatrix(n, n, nzLower, AiLo, AjLo, AALo);CHKERRCUSP(ierr);
      stat = cusparseMat->solveAnalysis();CHKERRCUSP(stat);

      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = cusparseMat;

      ierr = cudaFreeHost(AiLo);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AjLo);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AALo);CHKERRCUSP(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEBuildILUUpperTriMatrix"
static PetscErrorCode MatSeqAIJCUSPARSEBuildILUUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  PetscInt                     n                   = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMat        = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t             stat;
  const PetscInt               *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar              *aa = a->a,*v;
  PetscInt                     *AiUp, *AjUp;
  PetscScalar                  *AAUp;
  PetscInt                     i,nz, nzUpper, offset;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];

      /* Allocate Space for the upper triangular matrix */
      ierr = cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUSP(ierr);

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
        AAUp[offset] = 1./v[nz];
        AiUp[i]      = AiUp[i+1] - (nz+1);

        ierr = PetscMemcpy(&(AjUp[offset+1]), vi, nz*sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscMemcpy(&(AAUp[offset+1]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      cusparseMat = GPU_Matrix_Factory::getNew(MatCUSPARSEStorageFormats[cusparseTriFactors->format]);

      stat = cusparseMat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
      ierr = cusparseMat->setMatrix(n, n, nzUpper, AiUp, AjUp, AAUp);CHKERRCUSP(ierr);
      stat = cusparseMat->solveAnalysis();CHKERRCUSP(stat);

      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = cusparseMat;

      ierr = cudaFreeHost(AiUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AjUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AAUp);CHKERRCUSP(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU"
static PetscErrorCode MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           isrow = a->row,iscol = a->icol;
  PetscBool                    row_identity,col_identity;
  const PetscInt               *r,*c;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSEBuildILULowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSEBuildILUUpperTriMatrix(A);CHKERRQ(ierr);

  cusparseTriFactors->tempvec = new CUSPARRAY;
  cusparseTriFactors->tempvec->resize(n);

  A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  /*lower triangular indices */
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity) {
    ierr = cusparseTriFactors->loTriFactorPtr->setOrdIndices(r, n);CHKERRCUSP(ierr);
  }
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  /*upper triangular indices */
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity) {
    ierr = cusparseTriFactors->upTriFactorPtr->setOrdIndices(c, n);CHKERRCUSP(ierr);
  }
  ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEBuildICCTriMatrices"
static PetscErrorCode MatSeqAIJCUSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t             stat;
  PetscErrorCode               ierr;
  PetscInt                     *AiUp, *AjUp;
  PetscScalar                  *AAUp;
  PetscScalar                  *AALo;
  PetscInt                     nzUpper = a->nz,n = A->rmap->n,i,offset,nz,j;
  Mat_SeqSBAIJ                 *b = (Mat_SeqSBAIJ*)A->data;
  const PetscInt               *ai = b->i,*aj = b->j,*vj;
  const MatScalar              *aa = b->a,*v;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU) {
    try {
      /* Allocate Space for the upper triangular matrix */
      ierr = cudaMallocHost((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void**) &AALo, nzUpper*sizeof(PetscScalar));CHKERRCUSP(ierr);

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
        AAUp[offset] = 1.0/v[nz];
        AiUp[i]      = offset;
        AALo[offset] = 1.0/v[nz];

        offset+=1;
        if (nz>0) {
          ierr = PetscMemcpy(&(AjUp[offset]), vj, nz*sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscMemcpy(&(AAUp[offset]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
          for (j=offset; j<offset+nz; j++) {
            AAUp[j] = -AAUp[j];
            AALo[j] = AAUp[j]/v[nz];
          }
          offset+=nz;
        }
      }

      /* Build the upper triangular piece */
      cusparseMatUp = GPU_Matrix_Factory::getNew(MatCUSPARSEStorageFormats[cusparseTriFactors->format]);
      stat = cusparseMatUp->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSP(stat);
      ierr = cusparseMatUp->setMatrix(A->rmap->n, A->cmap->n, a->nz, AiUp, AjUp, AAUp);CHKERRCUSP(ierr);
      stat = cusparseMatUp->solveAnalysis();CHKERRCUSP(stat);
      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = cusparseMatUp;

      /* Build the lower triangular piece */
      cusparseMatLo = GPU_Matrix_Factory::getNew(MatCUSPARSEStorageFormats[cusparseTriFactors->format]);
      stat = cusparseMatLo->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
      ierr = cusparseMatLo->setMatrix(A->rmap->n, A->cmap->n, a->nz, AiUp, AjUp, AALo);CHKERRCUSP(ierr);
      stat = cusparseMatLo->solveAnalysis(CUSPARSE_OPERATION_TRANSPOSE);CHKERRCUSP(stat);
      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = cusparseMatLo;

      /* set this flag ... for performance logging */
      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->isSymmOrHerm = PETSC_TRUE;

      A->valid_GPU_matrix = PETSC_CUSP_BOTH;
      ierr = cudaFreeHost(AiUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AjUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AAUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AALo);CHKERRCUSP(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU"
static PetscErrorCode MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS                           ip = a->row;
  const PetscInt               *rip;
  PetscBool                    perm_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatSeqAIJCUSPARSEBuildICCTriMatrices(A);CHKERRQ(ierr);
  cusparseTriFactors->tempvec = new CUSPARRAY;
  cusparseTriFactors->tempvec->resize(n);
  /*lower triangular indices */
  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) {
    ierr = cusparseTriFactors->loTriFactorPtr->setOrdIndices(rip, n);CHKERRCUSP(ierr);
    ierr = cusparseTriFactors->upTriFactorPtr->setOrdIndices(rip, n);CHKERRCUSP(ierr);
  }
  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJCUSPARSE"
static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row,iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJCUSPARSEILUAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqAIJCUSPARSE"
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);

  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity) {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJCUSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJCUSPARSE;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJCUSPARSEICCAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEAnalyzeTransposeForSolve"
static PetscErrorCode MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t             stat;

  PetscFunctionBegin;
  stat = cusparseMatLo->initializeCusparseMatTranspose(MAT_cusparseHandle,
                                                       CUSPARSE_MATRIX_TYPE_TRIANGULAR,
                                                       CUSPARSE_FILL_MODE_UPPER,
                                                       CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSP(stat);
  stat = cusparseMatLo->solveAnalysisTranspose();CHKERRCUSP(stat);

  stat = cusparseMatUp->initializeCusparseMatTranspose(MAT_cusparseHandle,
                                                       CUSPARSE_MATRIX_TYPE_TRIANGULAR,
                                                       CUSPARSE_FILL_MODE_LOWER,
                                                       CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
  stat = cusparseMatUp->solveAnalysisTranspose();CHKERRCUSP(stat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSEGenerateTransposeForMult"
static PetscErrorCode MatSeqAIJCUSPARSEGenerateTransposeForMult(Mat A)
{
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  cusparseStatus_t   stat;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (cusparseMat->isSymmOrHerm) {
    stat = cusparseMat->mat->initializeCusparseMatTranspose(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
  } else {
    stat = cusparseMat->mat->initializeCusparseMatTranspose(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
  }
  ierr = cusparseMat->mat->setMatrixTranspose(A->rmap->n, A->cmap->n, a->nz, a->i, a->j, a->a);CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_SeqAIJCUSPARSE"
static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  CUSPARRAY                    *xGPU, *bGPU;
  cusparseStatus_t             stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY                    *tempGPU            = (CUSPARRAY*) cusparseTriFactors->tempvec;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  /* Analyze the matrix ... on the fly */
  if (!cusparseTriFactors->hasTranspose) {
    ierr = MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    cusparseTriFactors->hasTranspose=PETSC_TRUE;
  }

  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* solve with reordering */
  ierr = cusparseMatUp->reorderIn(xGPU, bGPU);CHKERRCUSP(ierr);
  stat = cusparseMatUp->solveTranspose(xGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatLo->solveTranspose(tempGPU, xGPU);CHKERRCUSP(stat);
  ierr = cusparseMatLo->reorderOut(xGPU);CHKERRCUSP(ierr);

  /* restore */
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);

  if (cusparseTriFactors->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*A->cmap->n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering"
static PetscErrorCode MatSolveTranspose_SeqAIJCUSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  CUSPARRAY                    *xGPU,*bGPU;
  cusparseStatus_t             stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY                    *tempGPU            = (CUSPARRAY*) cusparseTriFactors->tempvec;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  /* Analyze the matrix ... on the fly */
  if (!cusparseTriFactors->hasTranspose) {
    ierr = MatSeqAIJCUSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    cusparseTriFactors->hasTranspose=PETSC_TRUE;
  }

  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* solve */
  stat = cusparseMatUp->solveTranspose(bGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatLo->solveTranspose(tempGPU, xGPU);CHKERRCUSP(stat);

  /* restore */
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (cusparseTriFactors->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*A->cmap->n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqAIJCUSPARSE"
static PetscErrorCode MatSolve_SeqAIJCUSPARSE(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  CUSPARRAY                    *xGPU,*bGPU;
  cusparseStatus_t             stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY                    *tempGPU            = (CUSPARRAY*)cusparseTriFactors->tempvec;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* solve with reordering */
  ierr = cusparseMatLo->reorderIn(xGPU, bGPU);CHKERRCUSP(ierr);
  stat = cusparseMatLo->solve(xGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatUp->solve(tempGPU, xGPU);CHKERRCUSP(stat);
  ierr = cusparseMatUp->reorderOut(xGPU);CHKERRCUSP(ierr);

  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (cusparseTriFactors->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*A->cmap->n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqAIJCUSPARSE_NaturalOrdering"
static PetscErrorCode MatSolve_SeqAIJCUSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  CUSPARRAY                    *xGPU,*bGPU;
  cusparseStatus_t             stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY                    *tempGPU            = (CUSPARRAY*)cusparseTriFactors->tempvec;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  /* solve */
  stat = cusparseMatLo->solve(bGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatUp->solve(tempGPU, xGPU);CHKERRCUSP(stat);

  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (cusparseTriFactors->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*A->cmap->n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJCUSPARSECopyToGPU"
static PetscErrorCode MatSeqAIJCUSPARSECopyToGPU(Mat A)
{

  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           m = A->rmap->n,*ii,*ridx;
  PetscBool          symmetryTest=PETSC_FALSE, hermitianTest=PETSC_FALSE;
  PetscBool          symmetryOptionIsSet=PETSC_FALSE, symmetryOptionTest=PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU) {
    ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
    /*
      It may be possible to reuse nonzero structure with new matrix values but
      for simplicity and insured correctness we delete and build a new matrix on
      the GPU. Likely a very small performance hit.
    */
    if (cusparseMat->mat) {
      try {
        delete cusparseMat->mat;
        if (cusparseMat->tempvec) delete cusparseMat->tempvec;

      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
      }
    }
    try {
      cusparseMat->nonzerorow=0;
      for (int j = 0; j<m; j++) cusparseMat->nonzerorow += ((a->i[j+1]-a->i[j])>0);

      if (a->compressedrow.use) {
        m    = a->compressedrow.nrows;
        ii   = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
      } else {
        /* Forcing compressed row on the GPU ... only relevant for CSR storage */
        int k=0;
        ierr = PetscMalloc((cusparseMat->nonzerorow+1)*sizeof(PetscInt), &ii);CHKERRQ(ierr);
        ierr = PetscMalloc((cusparseMat->nonzerorow)*sizeof(PetscInt), &ridx);CHKERRQ(ierr);
        ii[0]=0;
        for (int j = 0; j<m; j++) {
          if ((a->i[j+1]-a->i[j])>0) {
            ii[k]  = a->i[j];
            ridx[k]= j;
            k++;
          }
        }
        ii[cusparseMat->nonzerorow] = a->nz;

        m = cusparseMat->nonzerorow;
      }

      /* Build our matrix ... first determine the GPU storage type */
      cusparseMat->mat = GPU_Matrix_Factory::getNew(MatCUSPARSEStorageFormats[cusparseMat->format]);

      /* Create the streams and events (if desired).  */
      PetscMPIInt size;
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
      ierr = cusparseMat->mat->buildStreamsAndEvents(size, &theBodyStream);CHKERRCUSP(ierr);

      ierr = MatIsSymmetricKnown(A,&symmetryOptionIsSet,&symmetryOptionTest);CHKERRQ(ierr);
      if ((symmetryOptionIsSet && !symmetryOptionTest) || !symmetryOptionIsSet) {
	/* CUSPARSE_FILL_MODE_UPPER and CUSPARSE_DIAG_TYPE_NON_UNIT are irrelevant here */
        cusparseStatus_t stat = cusparseMat->mat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
        cusparseMat->isSymmOrHerm = PETSC_FALSE;
      } else {
        ierr = MatIsSymmetric(A,0.0,&symmetryTest);CHKERRQ(ierr);
        if (symmetryTest) {
          cusparseStatus_t stat = cusparseMat->mat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
          cusparseMat->isSymmOrHerm = PETSC_TRUE;
        } else {
          ierr = MatIsHermitian(A,0.0,&hermitianTest);CHKERRQ(ierr);
          if (hermitianTest) {
            cusparseStatus_t stat = cusparseMat->mat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
            cusparseMat->isSymmOrHerm = PETSC_TRUE;
          } else {
            /* CUSPARSE_FILL_MODE_UPPER and CUSPARSE_DIAG_TYPE_NON_UNIT are irrelevant here */
            cusparseStatus_t stat = cusparseMat->mat->initializeCusparseMat(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
            cusparseMat->isSymmOrHerm = PETSC_FALSE;
          }
        }
      }

      /* lastly, build the matrix */
      ierr = cusparseMat->mat->setMatrix(m, A->cmap->n, a->nz, ii, a->j, a->a);CHKERRCUSP(ierr);
      cusparseMat->mat->setCPRowIndices(ridx, m);
      if (!a->compressedrow.use) {
        ierr = PetscFree(ii);CHKERRQ(ierr);
        ierr = PetscFree(ridx);CHKERRQ(ierr);
      }
      cusparseMat->tempvec = new CUSPARRAY;
      cusparseMat->tempvec->resize(m);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);

    A->valid_GPU_matrix = PETSC_CUSP_BOTH;

    ierr = PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetVecs_SeqAIJCUSPARSE"
static PetscErrorCode MatGetVecs_SeqAIJCUSPARSE(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (right) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJCUSPARSE"
static PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CUSPARRAY          *xarray,*yarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
     ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr); */
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
    ierr = cusparseMat->mat->multiply(xarray, yarray);CHKERRCUSP(ierr);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (!cusparseMat->mat->hasNonZeroStream()) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
  if (cusparseMat->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*cusparseMat->nonzerorow);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - cusparseMat->nonzerorow);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqAIJCUSPARSE"
static PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CUSPARRAY          *xarray,*yarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
     ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr); */
  if (!cusparseMat->hasTranspose) {
    ierr = MatSeqAIJCUSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    cusparseMat->hasTranspose=PETSC_TRUE;
  }
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
    ierr = cusparseMat->mat->multiplyTranspose(xarray, yarray);CHKERRCUSP(ierr);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (!cusparseMat->mat->hasNonZeroStream()) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
  if (cusparseMat->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 3.0*cusparseMat->nonzerorow);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz - cusparseMat->nonzerorow);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqAIJCUSPARSE"
static PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CUSPARRAY          *xarray,*yarray,*zarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
     ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr); */
  try {
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);

    /* multiply add */
    ierr = cusparseMat->mat->multiplyAdd(xarray, zarray);CHKERRCUSP(ierr);

    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);

  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (cusparseMat->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 2.0*cusparseMat->nonzerorow);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_SeqAIJCUSPARSE"
static PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;
  CUSPARRAY          *xarray,*yarray,*zarray;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
     ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr); */
  if (!cusparseMat->hasTranspose) {
    ierr = MatSeqAIJCUSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    cusparseMat->hasTranspose=PETSC_TRUE;
  }
  try {
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);

    /* multiply add with matrix transpose */
    ierr = cusparseMat->mat->multiplyAddTranspose(xarray, yarray);CHKERRCUSP(ierr);

    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);

  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (cusparseMat->isSymmOrHerm) {
    ierr = PetscLogFlops(4.0*a->nz - 2.0*cusparseMat->nonzerorow);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSPARSE"
static PetscErrorCode MatAssemblyEnd_SeqAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  }
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  A->ops->mult             = MatMult_SeqAIJCUSPARSE;
  A->ops->multadd          = MatMultAdd_SeqAIJCUSPARSE;
  A->ops->multtranspose    = MatMultTranspose_SeqAIJCUSPARSE;
  A->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqAIJCUSPARSE"
/*@
   MatCreateSeqAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

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

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSPARSE"
static PetscErrorCode MatDestroy_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    try {
      if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED) {
        delete (GPU_Matrix_Ifc*)(cusparseMat->mat);
      }
      if (cusparseMat->tempvec!=0) delete cusparseMat->tempvec;
      delete cusparseMat;
      A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  } else {
    /* The triangular factors */
    try {
      Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
      GPU_Matrix_Ifc               *cusparseMatLo      = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
      GPU_Matrix_Ifc               *cusparseMatUp      = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
      delete (GPU_Matrix_Ifc*) cusparseMatLo;
      delete (GPU_Matrix_Ifc*) cusparseMatUp;
      delete (CUSPARRAY*) cusparseTriFactors->tempvec;
      delete cusparseTriFactors;
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  if (MAT_cusparseHandle) {
    cusparseStatus_t stat;
    stat = cusparseDestroy(MAT_cusparseHandle);CHKERRCUSP(stat);

    MAT_cusparseHandle=0;
  }
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJCUSPARSE"
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  if (B->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.
       now build a GPU matrix data structure */
    B->spptr = new Mat_SeqAIJCUSPARSE;
    ((Mat_SeqAIJCUSPARSE*)B->spptr)->mat          = 0;
    ((Mat_SeqAIJCUSPARSE*)B->spptr)->tempvec      = 0;
    ((Mat_SeqAIJCUSPARSE*)B->spptr)->format       = MAT_CUSPARSE_CSR;
    ((Mat_SeqAIJCUSPARSE*)B->spptr)->hasTranspose = PETSC_FALSE;
    ((Mat_SeqAIJCUSPARSE*)B->spptr)->isSymmOrHerm = PETSC_FALSE;
  } else {
    /* NEXT, set the pointers to the triangular factors */
    B->spptr = new Mat_SeqAIJCUSPARSETriFactors;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->loTriFactorPtr = 0;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->upTriFactorPtr = 0;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->tempvec        = 0;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->format         = cusparseMatSolveStorageFormat;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->hasTranspose   = PETSC_FALSE;
    ((Mat_SeqAIJCUSPARSETriFactors*)B->spptr)->isSymmOrHerm   = PETSC_FALSE;
  }
  /* Create a single instance of the MAT_cusparseHandle for any matrix (matMult, TriSolve, ...) */
  if (!MAT_cusparseHandle) {
    cusparseStatus_t stat;
    stat = cusparseCreate(&MAT_cusparseHandle);CHKERRCUSP(stat);
  }
  /* Here we overload MatGetFactor_petsc_C which enables -mat_type aijcusparse to use the
     default cusparse tri solve. Note the difference with the implementation in
     MatCreate_SeqAIJCUSP in ../seqcusp/aijcusp.cu */
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatGetFactor_petsc_C",MatGetFactor_seqaij_cusparse);CHKERRQ(ierr);

  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJCUSPARSE;
  B->ops->destroy          = MatDestroy_SeqAIJCUSPARSE;
  B->ops->getvecs          = MatGetVecs_SeqAIJCUSPARSE;
  B->ops->setfromoptions   = MatSetFromOptions_SeqAIJCUSPARSE;
  B->ops->mult             = MatMult_SeqAIJCUSPARSE;
  B->ops->multadd          = MatMultAdd_SeqAIJCUSPARSE;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJCUSPARSE;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJCUSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);

  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;

  ierr = PetscObjectComposeFunction((PetscObject)B, "MatCUSPARSESetFormat_C", MatCUSPARSESetFormat_SeqAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*M
   MATSEQAIJCUSPARSE - MATAIJCUSPARSE = "(seq)aijcusparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. All matrix calculations are performed on Nvidia GPUs using
   the CUSPARSE library. This type is only available when using the 'txpetscgpu' package.
   Use --download-txpetscgpu to build/install PETSc to use different CUSPARSE library and
   the different GPU storage formats.

   Options Database Keys:
+  -mat_type aijcusparse - sets the matrix type to "seqaijcusparse" during a call to MatSetFromOptions()
.  -mat_cusparse_storage_format csr - sets the storage format of matrices (for MatMult and factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid). Only available with 'txpetscgpu' package.
.  -mat_cusparse_mult_storage_format csr - sets the storage format of matrices (for MatMult) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid). Only available with 'txpetscgpu' package.
-  -mat_cusparse_solve_storage_format csr - sets the storage format matrices (for factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid). Only available with 'txpetscgpu' package.

  Level: beginner

.seealso: MatCreateSeqAIJCUSPARSE(), MATAIJCUSPARSE, MatCreateAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/
