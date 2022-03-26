
#include <petsc/private/matimpl.h>  /*I "petscmat.h" I*/

typedef struct {
  PetscScalar diag;
} Mat_ConstantDiagonal;

static PetscErrorCode MatAXPY_ConstantDiagonal(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_ConstantDiagonal *yctx = (Mat_ConstantDiagonal*)Y->data;
  Mat_ConstantDiagonal *xctx = (Mat_ConstantDiagonal*)X->data;

  PetscFunctionBegin;
  yctx->diag += a*xctx->diag;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)A->data;

  PetscFunctionBegin;
  if (ncols) *ncols = 1;
  if (cols) {
    PetscCall(PetscMalloc1(1,cols));
    (*cols)[0] = row;
  }
  if (vals) {
    PetscCall(PetscMalloc1(1,vals));
    (*vals)[0] = ctx->diag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  PetscFunctionBegin;
  if (ncols) *ncols = 0;
  if (cols) {
    PetscCall(PetscFree(*cols));
  }
  if (vals) {
    PetscCall(PetscFree(*vals));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ConstantDiagonal(Mat A, Vec x, Vec y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)A->data;

  PetscFunctionBegin;
  PetscCall(VecAXPBY(y,ctx->diag,0.0,x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_ConstantDiagonal(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) {
    PetscCall(VecAXPBY(v3,ctx->diag,1.0,v1));
  } else {
    PetscCall(VecAXPBYPCZ(v3,ctx->diag,1.0,0.0,v1,v2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_ConstantDiagonal(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) {
    PetscCall(VecAXPBY(v3,ctx->diag,1.0,v1));
  } else {
    PetscCall(VecAXPBYPCZ(v3,ctx->diag,1.0,0.0,v1,v2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_ConstantDiagonal(Mat A,NormType type,PetscReal *nrm)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)A->data;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS || type == NORM_2 || type == NORM_1 || type == NORM_INFINITY) *nrm = PetscAbsScalar(ctx->diag);
  else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported norm");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_ConstantDiagonal(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])

{
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatCreateSubMatrices(B,n,irow,icol,scall,submat));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_ConstantDiagonal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal*)A->data;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(*B,A,A));
  PetscCall(MatSetType(*B,MATCONSTANTDIAGONAL));
  PetscCall(PetscLayoutReference(A->rmap,&(*B)->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&(*B)->cmap));
  if (op == MAT_COPY_VALUES) {
    Mat_ConstantDiagonal *bctx = (Mat_ConstantDiagonal*)(*B)->data;
    bctx->diag = actx->diag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_ConstantDiagonal(Mat mat,PetscBool *missing,PetscInt *dd)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ConstantDiagonal(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mat->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_ConstantDiagonal(Mat J,PetscViewer viewer)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;
  PetscBool            iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat    format;

    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer,"Diagonal value: %g + i %g\n",(double)PetscRealPart(ctx->diag),(double)PetscImaginaryPart(ctx->diag)));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer,"Diagonal value: %g\n",(double)(ctx->diag)));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_ConstantDiagonal(Mat J,MatAssemblyType mt)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_ConstantDiagonal(Mat J,Vec x,Vec y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;

  PetscFunctionBegin;
  PetscCall(VecAXPBY(y,ctx->diag,0.0,x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_ConstantDiagonal(Mat J,Vec x)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;

  PetscFunctionBegin;
  PetscCall(VecSet(x,ctx->diag));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_ConstantDiagonal(Mat Y,PetscScalar a)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  ctx->diag += a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ConstantDiagonal(Mat Y,PetscScalar a)
{
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  ctx->diag *= a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_ConstantDiagonal(Mat Y)
{
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  ctx->diag = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_ConstantDiagonal(Mat matin,Vec x,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec y)
{
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)matin->data;

  PetscFunctionBegin;
  if (ctx->diag == 0.0) matin->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else matin->factorerrortype = MAT_FACTOR_NOERROR;
  PetscCall(VecAXPBY(y,1.0/ctx->diag,0.0,x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_ConstantDiagonal(Mat A,MatInfoType flag,MatInfo *info)
{
  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = 1.0;
  info->nz_used      = 1.0;
  info->nz_unneeded  = 0.0;
  info->assemblies   = A->num_ass;
  info->mallocs      = 0.0;
  info->memory       = ((PetscObject)A)->mem;
  if (A->factortype) {
    info->fill_ratio_given  = 1.0;
    info->fill_ratio_needed = 1.0;
    info->factor_mallocs    = 0.0;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   MatCreateConstantDiagonal - Creates a matrix with a uniform value along the diagonal

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
-  diag - the diagonal value

   Output Parameter:
.  J - the diagonal matrix

   Level: advanced

   Notes:
    Only supports square matrices with the same number of local rows and columns

.seealso: MatDestroy(), MATCONSTANTDIAGONAL, MatScale(), MatShift(), MatMult(), MatGetDiagonal(), MatGetFactor(), MatSolve()

@*/
PetscErrorCode  MatCreateConstantDiagonal(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar diag,Mat *J)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,J));
  PetscCall(MatSetSizes(*J,m,n,M,N));
  PetscCall(MatSetType(*J,MATCONSTANTDIAGONAL));
  PetscCall(MatShift(*J,diag));
  PetscCall(MatSetUp(*J));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  MatCreate_ConstantDiagonal(Mat A)
{
  Mat_ConstantDiagonal *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->diag = 0.0;
  A->data   = (void*)ctx;

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;

  A->ops->mult             = MatMult_ConstantDiagonal;
  A->ops->multadd          = MatMultAdd_ConstantDiagonal;
  A->ops->multtranspose    = MatMultTranspose_ConstantDiagonal;
  A->ops->multtransposeadd = MatMultTransposeAdd_ConstantDiagonal;
  A->ops->norm             = MatNorm_ConstantDiagonal;
  A->ops->createsubmatrices= MatCreateSubMatrices_ConstantDiagonal;
  A->ops->duplicate        = MatDuplicate_ConstantDiagonal;
  A->ops->missingdiagonal  = MatMissingDiagonal_ConstantDiagonal;
  A->ops->getrow           = MatGetRow_ConstantDiagonal;
  A->ops->restorerow       = MatRestoreRow_ConstantDiagonal;
  A->ops->sor              = MatSOR_ConstantDiagonal;
  A->ops->shift            = MatShift_ConstantDiagonal;
  A->ops->scale            = MatScale_ConstantDiagonal;
  A->ops->getdiagonal      = MatGetDiagonal_ConstantDiagonal;
  A->ops->view             = MatView_ConstantDiagonal;
  A->ops->zeroentries      = MatZeroEntries_ConstantDiagonal;
  A->ops->assemblyend      = MatAssemblyEnd_ConstantDiagonal;
  A->ops->destroy          = MatDestroy_ConstantDiagonal;
  A->ops->getinfo          = MatGetInfo_ConstantDiagonal;
  A->ops->axpy             = MatAXPY_ConstantDiagonal;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATCONSTANTDIAGONAL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorNumeric_ConstantDiagonal(Mat fact,Mat A,const MatFactorInfo *info)
{
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal*)A->data,*fctx = (Mat_ConstantDiagonal*)fact->data;

  PetscFunctionBegin;
  if (actx->diag == 0.0) fact->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else fact->factorerrortype = MAT_FACTOR_NOERROR;
  fctx->diag = 1.0/actx->diag;
  fact->ops->solve = MatMult_ConstantDiagonal;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorSymbolic_LU_ConstantDiagonal(Mat fact,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->ops->lufactornumeric = MatFactorNumeric_ConstantDiagonal;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorSymbolic_Cholesky_ConstantDiagonal(Mat fact,Mat A,IS isrow,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->ops->choleskyfactornumeric = MatFactorNumeric_ConstantDiagonal;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_constantdiagonal_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n, N = A->rmap->N;

  PetscFunctionBegin;
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)A),n,n,N,N,0,B));

  (*B)->factortype = ftype;
  (*B)->ops->ilufactorsymbolic      = MatFactorSymbolic_LU_ConstantDiagonal;
  (*B)->ops->lufactorsymbolic       = MatFactorSymbolic_LU_ConstantDiagonal;
  (*B)->ops->iccfactorsymbolic      = MatFactorSymbolic_Cholesky_ConstantDiagonal;
  (*B)->ops->choleskyfactorsymbolic = MatFactorSymbolic_Cholesky_ConstantDiagonal;

  (*B)->ops->shift       = NULL;
  (*B)->ops->scale       = NULL;
  (*B)->ops->mult        = NULL;
  (*B)->ops->sor         = NULL;
  (*B)->ops->zeroentries = NULL;

  PetscCall(PetscFree((*B)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&(*B)->solvertype));
  PetscFunctionReturn(0);
}
