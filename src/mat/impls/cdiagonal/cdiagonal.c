
#include <petsc/private/matimpl.h>  /*I "petscmat.h" I*/

typedef struct {
  PetscScalar diag;
} Mat_ConstantDiagonal;


/* ----------------------------------------------------------------------------------------*/
static PetscErrorCode MatDestroy_ConstantDiagonal(Mat mat)
{
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_ConstantDiagonal(Mat J,PetscViewer viewer)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;
  PetscBool            iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat    format;

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"Diagonal value: %g\n",ctx->diag);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"Diagonal value: %g + i %g\n",PetscRealPart(ctx->diag),PetscImaginaryPart(ctx->diag));CHKERRQ(ierr);
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
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;

  PetscFunctionBegin;
  ierr = VecAXPBY(y,ctx->diag,0.0,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_ConstantDiagonal(Mat J,Vec x)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)J->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecSet(x,ctx->diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_ConstantDiagonal(Mat Y,PetscScalar a)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  if (a != 0.) {ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);}
  ctx->diag += a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ConstantDiagonal(Mat Y,PetscScalar a)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  if (a != 1.) {ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);}
  ctx->diag *= a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_ConstantDiagonal(Mat Y)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)Y->data;

  PetscFunctionBegin;
  if (ctx->diag != 0.0) {ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);}
  ctx->diag = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_ConstantDiagonal(Mat matin,Vec x,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec y)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx  = (Mat_ConstantDiagonal*)matin->data;

  PetscFunctionBegin;
  if (ctx->diag == 0.0) matin->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else matin->factorerrortype = MAT_FACTOR_NOERROR;
  ierr = VecAXPBY(y,1.0/ctx->diag,0.0,x);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATCONSTANTDIAGONAL);CHKERRQ(ierr);
  ierr = MatShift(*J,diag);CHKERRQ(ierr);
  ierr = MatSetUp(*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  MatCreate_ConstantDiagonal(Mat A)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *ctx;

  PetscFunctionBegin;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->diag = 0.0;
  A->data   = (void*)ctx;

  A->assembled        = PETSC_TRUE;
  A->preallocated     = PETSC_TRUE;
  A->ops->mult        = MatMult_ConstantDiagonal;
  A->ops->sor         = MatSOR_ConstantDiagonal;
  A->ops->shift       = MatShift_ConstantDiagonal;
  A->ops->scale       = MatScale_ConstantDiagonal;
  A->ops->getdiagonal = MatGetDiagonal_ConstantDiagonal;
  A->ops->view        = MatView_ConstantDiagonal;
  A->ops->zeroentries = MatZeroEntries_ConstantDiagonal;
  A->ops->assemblyend = MatAssemblyEnd_ConstantDiagonal;
  A->ops->destroy     = MatDestroy_ConstantDiagonal;
  A->ops->getinfo     = MatGetInfo_ConstantDiagonal;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATCONSTANTDIAGONAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorNumeric_ConstantDiagonal(Mat fact,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode       ierr;
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal*)A->data,*fctx = (Mat_ConstantDiagonal*)fact->data;

  PetscFunctionBegin;
  if (actx->diag == 0.0) fact->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else fact->factorerrortype = MAT_FACTOR_NOERROR;
  fctx->diag = 1.0/actx->diag;
  ierr = PetscObjectStateIncrease((PetscObject)fact);CHKERRQ(ierr);
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
  fact->ops->choleskyfactornumeric = MatFactorNumeric_ConstantDiagonal;  PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_constantdiagonal_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n, N = A->rmap->N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateConstantDiagonal(PetscObjectComm((PetscObject)A),n,n,N,N,0,B);CHKERRQ(ierr);

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

  ierr = PetscFree((*B)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&(*B)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
