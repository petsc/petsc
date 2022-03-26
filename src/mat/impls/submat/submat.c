
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  IS          isrow,iscol;      /* rows and columns in submatrix, only used to check consistency */
  Vec         lwork,rwork;      /* work vectors inside the scatters */
  Vec         lwork2,rwork2;    /* work vectors inside the scatters */
  VecScatter  lrestrict,rprolong;
  Mat         A;
} Mat_SubVirtual;

static PetscErrorCode MatScale_SubMatrix(Mat N,PetscScalar a)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(MatScale(Na->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_SubMatrix(Mat N,PetscScalar a)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(MatShift(Na->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SubMatrix(Mat N,Vec left,Vec right)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  if (right) {
    PetscCall(VecZeroEntries(Na->rwork));
    PetscCall(VecScatterBegin(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (left) {
    PetscCall(VecZeroEntries(Na->lwork));
    PetscCall(VecScatterBegin(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscCall(MatDiagonalScale(Na->A,left ? Na->lwork : NULL,right ? Na->rwork : NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SubMatrix(Mat N,Vec d)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonal(Na->A,Na->rwork));
  PetscCall(VecScatterBegin(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(Na->rwork));
  PetscCall(VecScatterBegin(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(MatMult(Na->A,Na->rwork,Na->lwork));
  PetscCall(VecScatterBegin(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(Na->rwork));
  PetscCall(VecScatterBegin(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  if (v1 == v2) {
    PetscCall(MatMultAdd(Na->A,Na->rwork,Na->rwork,Na->lwork));
  } else if (v2 == v3) {
    PetscCall(VecZeroEntries(Na->lwork));
    PetscCall(VecScatterBegin(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(MatMultAdd(Na->A,Na->rwork,Na->lwork,Na->lwork));
  } else {
    if (!Na->lwork2) {
      PetscCall(VecDuplicate(Na->lwork,&Na->lwork2));
    } else {
      PetscCall(VecZeroEntries(Na->lwork2));
    }
    PetscCall(VecScatterBegin(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(MatMultAdd(Na->A,Na->rwork,Na->lwork2,Na->lwork));
  }
  PetscCall(VecScatterBegin(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(Na->lwork));
  PetscCall(VecScatterBegin(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(MatMultTranspose(Na->A,Na->lwork,Na->rwork));
  PetscCall(VecScatterBegin(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(Na->lwork));
  PetscCall(VecScatterBegin(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  if (v1 == v2) {
    PetscCall(MatMultTransposeAdd(Na->A,Na->lwork,Na->lwork,Na->rwork));
  } else if (v2 == v3) {
    PetscCall(VecZeroEntries(Na->rwork));
    PetscCall(VecScatterBegin(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork,Na->rwork));
  } else {
    if (!Na->rwork2) {
      PetscCall(VecDuplicate(Na->rwork,&Na->rwork2));
    } else {
      PetscCall(VecZeroEntries(Na->rwork2));
    }
    PetscCall(VecScatterBegin(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork2,Na->rwork));
  }
  PetscCall(VecScatterBegin(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SubMatrix(Mat N)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&Na->isrow));
  PetscCall(ISDestroy(&Na->iscol));
  PetscCall(VecDestroy(&Na->lwork));
  PetscCall(VecDestroy(&Na->rwork));
  PetscCall(VecDestroy(&Na->lwork2));
  PetscCall(VecDestroy(&Na->rwork2));
  PetscCall(VecScatterDestroy(&Na->lrestrict));
  PetscCall(VecScatterDestroy(&Na->rprolong));
  PetscCall(MatDestroy(&Na->A));
  PetscCall(PetscFree(N->data));
  PetscFunctionReturn(0);
}

/*@
   MatCreateSubMatrixVirtual - Creates a virtual matrix that acts as a submatrix

   Collective on Mat

   Input Parameters:
+  A - matrix that we will extract a submatrix of
.  isrow - rows to be present in the submatrix
-  iscol - columns to be present in the submatrix

   Output Parameters:
.  newmat - new matrix

   Level: developer

   Notes:
   Most will use MatCreateSubMatrix which provides a more efficient representation if it is available.

.seealso: MatCreateSubMatrix(), MatSubMatrixVirtualUpdate()
@*/
PetscErrorCode MatCreateSubMatrixVirtual(Mat A,IS isrow,IS iscol,Mat *newmat)
{
  Vec            left,right;
  PetscInt       m,n;
  Mat            N;
  Mat_SubVirtual *Na;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscValidPointer(newmat,4);
  *newmat = NULL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&N));
  PetscCall(ISGetLocalSize(isrow,&m));
  PetscCall(ISGetLocalSize(iscol,&n));
  PetscCall(MatSetSizes(N,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)N,MATSUBMATRIX));

  PetscCall(PetscNewLog(N,&Na));
  N->data   = (void*)Na;

  PetscCall(PetscObjectReference((PetscObject)isrow));
  PetscCall(PetscObjectReference((PetscObject)iscol));
  Na->isrow = isrow;
  Na->iscol = iscol;

  PetscCall(PetscFree(N->defaultvectype));
  PetscCall(PetscStrallocpy(A->defaultvectype,&N->defaultvectype));
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  PetscCall(MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A));

  N->ops->destroy          = MatDestroy_SubMatrix;
  N->ops->mult             = MatMult_SubMatrix;
  N->ops->multadd          = MatMultAdd_SubMatrix;
  N->ops->multtranspose    = MatMultTranspose_SubMatrix;
  N->ops->multtransposeadd = MatMultTransposeAdd_SubMatrix;
  N->ops->scale            = MatScale_SubMatrix;
  N->ops->diagonalscale    = MatDiagonalScale_SubMatrix;
  N->ops->shift            = MatShift_SubMatrix;
  N->ops->convert          = MatConvert_Shell;
  N->ops->getdiagonal      = MatGetDiagonal_SubMatrix;

  PetscCall(MatSetBlockSizesFromMats(N,A,A));
  PetscCall(PetscLayoutSetUp(N->rmap));
  PetscCall(PetscLayoutSetUp(N->cmap));

  PetscCall(MatCreateVecs(A,&Na->rwork,&Na->lwork));
  PetscCall(MatCreateVecs(N,&right,&left));
  PetscCall(VecScatterCreate(Na->lwork,isrow,left,NULL,&Na->lrestrict));
  PetscCall(VecScatterCreate(right,NULL,Na->rwork,iscol,&Na->rprolong));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscCall(MatSetUp(N));

  N->assembled = PETSC_TRUE;
  *newmat      = N;
  PetscFunctionReturn(0);
}

/*@
   MatSubMatrixVirtualUpdate - Updates a submatrix

   Collective on Mat

   Input Parameters:
+  N - submatrix to update
.  A - full matrix in the submatrix
.  isrow - rows in the update (same as the first time the submatrix was created)
-  iscol - columns in the update (same as the first time the submatrix was created)

   Level: developer

   Notes:
   Most will use MatCreateSubMatrix which provides a more efficient representation if it is available.

.seealso: MatCreateSubMatrixVirtual()
@*/
PetscErrorCode  MatSubMatrixVirtualUpdate(Mat N,Mat A,IS isrow,IS iscol)
{
  PetscBool      flg;
  Mat_SubVirtual *Na;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(N,MAT_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,3);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,4);
  PetscCall(PetscObjectTypeCompare((PetscObject)N,MATSUBMATRIX,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix has wrong type");

  Na   = (Mat_SubVirtual*)N->data;
  PetscCall(ISEqual(isrow,Na->isrow,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different row indices");
  PetscCall(ISEqual(iscol,Na->iscol,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different column indices");

  PetscCall(PetscFree(N->defaultvectype));
  PetscCall(PetscStrallocpy(A->defaultvectype,&N->defaultvectype));
  PetscCall(MatDestroy(&Na->A));
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  PetscCall(MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A));
  PetscFunctionReturn(0);
}
