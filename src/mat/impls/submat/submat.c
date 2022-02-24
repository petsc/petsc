
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
  CHKERRQ(MatScale(Na->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_SubMatrix(Mat N,PetscScalar a)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatShift(Na->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SubMatrix(Mat N,Vec left,Vec right)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  if (right) {
    CHKERRQ(VecZeroEntries(Na->rwork));
    CHKERRQ(VecScatterBegin(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (left) {
    CHKERRQ(VecZeroEntries(Na->lwork));
    CHKERRQ(VecScatterBegin(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  }
  CHKERRQ(MatDiagonalScale(Na->A,left ? Na->lwork : NULL,right ? Na->rwork : NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SubMatrix(Mat N,Vec d)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatGetDiagonal(Na->A,Na->rwork));
  CHKERRQ(VecScatterBegin(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(VecZeroEntries(Na->rwork));
  CHKERRQ(VecScatterBegin(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(MatMult(Na->A,Na->rwork,Na->lwork));
  CHKERRQ(VecScatterBegin(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(VecZeroEntries(Na->rwork));
  CHKERRQ(VecScatterBegin(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
  if (v1 == v2) {
    CHKERRQ(MatMultAdd(Na->A,Na->rwork,Na->rwork,Na->lwork));
  } else if (v2 == v3) {
    CHKERRQ(VecZeroEntries(Na->lwork));
    CHKERRQ(VecScatterBegin(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(MatMultAdd(Na->A,Na->rwork,Na->lwork,Na->lwork));
  } else {
    if (!Na->lwork2) {
      CHKERRQ(VecDuplicate(Na->lwork,&Na->lwork2));
    } else {
      CHKERRQ(VecZeroEntries(Na->lwork2));
    }
    CHKERRQ(VecScatterBegin(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(MatMultAdd(Na->A,Na->rwork,Na->lwork2,Na->lwork));
  }
  CHKERRQ(VecScatterBegin(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(VecZeroEntries(Na->lwork));
  CHKERRQ(VecScatterBegin(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(Na->A,Na->lwork,Na->rwork));
  CHKERRQ(VecScatterBegin(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(VecZeroEntries(Na->lwork));
  CHKERRQ(VecScatterBegin(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE));
  if (v1 == v2) {
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->lwork,Na->lwork,Na->rwork));
  } else if (v2 == v3) {
    CHKERRQ(VecZeroEntries(Na->rwork));
    CHKERRQ(VecScatterBegin(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork,Na->rwork));
  } else {
    if (!Na->rwork2) {
      CHKERRQ(VecDuplicate(Na->rwork,&Na->rwork2));
    } else {
      CHKERRQ(VecZeroEntries(Na->rwork2));
    }
    CHKERRQ(VecScatterBegin(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork2,Na->rwork));
  }
  CHKERRQ(VecScatterBegin(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SubMatrix(Mat N)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&Na->isrow));
  CHKERRQ(ISDestroy(&Na->iscol));
  CHKERRQ(VecDestroy(&Na->lwork));
  CHKERRQ(VecDestroy(&Na->rwork));
  CHKERRQ(VecDestroy(&Na->lwork2));
  CHKERRQ(VecDestroy(&Na->rwork2));
  CHKERRQ(VecScatterDestroy(&Na->lrestrict));
  CHKERRQ(VecScatterDestroy(&Na->rprolong));
  CHKERRQ(MatDestroy(&Na->A));
  CHKERRQ(PetscFree(N->data));
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

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&N));
  CHKERRQ(ISGetLocalSize(isrow,&m));
  CHKERRQ(ISGetLocalSize(iscol,&n));
  CHKERRQ(MatSetSizes(N,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)N,MATSUBMATRIX));

  CHKERRQ(PetscNewLog(N,&Na));
  N->data   = (void*)Na;

  CHKERRQ(PetscObjectReference((PetscObject)isrow));
  CHKERRQ(PetscObjectReference((PetscObject)iscol));
  Na->isrow = isrow;
  Na->iscol = iscol;

  CHKERRQ(PetscFree(N->defaultvectype));
  CHKERRQ(PetscStrallocpy(A->defaultvectype,&N->defaultvectype));
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  CHKERRQ(MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A));

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

  CHKERRQ(MatSetBlockSizesFromMats(N,A,A));
  CHKERRQ(PetscLayoutSetUp(N->rmap));
  CHKERRQ(PetscLayoutSetUp(N->cmap));

  CHKERRQ(MatCreateVecs(A,&Na->rwork,&Na->lwork));
  CHKERRQ(MatCreateVecs(N,&right,&left));
  CHKERRQ(VecScatterCreate(Na->lwork,isrow,left,NULL,&Na->lrestrict));
  CHKERRQ(VecScatterCreate(right,NULL,Na->rwork,iscol,&Na->rprolong));
  CHKERRQ(VecDestroy(&left));
  CHKERRQ(VecDestroy(&right));
  CHKERRQ(MatSetUp(N));

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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)N,MATSUBMATRIX,&flg));
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix has wrong type");

  Na   = (Mat_SubVirtual*)N->data;
  CHKERRQ(ISEqual(isrow,Na->isrow,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different row indices");
  CHKERRQ(ISEqual(iscol,Na->iscol,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different column indices");

  CHKERRQ(PetscFree(N->defaultvectype));
  CHKERRQ(PetscStrallocpy(A->defaultvectype,&N->defaultvectype));
  CHKERRQ(MatDestroy(&Na->A));
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  CHKERRQ(MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A));
  PetscFunctionReturn(0);
}
