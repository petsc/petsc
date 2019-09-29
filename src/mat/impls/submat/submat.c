
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(Na->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_SubMatrix(Mat N,PetscScalar a)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShift(Na->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SubMatrix(Mat N,Vec left,Vec right)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (right) {
    ierr = VecZeroEntries(Na->rwork);CHKERRQ(ierr);
    ierr = VecScatterBegin(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->rprolong,right,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecZeroEntries(Na->lwork);CHKERRQ(ierr);
    ierr = VecScatterBegin(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->lrestrict,left,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = MatDiagonalScale(Na->A,left ? Na->lwork : NULL,right ? Na->rwork : NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SubMatrix(Mat N,Vec d)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetDiagonal(Na->A,Na->rwork);CHKERRQ(ierr); 
  ierr = VecScatterBegin(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->rprolong,Na->rwork,d,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(Na->rwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->rprolong,x,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(Na->A,Na->rwork,Na->lwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->lrestrict,Na->lwork,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(Na->rwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->rprolong,v1,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (v1 == v2) {
    ierr = MatMultAdd(Na->A,Na->rwork,Na->rwork,Na->lwork);CHKERRQ(ierr);
  } else if (v2 == v3) {
    ierr = VecZeroEntries(Na->lwork);CHKERRQ(ierr);
    ierr = VecScatterBegin(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->lrestrict,v2,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultAdd(Na->A,Na->rwork,Na->lwork,Na->lwork);CHKERRQ(ierr);
  } else {
    if (!Na->lwork2) {
      ierr = VecDuplicate(Na->lwork,&Na->lwork2);CHKERRQ(ierr);
    } else {
      ierr = VecZeroEntries(Na->lwork2);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->lrestrict,v2,Na->lwork2,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultAdd(Na->A,Na->rwork,Na->lwork2,Na->lwork);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->lrestrict,Na->lwork,v3,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SubMatrix(Mat N,Vec x,Vec y)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(Na->lwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->lrestrict,x,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(Na->A,Na->lwork,Na->rwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->rprolong,Na->rwork,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SubMatrix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(Na->lwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->lrestrict,v1,Na->lwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (v1 == v2) {
    ierr = MatMultTransposeAdd(Na->A,Na->lwork,Na->lwork,Na->rwork);CHKERRQ(ierr);
  } else if (v2 == v3) {
    ierr = VecZeroEntries(Na->rwork);CHKERRQ(ierr);
    ierr = VecScatterBegin(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->rprolong,v2,Na->rwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork,Na->rwork);CHKERRQ(ierr);
  } else {
    if (!Na->rwork2) {
      ierr = VecDuplicate(Na->rwork,&Na->rwork2);CHKERRQ(ierr);
    } else {
      ierr = VecZeroEntries(Na->rwork2);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(Na->rprolong,v2,Na->rwork2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(Na->A,Na->lwork,Na->rwork2,Na->rwork);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(Na->rprolong,Na->rwork,v3,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SubMatrix(Mat N)
{
  Mat_SubVirtual *Na = (Mat_SubVirtual*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&Na->isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&Na->iscol);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->lwork);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->rwork);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->lwork2);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->rwork2);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&Na->lrestrict);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&Na->rprolong);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscValidPointer(newmat,4);
  *newmat = 0;

  ierr = MatCreate(PetscObjectComm((PetscObject)A),&N);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&m);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&n);CHKERRQ(ierr);
  ierr = MatSetSizes(N,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)N,MATSUBMATRIX);CHKERRQ(ierr);

  ierr      = PetscNewLog(N,&Na);CHKERRQ(ierr);
  N->data   = (void*)Na;

  ierr      = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  Na->isrow = isrow;
  Na->iscol = iscol;

  ierr = PetscFree(N->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(A->defaultvectype,&N->defaultvectype);CHKERRQ(ierr);
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  ierr = MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A);CHKERRQ(ierr);

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

  ierr = MatSetBlockSizesFromMats(N,A,A);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(N->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(N->cmap);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&Na->rwork,&Na->lwork);CHKERRQ(ierr);
  ierr = MatCreateVecs(N,&right,&left);CHKERRQ(ierr);
  ierr = VecScatterCreate(Na->lwork,isrow,left,NULL,&Na->lrestrict);CHKERRQ(ierr);
  ierr = VecScatterCreate(right,NULL,Na->rwork,iscol,&Na->rprolong);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = MatSetUp(N);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscBool      flg;
  Mat_SubVirtual *Na;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(N,MAT_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,3);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,4);
  ierr = PetscObjectTypeCompare((PetscObject)N,MATSUBMATRIX,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix has wrong type");

  Na   = (Mat_SubVirtual*)N->data;
  ierr = ISEqual(isrow,Na->isrow,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different row indices");
  ierr = ISEqual(iscol,Na->iscol,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot update submatrix with different column indices");

  ierr = PetscFree(N->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(A->defaultvectype,&N->defaultvectype);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  /* Do not use MatConvert directly since MatShell has a duplicate operation which does not increase
     the reference count of the context. This is a problem if A is already of type MATSHELL */
  ierr = MatConvertFrom_Shell(A,MATSHELL,MAT_INITIAL_MATRIX,&Na->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
