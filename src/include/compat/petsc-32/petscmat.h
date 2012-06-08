#include "private/matimpl.h"

#undef __FUNCT__
#define __FUNCT__ "MatBlockSize_Check"
static PetscErrorCode
MatBlockSize_Check(Mat mat,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (bs < 1) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Invalid block size specified, must be positive but it is %D",bs);
  }
  if (mat->rmap->n != -1 && mat->rmap->n % bs) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Local row length %D not divisible by block size %D",
             mat->rmap->n,bs);
  }
  if (mat->rmap->N != -1 && mat->rmap->N % bs) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Global row length %D not divisible by block size %D",
             mat->rmap->N,bs);
  }
  if (mat->cmap->n != -1 && mat->cmap->n % bs) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Local column length %D not divisible by block size %D",
             mat->cmap->n,bs);
  }
  if (mat->cmap->N != -1 && mat->cmap->N % bs) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Global column length %D not divisible by block size %D",
             mat->cmap->N,bs);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBlockSize_SetUp"
static PetscErrorCode
MatBlockSize_SetUp(Mat mat,PetscInt bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscLayoutSetBlockSize(mat->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(mat->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetBlockSize_Patch"
static PetscErrorCode
MatSetBlockSize_Patch(Mat mat,PetscInt bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (bs < 1)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                        "Invalid block size specified, must be positive but it is %D",bs);
  /*if (mat->ops->setblocksize) {
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = (*mat->ops->setblocksize)(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
    } else */ 
  if (mat->rmap->bs == -1 || mat->cmap->bs == -1) {
    ierr = MatBlockSize_Check(mat,bs);CHKERRQ(ierr);
    ierr = MatBlockSize_SetUp(mat,bs);CHKERRQ(ierr);
  } else if (mat->rmap->bs != bs || mat->cmap->bs != bs) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,
             "Cannot set/change the block sizes %D,%D to %D for matrix type %s",
             mat->rmap->bs,mat->cmap->bs,bs,((PetscObject)mat)->type_name);
  }
  PetscFunctionReturn(0);
}
#undef  MatSetBlockSize
#define MatSetBlockSize MatSetBlockSize_Patch


#define MatSetNullSpace MatNullSpaceAttach
#define MatTransposeMatMult MatMatMultTranspose

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult"
static PetscErrorCode MatMatTransposeMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  PetscValidPointer(C,3);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_Compat"
static PetscErrorCode MatInvertBlockDiagonal_Compat(Mat mat,const PetscScalar **values)
{return MatInvertBlockDiagonal(mat,(PetscScalar**)values);}
#undef  MatInvertBlockDiagonal
#define MatInvertBlockDiagonal MatInvertBlockDiagonal_Compat
