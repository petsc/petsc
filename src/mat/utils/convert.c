
#include <petsc/private/matimpl.h>

/*
  MatConvert_Basic - Converts from any input format to another format.
  Does not do preallocation so in general will be slow
 */
PETSC_INTERN PetscErrorCode MatConvert_Basic(Mat mat,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               M;
  const PetscScalar *vwork;
  PetscInt          i,rstart,rend,nz;
  const PetscInt    *cwork;
  PetscBool         isSBAIJ;

  PetscFunctionBegin;
  if (!mat->ops->getrow) { /* missing get row, use matvecs */
    CHKERRQ(MatConvert_Shell(mat,newtype,reuse,newmat));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATSEQSBAIJ,&isSBAIJ));
  if (!isSBAIJ) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATMPISBAIJ,&isSBAIJ));
  }
  PetscCheckFalse(isSBAIJ,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot convert from SBAIJ matrix since cannot obtain entire rows of matrix");

  if (reuse == MAT_REUSE_MATRIX) {
    M = *newmat;
  } else {
    PetscInt m,n,lm,ln;
    CHKERRQ(MatGetSize(mat,&m,&n));
    CHKERRQ(MatGetLocalSize(mat,&lm,&ln));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)mat),&M));
    CHKERRQ(MatSetSizes(M,lm,ln,m,n));
    CHKERRQ(MatSetBlockSizesFromMats(M,mat,mat));
    CHKERRQ(MatSetType(M,newtype));
    CHKERRQ(MatSetUp(M));

    CHKERRQ(MatSetOption(M,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
    CHKERRQ(MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATSEQSBAIJ,&isSBAIJ));
    if (!isSBAIJ) {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATMPISBAIJ,&isSBAIJ));
    }
    if (isSBAIJ) {
      CHKERRQ(MatSetOption(M,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));
    }
  }

  CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatGetRow(mat,i,&nz,&cwork,&vwork));
    CHKERRQ(MatSetValues(M,1,&i,nz,cwork,vwork,INSERT_VALUES));
    CHKERRQ(MatRestoreRow(mat,i,&nz,&cwork,&vwork));
  }
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(mat,&M));
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(0);
}
