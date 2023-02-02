
#include <petsc/private/matimpl.h>

/*
  MatConvert_Basic - Converts from any input format to another format.
  Does not do preallocation so in general will be slow
 */
PETSC_INTERN PetscErrorCode MatConvert_Basic(Mat mat, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat                M;
  const PetscScalar *vwork;
  PetscInt           i, rstart, rend, nz;
  const PetscInt    *cwork;
  PetscBool          isSBAIJ;

  PetscFunctionBegin;
  if (!mat->ops->getrow) { /* missing get row, use matvecs */
    PetscCall(MatConvert_Shell(mat, newtype, reuse, newmat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATSEQSBAIJ, &isSBAIJ));
  if (!isSBAIJ) PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPISBAIJ, &isSBAIJ));
  PetscCheck(!isSBAIJ, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Cannot convert from SBAIJ matrix since cannot obtain entire rows of matrix");

  if (reuse == MAT_REUSE_MATRIX) {
    M = *newmat;
  } else {
    PetscInt m, n, lm, ln;
    PetscCall(MatGetSize(mat, &m, &n));
    PetscCall(MatGetLocalSize(mat, &lm, &ln));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)mat), &M));
    PetscCall(MatSetSizes(M, lm, ln, m, n));
    PetscCall(MatSetBlockSizesFromMats(M, mat, mat));
    PetscCall(MatSetType(M, newtype));
    PetscCall(MatSetUp(M));

    PetscCall(MatSetOption(M, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQSBAIJ, &isSBAIJ));
    if (!isSBAIJ) PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPISBAIJ, &isSBAIJ));
    if (isSBAIJ) PetscCall(MatSetOption(M, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  }

  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(mat, i, &nz, &cwork, &vwork));
    PetscCall(MatSetValues(M, 1, &i, nz, cwork, vwork, INSERT_VALUES));
    PetscCall(MatRestoreRow(mat, i, &nz, &cwork, &vwork));
  }
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(mat, &M));
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
