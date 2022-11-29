
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_1WD - Find the 1-way dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_1WD(Mat mat, MatOrderingType type, IS *row, IS *col)
{
  PetscInt        i, *mask, *xls, nblks, *xblk, *ls, nrow, *perm;
  const PetscInt *ia, *ja;
  PetscBool       done;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, &nrow, &ia, &ja, &done));
  PetscCheck(done, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Cannot get rows for matrix");

  PetscCall(PetscMalloc5(nrow, &mask, nrow + 1, &xls, nrow, &ls, nrow + 1, &xblk, nrow, &perm));
  SPARSEPACKgen1wd(&nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls);
  PetscCall(MatRestoreRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, NULL, &ia, &ja, &done));

  for (i = 0; i < nrow; i++) perm[i]--;

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrow, perm, PETSC_COPY_VALUES, row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrow, perm, PETSC_COPY_VALUES, col));
  PetscCall(PetscFree5(mask, xls, ls, xblk, perm));
  PetscFunctionReturn(0);
}
