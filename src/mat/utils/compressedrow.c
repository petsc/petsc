#include <petsc/private/matimpl.h> /*I   "petscmat.h"  I*/

/*@C
   MatCheckCompressedRow - Determines whether the compressed row matrix format should be used.
      Compressed row format provides high performance routines by taking advantage of zero rows.

   Collective

   Input Parameters:
+  A             - the matrix
.  nrows         - number of rows with nonzero entries
.  compressedrow - pointer to the struct Mat_CompressedRow
.  ai            - row pointer used by `MATSEQAIJ` and `MATSEQBAIJ`
.  mbs           - number of (block) rows represented by `ai`
-  ratio         - ratio of (num of zero rows)/m, used to determine if the compressed row format should be used

   Level: developer

   Note:
   Supported types are `MATAIJ`, `MATBAIJ` and `MATSBAIJ`.

   Developer Note:
   The reason this takes the `compressedrow`, `ai` and `mbs` arguments is because it is called by both the `MATSEQAIJ` and `MATSEQBAIJ` matrices and
   the values are not therefore obtained by directly taking the values from the matrix object.
   This is not a general public routine and hence is not listed in petscmat.h (it exposes a private data structure) but it is used
   by some preconditioners and hence is labeled as `PETSC_EXTERN`

.seealso: `Mat`, `MATAIJ`, `MATBAIJ`, `MATSBAIJ`.
@*/
PETSC_EXTERN PetscErrorCode MatCheckCompressedRow(Mat A, PetscInt nrows, Mat_CompressedRow *compressedrow, PetscInt *ai, PetscInt mbs, PetscReal ratio)
{
  PetscInt *cpi = NULL, *ridx = NULL, nz, i, row;

  PetscFunctionBegin;
  /* in case this is being reused, delete old space */
  PetscCall(PetscFree2(compressedrow->i, compressedrow->rindex));

  /* compute number of zero rows */
  nrows = mbs - nrows;

  /* if a large number of zero rows is found, use compressedrow data structure */
  if (nrows < ratio * mbs) {
    compressedrow->use = PETSC_FALSE;

    PetscCall(PetscInfo(A, "Found the ratio (num_zerorows %" PetscInt_FMT ")/(num_localrows %" PetscInt_FMT ") < %g. Do not use CompressedRow routines.\n", nrows, mbs, (double)ratio));
  } else {
    compressedrow->use = PETSC_TRUE;

    PetscCall(PetscInfo(A, "Found the ratio (num_zerorows %" PetscInt_FMT ")/(num_localrows %" PetscInt_FMT ") > %g. Use CompressedRow routines.\n", nrows, mbs, (double)ratio));

    /* set compressed row format */
    nrows = mbs - nrows; /* num of non-zero rows */
    PetscCall(PetscMalloc2(nrows + 1, &cpi, nrows, &ridx));
    row    = 0;
    cpi[0] = 0;
    for (i = 0; i < mbs; i++) {
      nz = ai[i + 1] - ai[i];
      if (nz == 0) continue;
      cpi[row + 1] = ai[i + 1]; /* compressed row pointer */
      ridx[row++]  = i;         /* compressed row local index */
    }
    compressedrow->nrows  = nrows;
    compressedrow->i      = cpi;
    compressedrow->rindex = ridx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
