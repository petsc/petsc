
#include <petsc-private/matimpl.h>  /*I   "petscmat.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "MatCheckCompressedRow"
/*@C
   MatCheckCompressedRow - Determines whether the compressed row matrix format should be used.
      If the format is to be used, this routine creates Mat_CompressedRow struct.
      Compressed row format provides high performance routines by taking advantage of zero rows.
      Supported types are MATAIJ, MATBAIJ and MATSBAIJ.

   Collective

   Input Parameters:
+  A             - the matrix
.  compressedrow - pointer to the struct Mat_CompressedRow
.  ai            - row pointer used by seqaij and seqbaij
.  mbs           - number of (block) rows represented by ai
-  ratio         - ratio of (num of zero rows)/m, used to determine if the compressed row format should be used

   Notes: By default PETSc will not check for compressed rows on sequential matrices. Call MatSetOption(Mat,MAT_CHECK_COMPRESSED_ROW,PETSC_TRUE); before
          MatAssemblyBegin() to have it check.

   Developer Note: The reason this takes the compressedrow, ai and mbs arguments is because it is called by both the SeqAIJ and SEQBAIJ matrices and
                   the values are not therefore obtained by directly taking the values from the matrix object.
   Level: developer
@*/
PetscErrorCode MatCheckCompressedRow(Mat A,Mat_CompressedRow *compressedrow,PetscInt *ai,PetscInt mbs,PetscReal ratio)
{
  PetscErrorCode ierr;
  PetscInt       nrows,*cpi=PETSC_NULL,*ridx=PETSC_NULL,nz,i,row;

  PetscFunctionBegin;
  if (!compressedrow->check) PetscFunctionReturn(0);

  /* in case this is being reused, delete old space */
  ierr = PetscFree2(compressedrow->i,compressedrow->rindex);CHKERRQ(ierr);
  compressedrow->i      = PETSC_NULL;
  compressedrow->rindex = PETSC_NULL;


  /* compute number of zero rows */
  nrows = 0;
  for (i=0; i<mbs; i++){        /* for each row */
    nz = ai[i+1] - ai[i];       /* number of nonzeros */
    if (nz == 0) nrows++;
  }

  /* if a large number of zero rows is found, use compressedrow data structure */
  if (nrows < ratio*mbs) {
    compressedrow->use = PETSC_FALSE;
    ierr = PetscInfo3(A,"Found the ratio (num_zerorows %d)/(num_localrows %d) < %G. Do not use CompressedRow routines.\n",nrows,mbs,ratio);CHKERRQ(ierr);
  } else {
    compressedrow->use = PETSC_TRUE;
    ierr = PetscInfo3(A,"Found the ratio (num_zerorows %d)/(num_localrows %d) > %G. Use CompressedRow routines.\n",nrows,mbs,ratio);CHKERRQ(ierr);

    /* set compressed row format */
    nrows = mbs - nrows; /* num of non-zero rows */
    ierr = PetscMalloc2(nrows+1,PetscInt,&cpi,nrows,PetscInt,&ridx);CHKERRQ(ierr);
    row    = 0;
    cpi[0] = 0;
    for (i=0; i<mbs; i++){
      nz = ai[i+1] - ai[i];
      if (nz == 0) continue;
      cpi[row+1]  = ai[i+1];    /* compressed row pointer */
      ridx[row++] = i;          /* compressed row local index */
    }
    compressedrow->nrows  = nrows;
    compressedrow->i      = cpi;
    compressedrow->rindex = ridx;
  }

  PetscFunctionReturn(0);
}
