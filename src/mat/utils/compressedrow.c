
#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "Mat_CheckCompressedRow"
/*@C
   Mat_CheckCompressedRow - Determines whether the compressed row matrix format should be used. If
                            the format is to be used, this routine creates Mat_CompressedRow struct.

                            Compressed row format provides high performance routines by
                            taking advantage of zero rows.

   Collective

   Input Parameters:
+  A             - the matrix
.  compressedrow - pointer to the struct Mat_CompressedRow
.  ai            - row pointer used by seqaij and seqbaij
-  ratio         - ratio of (num of zero rows)/m, used to determine if the compressed row format should be used

   Level: developer
@*/
PetscErrorCode Mat_CheckCompressedRow(Mat A,Mat_CompressedRow *compressedrow,PetscInt *ai,PetscReal ratio) 
{
  PetscErrorCode ierr;
  PetscInt       nrows,*cpi=PETSC_NULL,*ridx=PETSC_NULL,nz,i,row,m=A->m/A->bs; 

  PetscFunctionBegin;  
  if (!compressedrow->use) PetscFunctionReturn(0);
  if (compressedrow->checked && !A->same_nonzero){
    ierr = PetscFree(compressedrow->i);CHKERRQ(ierr); 
    compressedrow->rindex = PETSC_NULL;
    PetscLogInfo(A,"Mat_CheckCompressedRow: Mat structure might be changed. Free memory and recheck.\n");
  }
  compressedrow->checked = PETSC_TRUE; 

  /* compute number of zero rows */
  nrows = 0; 
  for (i=0; i<m; i++){                /* for each row */
    nz = ai[i+1] - ai[i];       /* number of nonzeros */
    if (nz == 0) nrows++;
  }
  /* if enough zero rows are found, use compressedrow data structure */
  if (nrows < ratio*m) {
    compressedrow->use = PETSC_FALSE; 
    PetscLogInfo(A,"Mat_CheckCompressedRow: Found the ratio (num_zerorows %d)/(num_localrows %d) < %g. Do not use CompressedRow routines.\n",nrows,m,ratio);
  } else {
    compressedrow->use = PETSC_TRUE; 
    PetscLogInfo(A,"Mat_CheckCompressedRow: Found the ratio (num_zerorows %d)/(num_localrows %d) > %g. Use CompressedRow routines.\n",nrows,m,ratio);

    /* set compressed row format */
    nrows = m - nrows; /* num of non-zero rows */
    ierr = PetscMalloc((2*nrows+1)*sizeof(PetscInt),&cpi);CHKERRQ(ierr);
    ridx = cpi + nrows + 1;
    row    = 0;
    cpi[0] = 0; 
    for (i=0; i<m; i++){                
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
