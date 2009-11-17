#define PETSCMAT_DLL

#include "private/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "Mat_CheckCompressedRow"
/*@C
   Mat_CheckCompressedRow - Determines whether the compressed row matrix format should be used. 
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

   Level: developer
@*/
PetscErrorCode Mat_CheckCompressedRow(Mat A,Mat_CompressedRow *compressedrow,PetscInt *ai,PetscInt mbs,PetscReal ratio) 
{
  PetscErrorCode ierr;
  PetscInt       nrows,*cpi=PETSC_NULL,*ridx=PETSC_NULL,nz,i,row; 

  PetscFunctionBegin;  
  if (!compressedrow->use) PetscFunctionReturn(0);
  if (compressedrow->checked){
    if (!A->same_nonzero){
      ierr = PetscFree2(compressedrow->i,compressedrow->rindex);CHKERRQ(ierr); 
      compressedrow->i      = PETSC_NULL;
      compressedrow->rindex = PETSC_NULL;
      ierr = PetscInfo(A,"Mat structure might be changed. Free memory and recheck.\n");CHKERRQ(ierr);
    } else if (!compressedrow->i) {
      /* Don't know why this occures. For safe, recheck. */
      ierr = PetscInfo(A,"compressedrow.checked, but compressedrow.i==null. Recheck.\n");CHKERRQ(ierr);
    } else { /* use compressedrow, checked, A->same_nonzero = PETSC_TRUE. Skip check */
      ierr = PetscInfo7(A,"Skip check. m: %d, n: %d,M: %d, N: %d,nrows: %d, ii: %p, type: %s\n",A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,compressedrow->nrows,compressedrow->i,((PetscObject)A)->type_name);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    }
  }
  compressedrow->checked = PETSC_TRUE; 

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
