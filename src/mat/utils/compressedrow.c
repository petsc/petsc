
#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

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
-  ratio         - ratio of (num of zero rows)/m, used to determine if the compressed row format should be used

   Level: developer
@*/
PetscErrorCode Mat_CheckCompressedRow(Mat A,Mat_CompressedRow *compressedrow,PetscInt *ai,PetscReal ratio) 
{
  PetscErrorCode ierr;
  PetscInt       nrows,*cpi=PETSC_NULL,*ridx=PETSC_NULL,nz,i,row,m=A->m; 
  MatType        mtype;
  PetscMPIInt    size;  
  PetscTruth     aij;

  PetscFunctionBegin;  
  if (!compressedrow->use) PetscFunctionReturn(0);
  if (compressedrow->checked){
    if (!A->same_nonzero){
      ierr = PetscFree(compressedrow->i);CHKERRQ(ierr); 
      compressedrow->rindex = PETSC_NULL;
      PetscLogInfo(A,"Mat_CheckCompressedRow: Mat structure might be changed. Free memory and recheck.\n");
    } else if (compressedrow->i == PETSC_NULL) {
      /* Don't know why this occures. For safe, recheck. */
      PetscLogInfo(A,"Mat_CheckCompressedRow: compressedrow.checked, but compressedrow.i==null. Recheck.\n");
    } else { /* use compressedrow, checked, A->same_nonzero = PETSC_TRUE. Skip check */
      PetscLogInfo(A,"Mat_CheckCompressedRow: Skip check. m: %d, n: %d,M: %d, N: %d,nrows: %d, ii: %p, type: %s\n",A->m,A->n,A->M,A->N,compressedrow->nrows,compressedrow->i,A->type_name);
      PetscFunctionReturn(0); 
    }
  }
  compressedrow->checked = PETSC_TRUE; 

  /* set m=A->m/A->bs for BAIJ and SBAIJ matrices */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype,MATAIJ,&aij);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr); 
  if (!aij){
    if (size == 1){
      ierr = PetscStrcmp(mtype,MATSEQAIJ,&aij);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcmp(mtype,MATMPIAIJ,&aij);CHKERRQ(ierr);
    }
  }
  if (!aij){
    m = m/A->bs;
  } 

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
