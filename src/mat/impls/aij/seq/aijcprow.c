/*
  This file provides high performance routines for the compressed row SeqAIJ 
  format by taking advantage of rows without non-zero entries (zero rows).
*/
#include "src/mat/impls/aij/seq/aij.h"                

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJ_CompressedRow"
static PetscErrorCode MatMult_SeqAIJ_CompressedRow(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *x,*y,*aa,sum;
  PetscInt       m,*rindex,nz,i,j,*aj,*ai;

  PetscFunctionBegin;  
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  m      = a->compressedrow.nrows;
  ai     = a->compressedrow.i;
  rindex = a->compressedrow.rindex;
  for (i=0; i<m; i++){
    nz  = ai[i+1] - ai[i]; 
    aj  = a->j + ai[i];
    aa  = a->a + ai[i];
    sum = 0.0;
    for (j=0; j<nz; j++) sum += (*aa++)*x[*aj++]; 
    y[*rindex++] = sum;
  }

  PetscLogFlops(2*a->nz - m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Almost same code as the MatMult_SeqAij_CompressedRow() */
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJ_CompressedRow"
static PetscErrorCode MatMultAdd_SeqAIJ_CompressedRow(Mat A,Vec xx,Vec zz,Vec yy)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *x,*y,*z,*aa,sum;
  PetscInt       m,*rindex,nz,i,j,*aj,*ai;

  PetscFunctionBegin;  
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  m      = a->compressedrow.nrows;
  ai     = a->compressedrow.i;
  rindex = a->compressedrow.rindex;
  for (i=0; i<m; i++){
    nz  = ai[i+1] - ai[i]; 
    aj  = a->j + ai[i];
    aa  = a->a + ai[i];
    sum = y[*rindex];
    for (j=0; j<nz; j++) sum += (*aa++)*x[*aj++]; 
    z[*rindex++] = sum;
  }
  PetscLogFlops(2*a->nz - m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    samestructure indicates that the matrix has not changed its nonzero structure so we 
    do not need to recompute the inodes 
*/
#undef __FUNCT__  
#define __FUNCT__ "Mat_AIJ_CheckCompressedRow"
PetscErrorCode Mat_AIJ_CheckCompressedRow(Mat A,PetscTruth samestructure)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nrows,*cpi=PETSC_NULL,*rindex=PETSC_NULL,nz,m,i,row,*ai;
  PetscTruth     flg; 

  PetscFunctionBegin;  
  if (samestructure && a->compressedrow.checked) PetscFunctionReturn(0);
  a->compressedrow.checked = PETSC_TRUE; 

  if (!a->compressedrow.use) {PetscLogInfo(A,"Mat_AIJ_CheckCompressedRow: Not using CompressedRow routines due to MatSetOption(MAT_DO_NOT_USE_COMPRESSEDROW\n"); PetscFunctionReturn(0);}
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_no_compressedrow",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscLogInfo(A,"Mat_AIJ_CheckCompressedRow: Not using CompressedRow routines due to -mat_aij_no_compressedrow\n");PetscFunctionReturn(0);
  }
 
  /* compute number of zeros rows */
  m     = A->m;
  nrows = 0; 
  ai    = a->i;
  for (i=0; i<m; i++){                /* For each row */
    nz = ai[i+1] - ai[i];       /* Number of nonzeros */
    if (nz == 0) nrows++;
  }
  /* if enough zero rows are found, use compressedrow data structure */
  if (nrows < 0.4*m) {
    a->compressedrow.use = PETSC_FALSE; 
  } else {
    a->compressedrow.use = PETSC_TRUE;
    PetscLogInfo(A,"Mat_AIJ_CheckCompressedRow: Found %D zero rows out of %D local rows. Using CompressedRow routines.\n", nrows,m);
    /* set compressed row format */
    nrows = m - nrows; /* num of non-zero rows */
    ierr = PetscMalloc((2*nrows+1)*sizeof(PetscInt),&cpi);CHKERRQ(ierr);
    rindex = cpi + nrows + 1;
    row    = 0;
    cpi[0] = 0; 
    for (i=0; i<m; i++){                
      nz = ai[i+1] - ai[i];
      if (nz == 0) continue;
      cpi[row+1]  = ai[i+1];    /* compressed row pointer */
      rindex[row] = i;          /* compressed row local index */
      row++;
    }
    a->compressedrow.nrows  = nrows;
    a->compressedrow.i      = cpi;
    a->compressedrow.rindex = rindex;
    A->ops->mult            = MatMult_SeqAIJ_CompressedRow;
    A->ops->multadd         = MatMultAdd_SeqAIJ_CompressedRow; 
  }
  PetscFunctionReturn(0);
}
