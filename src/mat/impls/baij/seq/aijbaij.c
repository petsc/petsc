
#include "src/mat/impls/baij/seq/baij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqBAI_SeqAIJ"
PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat A,const MatType newtype,Mat *newmat) 
{
  Mat            B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       bs = A->bs,*ai = a->i,*aj = a->j,n = A->M/bs,i,j,k;
  PetscInt       *rowlengths,*rows,*cols,maxlen = 0,ncols;
  PetscScalar    *aa = a->a;

  PetscFunctionBegin;
  ierr = PetscMalloc(n*bs*sizeof(int),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    maxlen = PetscMax(maxlen,(ai[i+1] - ai[i]));
    for (j=0; j<bs; j++) {
      rowlengths[i*bs+j] = bs*(ai[i+1] - ai[i]);
    }
  }
  ierr = MatCreate(A->comm,A->m,A->n,A->m,A->n,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,rowlengths);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMN_ORIENTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROWS_SORTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = PetscMalloc(bs*sizeof(int),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(bs*maxlen*sizeof(int),&cols);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      rows[j] = i*bs+j;
    }
    ncols = ai[i+1] - ai[i];
    for (k=0; k<ncols; k++) {
      for (j=0; j<bs; j++) {
        cols[k*bs+j] = bs*(*aj) + j;
      }
      aj++;
    }
    ierr  = MatSetValues(B,bs,rows,bs*ncols,cols,aa,INSERT_VALUES);CHKERRQ(ierr);
    aa   += ncols*bs*bs;
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  B->bs = A->bs;

  /* Fake support for "inplace" convert. */
  if (B == A) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include "src/mat/impls/aij/seq/aij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqAIJ_SeqBAIJ"
PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ(Mat A,const MatType newtype,Mat *newmat) 
{
  Mat            B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  Mat_SeqBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,m=A->M,n=A->N,i,*rowlengths;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");

  ierr = PetscMalloc(m*sizeof(int),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    rowlengths[i] = ai[i+1] - ai[i];
  }
  ierr = MatCreate(A->comm,m,n,m,n,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(B,1,0,rowlengths);CHKERRQ(ierr);
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_ROW_ORIENTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROWS_SORTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
  b  = (Mat_SeqBAIJ*)(B->data);

  ierr = PetscMemcpy(b->i,a->i,(m+1)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(b->ilen,a->ilen,m*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(b->j,a->j,a->nz*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(b->a,a->a,a->nz*sizeof(MatScalar));CHKERRQ(ierr);
 
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (B == A) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
