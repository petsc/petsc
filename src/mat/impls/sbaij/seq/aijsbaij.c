/*$Id: aijsbaij.c,v 1.9 2001/08/07 03:02:55 balay Exp $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqSBAI_SeqAIJ"
int MatConvert_SeqSBAIJ_SeqAIJ(Mat A,const MatType newtype,Mat *newmat) 
{
  Mat          B;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data; 
  Mat_SeqAIJ   *b;
  int          ierr,bs = a->bs,*ai=a->i,*aj=a->j,m=A->M/bs,*bi,*bj,
               i,*rowlengths,nz,*rowstart;
  PetscScalar  *av,*bv;

  PetscFunctionBegin;
  /* compute rowlengths of newmat */
  ierr = PetscMalloc(m*bs*sizeof(int),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) rowlengths[i] = 0;
  aj = a->j;
  for (i=0; i<m; i++) {
    nz = ai[i+1] - ai[i];
    rowlengths[i] += nz; /* upper triangular part */
    aj++; nz--;          /* skip diagonal */
    while (nz--) { rowlengths[*aj++]++; }  /* lower triangular part */
  }
  
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,m,0,rowlengths,&B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROWS_SORTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
  b  = (Mat_SeqAIJ*)(B->data);
  bi = b->i;
  bj = b->j; 
  bv = b->a; 

  /* set b->i */
  rowstart = rowlengths; /* rowstart renames rowlengths for code understanding */
  bi[0] = 0;
  for (i=0; i<m; i++){
    b->ilen[i]  = rowlengths[i];
    bi[i+1]     = bi[i] + rowlengths[i]; 
    rowstart[i] = bi[i];  
  }
  if (bi[m] != 2*a->nz - m) SETERRQ2(1,"bi[m]: %d != 2*a->nz-m: %d\n",bi[m],2*a->nz - m);

  /* set b->j and b->a */
  aj = a->j; av = a->a;
  for (i=0; i<m; i++) {
    /* diagonal */
    *(bj + rowstart[i]) = *aj; aj++;
    *(bv + rowstart[i]) = *av; av++; 
    rowstart[i]++; 
    nz = ai[i+1] - ai[i] - 1;
    while (nz--){
      /* lower triangular part */
      *(bj + rowstart[*aj]) = i; 
      *(bv + rowstart[*aj]) = *av; 
      rowstart[*aj]++; 
      /* upper triangular part */
      *(bj + rowstart[i]) = *aj; aj++;
      *(bv + rowstart[i]) = *av; av++; 
      rowstart[i]++; 
    }
  }
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (*newmat == A) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;
  
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqAIJ_SeqSBAIJ"
int MatConvert_SeqAIJ_SeqSBAIJ(Mat A,const MatType newtype,Mat *newmat) {
  Mat          B;
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data; 
  Mat_SeqSBAIJ *b;
  int          ierr,*ai=a->i,*aj,m=A->M,n=A->N,i,j,
               *bi,*bj,*rowlengths;
  PetscScalar  *av,*bv;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (!a->diag){
    ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr); 
  }

  ierr = PetscMalloc(m*sizeof(int),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    rowlengths[i] = ai[i+1] - a->diag[i];
  }
  ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,1,m,n,0,rowlengths,&B);CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_ROW_ORIENTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROWS_SORTED);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
  b  = (Mat_SeqSBAIJ*)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  bi[0] = 0;
  for (i=0; i<m; i++) {
    aj = a->j + a->diag[i];
    av = a->a + a->diag[i];    
    for (j=0; j<rowlengths[i]; j++){
      *bj = *aj; bj++; aj++;
      *bv = *av; bv++; av++;
    }
    bi[i+1]    = bi[i] + rowlengths[i];
    b->ilen[i] = rowlengths[i];
  }
 
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (*newmat == A) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;

  PetscFunctionReturn(0);
}
EXTERN_C_END

