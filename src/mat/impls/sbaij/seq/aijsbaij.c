/*$Id: aijsbaij.c,v 1.9 2001/08/07 03:02:55 balay Exp $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqAIJ_SeqSBAIJ"
int MatConvert_SeqAIJ_SeqSBAIJ(Mat A,MatType newtype,Mat *newmat) {
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

