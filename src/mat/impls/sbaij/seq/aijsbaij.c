/*$Id: aijsbaij.c,v 1.9 2001/08/07 03:02:55 balay Exp $*/

#include "src/mat/impls/aij/seq/aij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqAIJ_SeqSBAIJ"
int MatConvert_SeqAIJ_SeqSBAIJ(Mat A,MatType newtype,Mat *B)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data; 
  int          ierr,*ai=a->i,*aj,m=A->M,n=A->N,i;
  int          *rowlengths;
  PetscScalar  *av;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_ERR_SUP,"Matrix must be a square matrix");
  if (!a->diag){
    ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr); 
  }

  ierr = PetscMalloc(m*sizeof(int),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    rowlengths[i] = ai[i+1] - a->diag[i];
  }
  ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,1,m,n,0,rowlengths,B);CHKERRQ(ierr);

  ierr = MatSetOption(*B,MAT_ROW_ORIENTED);CHKERRQ(ierr);
  ierr = MatSetOption(*B,MAT_ROWS_SORTED);CHKERRQ(ierr);
  ierr = MatSetOption(*B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
  for (i=0; i<m; i++) {
    aj   = a->j + a->diag[i];
    av   = a->a + a->diag[i];
    ierr = MatSetValues(*B,1,&i,rowlengths[i],aj,av,INSERT_VALUES);CHKERRQ(ierr);
  }
 
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

