/*$Id: convert.c,v 1.76 2001/08/07 03:03:20 balay Exp $*/

#include "src/mat/matimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "MatConvert_Basic"
/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.

  Does not do preallocation so in general will be slow
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *newmat) {
  Mat          M;
  PetscScalar  *vwork;
  int          ierr,i,nz,m,n,*cwork,rstart,rend,lm,ln;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lm,&ln);CHKERRQ(ierr);

  if (ln == n) ln = PETSC_DECIDE; /* try to preserve column ownership */

  ierr = MatCreate(mat->comm,lm,ln,m,n,&M);CHKERRQ(ierr);
  ierr = MatSetType(M,newtype);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (*newmat == A) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = M;
  PetscFunctionReturn(0);
}
