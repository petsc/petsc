/*$Id: convert.c,v 1.71 2001/01/15 21:46:25 bsmith Exp bsmith $*/

#include "src/mat/matimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatConvert_Basic"
/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.

  Does not do preallocation so in general will be slow
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *M)
{
  Scalar     *vwork;
  int        ierr,i,nz,m,n,*cwork,rstart,rend,lm;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lm,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(mat->comm,lm,PETSC_DECIDE,m,n,M);CHKERRQ(ierr);
  ierr = MatSetType(*M,newtype);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
