/*$Id: convert.c,v 1.65 2000/01/11 21:01:18 bsmith Exp bsmith $*/

#include "src/mat/matimpl.h"

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatConvert_Basic"
/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *M)
{
  Scalar *vwork;
  int    ierr,i,nz,m,n,*cwork,rstart,rend,lm,ln;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lm,&ln);CHKERRQ(ierr);
  if (newtype == MATSAME) newtype = (MatType)mat->type;
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(mat->comm,m,n,0,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
    case MATMPIROWBS:
      if (m != n) SETERRQ(PETSC_ERR_SUP,0,"MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(mat->comm,PETSC_DECIDE,m,0,PETSC_NULL,PETSC_NULL,M);CHKERRQ(ierr);
      break;
#endif
    case MATMPIAIJ:
      if (ln == n) ln = PETSC_DECIDE;
      ierr = MatCreateMPIAIJ(mat->comm,lm,ln,m,n,0,PETSC_NULL,0,PETSC_NULL,M);CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(mat->comm,m,n,PETSC_NULL,M);CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(mat->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,PETSC_NULL,M);CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);    
      ierr = MatCreateSeqBDiag(mat->comm,m,n,0,bs,PETSC_NULL,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);   
      ierr = MatCreateMPIBDiag(mat->comm,PETSC_DECIDE,m,n,0,bs,PETSC_NULL,
             PETSC_NULL,M);CHKERRQ(ierr); 
      break;
      }
    case MATSEQBAIJ:
      ierr = MatCreateSeqBAIJ(mat->comm,1,m,n,0,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
    case MATMPIBAIJ:
      ierr = MatCreateMPIBAIJ(mat->comm,1,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,PETSC_NULL,0,PETSC_NULL,M);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_ERR_SUP,0,"Matrix type is not currently supported");
  }
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
