#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: convert.c,v 1.61 1997/12/01 01:55:30 bsmith Exp balay $";
#endif

#include "src/mat/matimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatConvert_Basic"
/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *M)
{
  Scalar *vwork;
  int    ierr, i, nz, m, n, *cwork, rstart, rend,flg;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  if (newtype == MATSAME) newtype = (MatType)mat->type;
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(mat->comm,m,n,0,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(PETSC_ERR_SUP,0,"MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(mat->comm,PETSC_DECIDE,m,0,PETSC_NULL,
             PETSC_NULL,M);CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(mat->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,PETSC_NULL,0,PETSC_NULL,M);CHKERRQ(ierr);
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
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);    
      ierr = MatCreateSeqBDiag(mat->comm,m,n,0,bs,PETSC_NULL,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);   
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
  ierr = MatGetOwnershipRange(*M,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
