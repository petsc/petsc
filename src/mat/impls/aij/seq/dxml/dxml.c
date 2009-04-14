#define PETSCMAT_DLL

/*$Id: dxml.c,v 1.24 2001/08/07 03:02:47 balay Exp $*/

/* 
        Provides an interface to the DEC Alpha DXML library
     At the moment the DXNL library only offers sparse matrix vector product.
     Note: matrix i,j index must be 1-based (Fortran style)!
*/
#include "../src/mat/impls/aij/seq/aij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJ_DXML"
static int MatMult_SeqAIJ_DXML(Mat A,Vec x,Vec y)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *xx,*yy;
  int          ierr,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  dmatvec_genr_(&zero,a->a,a->i,a->j,&a->nz,0,xx,yy,&A->rmap->n);
  ierr = PetscLogFlops(2.0*a->nz - A->rmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseDXML_SeqAIJ"
int MatUseDXML_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"Presently not being supported");
  A->ops->mult = MatMult_SeqAIJ_DXML;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_DXML"
int PETSCMAT_DLLEXPORT MatCreate_SeqAIJ_DXML(Mat A) {
  int ierr;
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"Presently not being supported");
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatUseDXML_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
