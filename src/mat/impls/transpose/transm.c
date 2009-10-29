#define PETSCMAT_DLL

#include "private/matimpl.h"          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_Transpose;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Transpose"
PetscErrorCode MatMult_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__   
#define __FUNCT__ "MatMultAdd_Transpose" 
PetscErrorCode MatMultAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3) 
{ 
  Mat_Transpose  *Na = (Mat_Transpose*)N->data; 
  PetscErrorCode ierr; 
 
  PetscFunctionBegin; 
  ierr = MatMultTransposeAdd(Na->A,v1,v2,v3);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Transpose"
PetscErrorCode MatMultTranspose_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__   
#define __FUNCT__ "MatMultTransposeAdd_Transpose" 
PetscErrorCode MatMultTransposeAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3) 
{ 
  Mat_Transpose  *Na = (Mat_Transpose*)N->data; 
  PetscErrorCode ierr; 
 
  PetscFunctionBegin; 
  ierr = MatMultAdd(Na->A,v1,v2,v3);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Transpose"
PetscErrorCode MatDestroy_Transpose(Mat N)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Na->A) { ierr = MatDestroy(Na->A);CHKERRQ(ierr); }
  ierr = PetscFree(Na);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "MatCreateTranspose"
/*@
      MatCreateTranspose - Creates a new matrix object that behaves like A'

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'

   Level: intermediate

   Notes: The transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by using the MatMultTranspose() on
          the original matrix

.seealso: MatCreateNormal(), MatMult(), MatMultTranspose(), MatCreate()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateTranspose(Mat A,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_Transpose  *Na;  

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A)->comm,N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATTRANSPOSEMAT);CHKERRQ(ierr);
  
  ierr      = PetscNewLog(*N,Mat_Transpose,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr      = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  Na->A     = A;

  (*N)->ops->destroy          = MatDestroy_Transpose;
  (*N)->ops->mult             = MatMult_Transpose;
  (*N)->ops->multadd          = MatMultAdd_Transpose;
  (*N)->ops->multtranspose    = MatMultTranspose_Transpose; 
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_Transpose; 
  (*N)->assembled             = PETSC_TRUE;

  ierr = PetscLayoutSetBlockSize((*N)->rmap,A->cmap->bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize((*N)->cmap,A->rmap->bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->cmap);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

