/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/

#include "src/mat/matimpl.h"          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
  Vec w;
} Mat_Normal;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Normal"
int MatMult_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal *Na = (Mat_Normal*)N->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,Na->w);CHKERRQ(ierr);
  ierr = MatMultTranspose(Na->A,Na->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Normal"
int MatDestroy_Normal(Mat N)
{
  Mat_Normal *Na = (Mat_Normal*)N->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscObjectDereference((PetscObject)Na->A);CHKERRQ(ierr);
  ierr = VecDestroy(Na->w);CHKERRQ(ierr);
  ierr = PetscFree(Na);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  

#undef __FUNCT__  
#define __FUNCT__ "MatCreateNormal"
/*@
      MatCreateNormal - Creates a new matrix object that behaves like A'*A.

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'*A

   Notes: The product A'*A is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by first multiplying by
          A and then A'
@*/
int MatCreateNormal(Mat A,Mat *N)
{
  int        ierr,m,n;
  Mat_Normal *Na;  

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(A->comm,n,n,PETSC_DECIDE,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATNORMAL);CHKERRQ(ierr);
  
  ierr      = PetscNew(Mat_Normal,&Na);CHKERRQ(ierr);
  Na->A     = A;
  ierr      = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  (*N)->data = (void*) Na;

  ierr    = VecCreateMPI(A->comm,m,PETSC_DECIDE,&Na->w);CHKERRQ(ierr);
  (*N)->ops->destroy = MatDestroy_Normal;
  (*N)->ops->mult    = MatMult_Normal;
  (*N)->assembled    = PETSC_TRUE;
  (*N)->N            = A->N;
  (*N)->M            = A->N;
  (*N)->n            = A->n;
  (*N)->m            = A->n;
  PetscFunctionReturn(0);
}

