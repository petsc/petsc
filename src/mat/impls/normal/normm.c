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
  
/*
      Slow, nonscalable version
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Normal"
int MatGetDiagonal_Normal(Mat N,Vec v)
{
  Mat_Normal  *Na = (Mat_Normal*)N->data;
  Mat         A = Na->A;
  int         ierr,i,j,rstart,rend,nnz,*cols;
  PetscScalar *diag,*work,*values;
  PetscMap    cmap;

  PetscFunctionBegin;
  ierr = PetscMalloc(2*A->N*sizeof(PetscScalar),&diag);CHKERRQ(ierr);
  work = diag + A->N;
  ierr = PetscMemzero(work,A->N*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nnz,&cols,&values);CHKERRQ(ierr);
    for (j=0; j<nnz; j++) {
      work[cols[j]] += values[j]*values[j];
    }
    ierr = MatRestoreRow(A,i,&nnz,&cols,&values);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(work,diag,A->N,MPIU_SCALAR,MPI_SUM,N->comm);CHKERRQ(ierr);
  ierr = MatGetPetscMaps(A,PETSC_NULL,&cmap);CHKERRQ(ierr);
  ierr = PetscMapGetLocalRange(cmap,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArray(v,&values);CHKERRQ(ierr);
  ierr = PetscMemcpy(values,diag+rstart,(rend-rstart)*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&values);CHKERRQ(ierr);
  ierr = PetscFree(diag);CHKERRQ(ierr);
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
  (*N)->ops->destroy     = MatDestroy_Normal;
  (*N)->ops->mult        = MatMult_Normal;
  (*N)->ops->getdiagonal = MatGetDiagonal_Normal;
  (*N)->assembled        = PETSC_TRUE;
  (*N)->N                = A->N;
  (*N)->M                = A->N;
  (*N)->n                = A->n;
  (*N)->m                = A->n;
  PetscFunctionReturn(0);
}

