#define PETSCMAT_DLL

#include "include/private/matimpl.h"          /*I "petscmat.h" I*/

typedef struct {
  Mat A,B,C,D;
} Mat_SchurComplement;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SchurComplement"
PetscErrorCode MatMult_SchurComplement(Mat N,Vec x,Vec y)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SchurComplement"
PetscErrorCode MatDestroy_SchurComplement(Mat N)
{
  Mat_SchurComplement  *Na = (Mat_SchurComplement*)N->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (Na->A) {ierr = MatDestroy(Na->A);CHKERRQ(ierr);}
  if (Na->B) {ierr = MatDestroy(Na->B);CHKERRQ(ierr);}
  if (Na->C) {ierr = MatDestroy(Na->C);CHKERRQ(ierr);}
  if (Na->D) {ierr = MatDestroy(Na->D);CHKERRQ(ierr);}
  ierr = KSPDestroy(Na->ksp);CHKERRQ(ierr);
  ierr = PetscFree(Na);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSchurComplement"
/*@
      MatCreateSchurComplement - Creates a new matrix object that behaves like the Schur complement of a matrix

   Collective on Mat

   Input Parameter:
.   A,B,C,D  - the four parts of the original matrix (D is optional)

   Output Parameter:
.   N - the matrix that the Schur complement D - C inv(A) B

   Level: intermediate

   Notes: The Schur complement is NOT actually formed! Rather this 
          object performs the matrix-vector product by using the the formula for
          the Schur complement and a KSP solver to approximate the action of inv(A)

          All four matrices must have the same MPI communicator

.seealso: MatCreateNormal(), MatMult(), MatCreate(), MatSchurComplementGetKSP()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSchurComplement(Mat A,Mat B,Mat C,Mat D,Mat *N)
{
  PetscErrorCode       ierr;
  PetscInt             m,n;
  Mat_SchurComplement  *Na;  

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,0);
  PetscCheckSameComm(A,0,B,1);
  PetscCheckSameComm(A,0,C,2);
  if (D)   PetscCheckSameComm(A,0,D,3);

  ierr = MatGetLocalSize(D,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A)->comm,N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
  
  ierr      = PetscNewLog(*N,Mat_SchurComplement,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr      = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  Na->A     = A;
  Na->B     = B;
  Na->C     = C;
  Na->D     = D;

  (*N)->ops->destroy     = MatDestroy_SchurComplement;
  (*N)->ops->mult        = MatMult_SchurComplement;
  (*N)->assembled        = PETSC_TRUE;

  /* treats the new matrix as having block size of 1 which is most likely the case */
  (*N)->rmap.bs = (*N)->cmap.bs = 1;
  ierr = PetscMapSetUp(&(*N)->rmap);CHKERRQ(ierr);
  ierr = PetscMapSetUp(&(*N)->cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

