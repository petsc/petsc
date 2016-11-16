
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat         A,U,V;
  Vec         work1,work2; /* Sequential vectors that hold partial products */
  PetscMPIInt nwork;       /* length of work vectors */
} Mat_LRC;


PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqDense(Mat,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "MatMult_LRC"
PetscErrorCode MatMult_LRC(Mat N,Vec x,Vec y)
{
  Mat_LRC        *Na = (Mat_LRC*)N->data;
  PetscErrorCode ierr;
  PetscScalar    *w1,*w2;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);

  /* multiply the local part of V with the local part of x */
  /* note in this call x is treated as a sequential vector  */
  ierr = MatMultTranspose_SeqDense(Na->V,x,Na->work1);CHKERRQ(ierr);

  /* Form the sum of all the local multiplies : this is work2 = V'*x =
     sum_{all processors} work1 */

  ierr = VecGetArray(Na->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(Na->work2,&w2);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(w1,w2,Na->nwork,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N));CHKERRQ(ierr);
  ierr = VecRestoreArray(Na->work1,&w1);CHKERRQ(ierr);
  ierr = VecRestoreArray(Na->work2,&w2);CHKERRQ(ierr);

  /* multiply-sub y = y  + U*work2 */
  /* note in this call y is treated as a sequential vector  */
  ierr = MatMultAdd_SeqDense(Na->U,Na->work2,y,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_LRC"
PetscErrorCode MatDestroy_LRC(Mat N)
{
  Mat_LRC        *Na = (Mat_LRC*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->U);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->V);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work1);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work2);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateLRC"
/*@
   MatCreateLRC - Creates a new matrix object that behaves like A + U*V'

   Collective on Mat

   Input Parameters:
+  A  - the (sparse) matrix
-  U, V - two dense rectangular (tall and skinny) matrices

   Output Parameter:
.  N - the matrix that represents A + U*V'

   Notes:
   The matrix A + U*V' is not formed! Rather the new matrix
   object performs the matrix-vector product by first multiplying by
   A and then adding the other term.

   Level: intermediate
@*/
PetscErrorCode MatCreateLRC(Mat A,Mat U,Mat V,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n,k,m1,n1,k1;
  Mat_LRC        *Na;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(U,MAT_CLASSID,2);
  PetscValidHeaderSpecific(V,MAT_CLASSID,3);

  ierr = MatGetSize(U,NULL,&k);CHKERRQ(ierr);
  ierr = MatGetSize(V,NULL,&k1);CHKERRQ(ierr);
  if (k!=k1) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"U and V have different number of columns");
  ierr = MatGetLocalSize(U,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(V,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
  if (m!=m1) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"Local dimensions of U and A do not match");
  if (n!=n1) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Local dimensions of V and A do not match");

  ierr = MatCreate(PetscObjectComm((PetscObject)A),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATLRC);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*)Na;
  Na->A      = A;

  ierr = MatDenseGetLocalMatrix(U,&Na->U);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(V,&Na->V);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Na->U);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Na->V);CHKERRQ(ierr);

  ierr      = VecCreateSeq(PETSC_COMM_SELF,U->cmap->N,&Na->work1);CHKERRQ(ierr);
  ierr      = VecDuplicate(Na->work1,&Na->work2);CHKERRQ(ierr);
  Na->nwork = U->cmap->N;

  (*N)->ops->destroy = MatDestroy_LRC;
  (*N)->ops->mult    = MatMult_LRC;
  (*N)->assembled    = PETSC_TRUE;
  (*N)->cmap->N      = V->rmap->N;
  (*N)->rmap->N      = U->rmap->N;
  (*N)->cmap->n      = V->rmap->n;
  (*N)->rmap->n      = U->rmap->n;
  PetscFunctionReturn(0);
}

