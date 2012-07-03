
#include <petsc-private/matimpl.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

typedef struct {
  Mat         A,U,V;
  Vec         work1,work2;/* Sequential (big) vectors that hold partial products */
  PetscMPIInt nwork;      /* length of work vectors */
} Mat_LRC;



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
  ierr = MPI_Allreduce(w1,w2,Na->nwork,MPIU_SCALAR,MPIU_SUM,((PetscObject)N)->comm);CHKERRQ(ierr);
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

   Input Parameter:
+   A  - the (sparse) matrix
-   U. V - two dense rectangular (tall and skinny) matrices

   Output Parameter:
.   N - the matrix that represents A + U*V'

   Level: intermediate

   Notes: The matrix A + U*V' is not formed! Rather the new matrix
          object performs the matrix-vector product by first multiplying by
          A and then adding the other term
@*/
PetscErrorCode  MatCreateLRC(Mat A,Mat U, Mat V,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_LRC        *Na;  

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A)->comm,N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATLRC);CHKERRQ(ierr);
  
  ierr       = PetscNewLog(*N,Mat_LRC,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  Na->A      = A;

  ierr      = MatDenseGetLocalMatrix(U,&Na->U);CHKERRQ(ierr);
  ierr      = MatDenseGetLocalMatrix(V,&Na->V);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)Na->U);CHKERRQ(ierr);
  ierr      = PetscObjectReference((PetscObject)Na->V);CHKERRQ(ierr);

  ierr                   = VecCreateSeq(PETSC_COMM_SELF,U->cmap->N,&Na->work1);CHKERRQ(ierr);
  ierr                   = VecDuplicate(Na->work1,&Na->work2);CHKERRQ(ierr);
  Na->nwork              = U->cmap->N;

  (*N)->ops->destroy     = MatDestroy_LRC;
  (*N)->ops->mult        = MatMult_LRC;
  (*N)->assembled        = PETSC_TRUE;
  (*N)->cmap->N                = A->cmap->N;
  (*N)->rmap->N                = A->cmap->N;
  (*N)->cmap->n                = A->cmap->n;
  (*N)->rmap->n                = A->cmap->n;
  PetscFunctionReturn(0);
}

