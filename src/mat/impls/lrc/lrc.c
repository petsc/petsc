
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat         A;           /* sparse matrix */
  Mat         U,V;         /* dense tall-skinny matrices */
  Vec         c;           /* sequential vector containing the diagonal of C */
  Vec         work1,work2; /* sequential vectors that hold partial products */
  PetscMPIInt nwork;       /* length of work vectors */
  Vec         xl,yl;       /* auxiliary sequential vectors for matmult operation */
} Mat_LRC;

PetscErrorCode MatMult_LRC(Mat N,Vec x,Vec y)
{
  Mat_LRC           *Na = (Mat_LRC*)N->data;
  Mat               Uloc,Vloc;
  PetscErrorCode    ierr;
  PetscScalar       *w1,*w2;
  const PetscScalar *a;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&a);CHKERRQ(ierr);
  ierr = VecPlaceArray(Na->xl,a);CHKERRQ(ierr);
  ierr = VecGetLocalVector(y,Na->yl);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(Na->U,&Uloc);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(Na->V,&Vloc);CHKERRQ(ierr);

  /* multiply the local part of V with the local part of x */
#if defined(PETSC_USE_COMPLEX)
  ierr = MatMultHermitianTranspose(Vloc,Na->xl,Na->work1);CHKERRQ(ierr);
#else
  ierr = MatMultTranspose(Vloc,Na->xl,Na->work1);CHKERRQ(ierr);
#endif

  /* form the sum of all the local multiplies: this is work2 = V'*x =
     sum_{all processors} work1 */
  ierr = VecGetArray(Na->work1,&w1);CHKERRQ(ierr);
  ierr = VecGetArray(Na->work2,&w2);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(w1,w2,Na->nwork,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N));CHKERRMPI(ierr);
  ierr = VecRestoreArray(Na->work1,&w1);CHKERRQ(ierr);
  ierr = VecRestoreArray(Na->work2,&w2);CHKERRQ(ierr);

  if (Na->c) {  /* work2 = C*work2 */
    ierr = VecPointwiseMult(Na->work2,Na->c,Na->work2);CHKERRQ(ierr);
  }

  if (Na->A) {
    /* form y = A*x */
    ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
    /* multiply-add y = y + U*work2 */
    ierr = MatMultAdd(Uloc,Na->work2,Na->yl,Na->yl);CHKERRQ(ierr);
  } else {
    /* multiply y = U*work2 */
    ierr = MatMult(Uloc,Na->work2,Na->yl);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(x,&a);CHKERRQ(ierr);
  ierr = VecResetArray(Na->xl);CHKERRQ(ierr);
  ierr = VecRestoreLocalVector(y,Na->yl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_LRC(Mat N)
{
  Mat_LRC        *Na = (Mat_LRC*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->U);CHKERRQ(ierr);
  ierr = MatDestroy(&Na->V);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->c);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work1);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->work2);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->yl);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatLRCGetMats_C",0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLRCGetMats_LRC(Mat N,Mat *A,Mat *U,Vec *c,Mat *V)
{
  Mat_LRC *Na = (Mat_LRC*)N->data;

  PetscFunctionBegin;
  if (A) *A = Na->A;
  if (U) *U = Na->U;
  if (c) *c = Na->c;
  if (V) *V = Na->V;
  PetscFunctionReturn(0);
}

/*@
   MatLRCGetMats - Returns the constituents of an LRC matrix

   Collective on Mat

   Input Parameter:
.  N - matrix of type LRC

   Output Parameters:
+  A - the (sparse) matrix
.  U - first dense rectangular (tall and skinny) matrix
.  c - a sequential vector containing the diagonal of C
-  V - second dense rectangular (tall and skinny) matrix

   Note:
   The returned matrices need not be destroyed by the caller.

   Level: intermediate

.seealso: MatCreateLRC()
@*/
PetscErrorCode  MatLRCGetMats(Mat N,Mat *A,Mat *U,Vec *c,Mat *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(N,"MatLRCGetMats_C",(Mat,Mat*,Mat*,Vec*,Mat*),(N,A,U,c,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateLRC - Creates a new matrix object that behaves like A + U*C*V'

   Collective on Mat

   Input Parameters:
+  A    - the (sparse) matrix (can be NULL)
.  U, V - two dense rectangular (tall and skinny) matrices
-  c    - a sequential vector containing the diagonal of C (can be NULL)

   Output Parameter:
.  N    - the matrix that represents A + U*C*V'

   Notes:
   The matrix A + U*C*V' is not formed! Rather the new matrix
   object performs the matrix-vector product by first multiplying by
   A and then adding the other term.

   C is a diagonal matrix (represented as a vector) of order k,
   where k is the number of columns of both U and V.

   If A is NULL then the new object behaves like a low-rank matrix U*C*V'.

   Use V=U (or V=NULL) for a symmetric low-rank correction, A + U*C*U'.

   If c is NULL then the low-rank correction is just U*V'.

   Level: intermediate

.seealso: MatLRCGetMats()
@*/
PetscErrorCode MatCreateLRC(Mat A,Mat U,Vec c,Mat V,Mat *N)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n,k,m1,n1,k1;
  Mat_LRC        *Na;

  PetscFunctionBegin;
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(U,MAT_CLASSID,2);
  if (c) PetscValidHeaderSpecific(c,VEC_CLASSID,3);
  if (V) PetscValidHeaderSpecific(V,MAT_CLASSID,4);
  else V=U;
  if (A) PetscCheckSameComm(A,1,U,2);
  PetscCheckSameComm(U,2,V,4);

  ierr = PetscObjectTypeCompareAny((PetscObject)U,&match,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Matrix U must be of type dense");
  if (V) {
    ierr = PetscObjectTypeCompareAny((PetscObject)V,&match,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (!match) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Matrix V must be of type dense");
  }

  ierr = MatGetSize(U,NULL,&k);CHKERRQ(ierr);
  ierr = MatGetSize(V,NULL,&k1);CHKERRQ(ierr);
  if (k!=k1) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"U and V have different number of columns (%" PetscInt_FMT " vs %" PetscInt_FMT ")",k,k1);
  ierr = MatGetLocalSize(U,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(V,&n,NULL);CHKERRQ(ierr);
  if (A) {
    ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
    if (m!=m1) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"Local dimensions of U %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",m,m1);
    if (n!=n1) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Local dimensions of V %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",n,n1);
  }
  if (c) {
    ierr = VecGetSize(c,&k1);CHKERRQ(ierr);
    if (k!=k1) SETERRQ(PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"The length of c %" PetscInt_FMT " does not match the number of columns of U and V (%" PetscInt_FMT ")",k1,k);
    ierr = VecGetLocalSize(c,&k1);CHKERRQ(ierr);
    if (k!=k1) SETERRQ(PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"c must be a sequential vector");
  }

  ierr = MatCreate(PetscObjectComm((PetscObject)U),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATLRC);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*)Na;
  Na->A      = A;
  Na->U      = U;
  Na->c      = c;
  Na->V      = V;

  if (A) { ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)Na->U);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Na->V);CHKERRQ(ierr);
  if (c) { ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr); }

  ierr = VecCreateSeq(PETSC_COMM_SELF,U->cmap->N,&Na->work1);CHKERRQ(ierr);
  ierr = VecDuplicate(Na->work1,&Na->work2);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(U->cmap->N,&Na->nwork);CHKERRQ(ierr);

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,V->rmap->n,NULL,&Na->xl);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,U->rmap->n,NULL,&Na->yl);CHKERRQ(ierr);

  (*N)->ops->destroy = MatDestroy_LRC;
  (*N)->ops->mult    = MatMult_LRC;
  (*N)->assembled    = PETSC_TRUE;
  (*N)->preallocated = PETSC_TRUE;
  (*N)->cmap->N      = V->rmap->N;
  (*N)->rmap->N      = U->rmap->N;
  (*N)->cmap->n      = V->rmap->n;
  (*N)->rmap->n      = U->rmap->n;

  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatLRCGetMats_C",MatLRCGetMats_LRC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
