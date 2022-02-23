
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

PETSC_EXTERN PetscErrorCode VecGetRootType_Private(Vec,VecType*);

typedef struct {
  Mat A;           /* sparse matrix */
  Mat U,V;         /* dense tall-skinny matrices */
  Vec c;           /* sequential vector containing the diagonal of C */
  Vec work1,work2; /* sequential vectors that hold partial products */
  Vec xl,yl;       /* auxiliary sequential vectors for matmult operation */
} Mat_LRC;

static PetscErrorCode MatMult_LRC_kernel(Mat N,Vec x,Vec y,PetscBool transpose)
{
  Mat_LRC        *Na = (Mat_LRC*)N->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            U,V;

  PetscFunctionBegin;
  U = transpose ? Na->V : Na->U;
  V = transpose ? Na->U : Na->V;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)N),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = MatMultHermitianTranspose(V,x,Na->work1);CHKERRQ(ierr);
    if (Na->c) {
      ierr = VecPointwiseMult(Na->work1,Na->c,Na->work1);CHKERRQ(ierr);
    }
    if (Na->A) {
      if (transpose) {
        ierr = MatMultTranspose(Na->A,x,y);CHKERRQ(ierr);
      } else {
        ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
      }
      ierr = MatMultAdd(U,Na->work1,y,y);CHKERRQ(ierr);
    } else {
      ierr = MatMult(U,Na->work1,y);CHKERRQ(ierr);
    }
  } else {
    Mat               Uloc,Vloc;
    Vec               yl,xl;
    const PetscScalar *w1;
    PetscScalar       *w2;
    PetscInt          nwork;
    PetscMPIInt       mpinwork;

    xl = transpose ? Na->yl : Na->xl;
    yl = transpose ? Na->xl : Na->yl;
    ierr = VecGetLocalVector(y,yl);CHKERRQ(ierr);
    ierr = MatDenseGetLocalMatrix(U,&Uloc);CHKERRQ(ierr);
    ierr = MatDenseGetLocalMatrix(V,&Vloc);CHKERRQ(ierr);

    /* multiply the local part of V with the local part of x */
    ierr = VecGetLocalVectorRead(x,xl);CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(Vloc,xl,Na->work1);CHKERRQ(ierr);
    ierr = VecRestoreLocalVectorRead(x,xl);CHKERRQ(ierr);

    /* form the sum of all the local multiplies: this is work2 = V'*x =
       sum_{all processors} work1 */
    ierr = VecGetArrayRead(Na->work1,&w1);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(Na->work2,&w2);CHKERRQ(ierr);
    ierr = VecGetLocalSize(Na->work1,&nwork);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(nwork,&mpinwork);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(w1,w2,mpinwork,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N));CHKERRMPI(ierr);
    ierr = VecRestoreArrayRead(Na->work1,&w1);CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(Na->work2,&w2);CHKERRQ(ierr);

    if (Na->c) {  /* work2 = C*work2 */
      ierr = VecPointwiseMult(Na->work2,Na->c,Na->work2);CHKERRQ(ierr);
    }

    if (Na->A) {
      /* form y = A*x or A^t*x */
      if (transpose) {
        ierr = MatMultTranspose(Na->A,x,y);CHKERRQ(ierr);
      } else {
        ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
      }
      /* multiply-add y = y + U*work2 */
      ierr = MatMultAdd(Uloc,Na->work2,yl,yl);CHKERRQ(ierr);
    } else {
      /* multiply y = U*work2 */
      ierr = MatMult(Uloc,Na->work2,yl);CHKERRQ(ierr);
    }

    ierr = VecRestoreLocalVector(y,yl);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_LRC(Mat N,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult_LRC_kernel(N,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_LRC(Mat N,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult_LRC_kernel(N,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_LRC(Mat N)
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
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatLRCGetMats_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLRCGetMats_LRC(Mat N,Mat *A,Mat *U,Vec *c,Mat *V)
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
PetscErrorCode MatLRCGetMats(Mat N,Mat *A,Mat *U,Vec *c,Mat *V)
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
-  c    - a vector containing the diagonal of C (can be NULL)

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
   If a sequential c vector is used for a parallel matrix,
   PETSc assumes that the values of the vector are consistently set across processors.

   Level: intermediate

.seealso: MatLRCGetMats()
@*/
PetscErrorCode MatCreateLRC(Mat A,Mat U,Vec c,Mat V,Mat *N)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n,k,m1,n1,k1;
  Mat_LRC        *Na;
  Mat            Uloc;
  PetscMPIInt    size, csize = 0;

  PetscFunctionBegin;
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(U,MAT_CLASSID,2);
  if (c) PetscValidHeaderSpecific(c,VEC_CLASSID,3);
  if (V) {
    PetscValidHeaderSpecific(V,MAT_CLASSID,4);
    PetscCheckSameComm(U,2,V,4);
  }
  if (A) PetscCheckSameComm(A,1,U,2);

  if (!V) V = U;
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)U,&match,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Matrix U must be of type dense, found %s",((PetscObject)U)->type_name);
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)V,&match,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Matrix V must be of type dense, found %s",((PetscObject)V)->type_name);
  ierr = PetscStrcmp(U->defaultvectype,V->defaultvectype,&match);CHKERRQ(ierr);
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_WRONG,"Matrix U and V must have the same VecType %s != %s",U->defaultvectype,V->defaultvectype);
  if (A) {
    ierr = PetscStrcmp(A->defaultvectype,U->defaultvectype,&match);CHKERRQ(ierr);
    PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_WRONG,"Matrix A and U must have the same VecType %s != %s",A->defaultvectype,U->defaultvectype);
  }

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)U),&size);CHKERRMPI(ierr);
  ierr = MatGetSize(U,NULL,&k);CHKERRQ(ierr);
  ierr = MatGetSize(V,NULL,&k1);CHKERRQ(ierr);
  PetscCheckFalse(k != k1,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"U and V have different number of columns (%" PetscInt_FMT " vs %" PetscInt_FMT ")",k,k1);
  ierr = MatGetLocalSize(U,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(V,&n,NULL);CHKERRQ(ierr);
  if (A) {
    ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
    PetscCheckFalse(m != m1,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local dimensions of U %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",m,m1);
    PetscCheckFalse(n != n1,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local dimensions of V %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",n,n1);
  }
  if (c) {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)c),&csize);CHKERRMPI(ierr);
    ierr = VecGetSize(c,&k1);CHKERRQ(ierr);
    PetscCheckFalse(k != k1,PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"The length of c %" PetscInt_FMT " does not match the number of columns of U and V (%" PetscInt_FMT ")",k1,k);
    PetscCheckFalse(csize != 1 && csize != size, PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"U and c must have the same communicator size %d != %d",size,csize);
  }

  ierr = MatCreate(PetscObjectComm((PetscObject)U),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetVecType(*N,U->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATLRC);CHKERRQ(ierr);
  /* Flag matrix as symmetric if A is symmetric and U == V */
  ierr = MatSetOption(*N,MAT_SYMMETRIC,(PetscBool)((A ? A->symmetric : PETSC_TRUE) && U == V));CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*)Na;
  Na->A      = A;
  Na->U      = U;
  Na->c      = c;
  Na->V      = V;

  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Na->U);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Na->V);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);

  ierr = MatDenseGetLocalMatrix(Na->U,&Uloc);CHKERRQ(ierr);
  ierr = MatCreateVecs(Uloc,&Na->work1,NULL);CHKERRQ(ierr);
  if (size != 1) {
    Mat Vloc;

    if (Na->c && csize != 1) { /* scatter parallel vector to sequential */
      VecScatter sct;

      ierr = VecScatterCreateToAll(Na->c,&sct,&c);CHKERRQ(ierr);
      ierr = VecScatterBegin(sct,Na->c,c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(sct,Na->c,c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&sct);CHKERRQ(ierr);
      ierr = VecDestroy(&Na->c);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)c);CHKERRQ(ierr);
      Na->c = c;
    }
    ierr = MatDenseGetLocalMatrix(Na->V,&Vloc);CHKERRQ(ierr);
    ierr = VecDuplicate(Na->work1,&Na->work2);CHKERRQ(ierr);
    ierr = MatCreateVecs(Vloc,NULL,&Na->xl);CHKERRQ(ierr);
    ierr = MatCreateVecs(Uloc,NULL,&Na->yl);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->work1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->work1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->xl);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->yl);CHKERRQ(ierr);

  /* Internally create a scaling vector if roottypes do not match */
  if (Na->c) {
    VecType rt1,rt2;

    ierr = VecGetRootType_Private(Na->work1,&rt1);CHKERRQ(ierr);
    ierr = VecGetRootType_Private(Na->c,&rt2);CHKERRQ(ierr);
    ierr = PetscStrcmp(rt1,rt2,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = VecDuplicate(Na->c,&c);CHKERRQ(ierr);
      ierr = VecCopy(Na->c,c);CHKERRQ(ierr);
      ierr = VecDestroy(&Na->c);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)*N,(PetscObject)c);CHKERRQ(ierr);
      Na->c = c;
    }
  }

  (*N)->ops->destroy       = MatDestroy_LRC;
  (*N)->ops->mult          = MatMult_LRC;
  (*N)->ops->multtranspose = MatMultTranspose_LRC;

  (*N)->assembled    = PETSC_TRUE;
  (*N)->preallocated = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatLRCGetMats_C",MatLRCGetMats_LRC);CHKERRQ(ierr);
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
