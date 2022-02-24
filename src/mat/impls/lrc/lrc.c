
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
  PetscMPIInt    size;
  Mat            U,V;

  PetscFunctionBegin;
  U = transpose ? Na->V : Na->U;
  V = transpose ? Na->U : Na->V;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)N),&size));
  if (size == 1) {
    CHKERRQ(MatMultHermitianTranspose(V,x,Na->work1));
    if (Na->c) {
      CHKERRQ(VecPointwiseMult(Na->work1,Na->c,Na->work1));
    }
    if (Na->A) {
      if (transpose) {
        CHKERRQ(MatMultTranspose(Na->A,x,y));
      } else {
        CHKERRQ(MatMult(Na->A,x,y));
      }
      CHKERRQ(MatMultAdd(U,Na->work1,y,y));
    } else {
      CHKERRQ(MatMult(U,Na->work1,y));
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
    CHKERRQ(VecGetLocalVector(y,yl));
    CHKERRQ(MatDenseGetLocalMatrix(U,&Uloc));
    CHKERRQ(MatDenseGetLocalMatrix(V,&Vloc));

    /* multiply the local part of V with the local part of x */
    CHKERRQ(VecGetLocalVectorRead(x,xl));
    CHKERRQ(MatMultHermitianTranspose(Vloc,xl,Na->work1));
    CHKERRQ(VecRestoreLocalVectorRead(x,xl));

    /* form the sum of all the local multiplies: this is work2 = V'*x =
       sum_{all processors} work1 */
    CHKERRQ(VecGetArrayRead(Na->work1,&w1));
    CHKERRQ(VecGetArrayWrite(Na->work2,&w2));
    CHKERRQ(VecGetLocalSize(Na->work1,&nwork));
    CHKERRQ(PetscMPIIntCast(nwork,&mpinwork));
    CHKERRMPI(MPIU_Allreduce(w1,w2,mpinwork,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N)));
    CHKERRQ(VecRestoreArrayRead(Na->work1,&w1));
    CHKERRQ(VecRestoreArrayWrite(Na->work2,&w2));

    if (Na->c) {  /* work2 = C*work2 */
      CHKERRQ(VecPointwiseMult(Na->work2,Na->c,Na->work2));
    }

    if (Na->A) {
      /* form y = A*x or A^t*x */
      if (transpose) {
        CHKERRQ(MatMultTranspose(Na->A,x,y));
      } else {
        CHKERRQ(MatMult(Na->A,x,y));
      }
      /* multiply-add y = y + U*work2 */
      CHKERRQ(MatMultAdd(Uloc,Na->work2,yl,yl));
    } else {
      /* multiply y = U*work2 */
      CHKERRQ(MatMult(Uloc,Na->work2,yl));
    }

    CHKERRQ(VecRestoreLocalVector(y,yl));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_LRC(Mat N,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(MatMult_LRC_kernel(N,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_LRC(Mat N,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(MatMult_LRC_kernel(N,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_LRC(Mat N)
{
  Mat_LRC        *Na = (Mat_LRC*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&Na->A));
  CHKERRQ(MatDestroy(&Na->U));
  CHKERRQ(MatDestroy(&Na->V));
  CHKERRQ(VecDestroy(&Na->c));
  CHKERRQ(VecDestroy(&Na->work1));
  CHKERRQ(VecDestroy(&Na->work2));
  CHKERRQ(VecDestroy(&Na->xl));
  CHKERRQ(VecDestroy(&Na->yl));
  CHKERRQ(PetscFree(N->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatLRCGetMats_C",NULL));
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
  PetscFunctionBegin;
  CHKERRQ(PetscUseMethod(N,"MatLRCGetMats_C",(Mat,Mat*,Mat*,Vec*,Mat*),(N,A,U,c,V)));
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
  CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)U,&match,MATSEQDENSE,MATMPIDENSE,""));
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Matrix U must be of type dense, found %s",((PetscObject)U)->type_name);
  CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)V,&match,MATSEQDENSE,MATMPIDENSE,""));
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Matrix V must be of type dense, found %s",((PetscObject)V)->type_name);
  CHKERRQ(PetscStrcmp(U->defaultvectype,V->defaultvectype,&match));
  PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_WRONG,"Matrix U and V must have the same VecType %s != %s",U->defaultvectype,V->defaultvectype);
  if (A) {
    CHKERRQ(PetscStrcmp(A->defaultvectype,U->defaultvectype,&match));
    PetscCheckFalse(!match,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_WRONG,"Matrix A and U must have the same VecType %s != %s",A->defaultvectype,U->defaultvectype);
  }

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)U),&size));
  CHKERRQ(MatGetSize(U,NULL,&k));
  CHKERRQ(MatGetSize(V,NULL,&k1));
  PetscCheckFalse(k != k1,PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_INCOMP,"U and V have different number of columns (%" PetscInt_FMT " vs %" PetscInt_FMT ")",k,k1);
  CHKERRQ(MatGetLocalSize(U,&m,NULL));
  CHKERRQ(MatGetLocalSize(V,&n,NULL));
  if (A) {
    CHKERRQ(MatGetLocalSize(A,&m1,&n1));
    PetscCheckFalse(m != m1,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local dimensions of U %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",m,m1);
    PetscCheckFalse(n != n1,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local dimensions of V %" PetscInt_FMT " and A %" PetscInt_FMT " do not match",n,n1);
  }
  if (c) {
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)c),&csize));
    CHKERRQ(VecGetSize(c,&k1));
    PetscCheckFalse(k != k1,PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"The length of c %" PetscInt_FMT " does not match the number of columns of U and V (%" PetscInt_FMT ")",k1,k);
    PetscCheckFalse(csize != 1 && csize != size, PetscObjectComm((PetscObject)c),PETSC_ERR_ARG_INCOMP,"U and c must have the same communicator size %d != %d",size,csize);
  }

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)U),N));
  CHKERRQ(MatSetSizes(*N,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetVecType(*N,U->defaultvectype));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)*N,MATLRC));
  /* Flag matrix as symmetric if A is symmetric and U == V */
  CHKERRQ(MatSetOption(*N,MAT_SYMMETRIC,(PetscBool)((A ? A->symmetric : PETSC_TRUE) && U == V)));

  CHKERRQ(PetscNewLog(*N,&Na));
  (*N)->data = (void*)Na;
  Na->A      = A;
  Na->U      = U;
  Na->c      = c;
  Na->V      = V;

  CHKERRQ(PetscObjectReference((PetscObject)A));
  CHKERRQ(PetscObjectReference((PetscObject)Na->U));
  CHKERRQ(PetscObjectReference((PetscObject)Na->V));
  CHKERRQ(PetscObjectReference((PetscObject)c));

  CHKERRQ(MatDenseGetLocalMatrix(Na->U,&Uloc));
  CHKERRQ(MatCreateVecs(Uloc,&Na->work1,NULL));
  if (size != 1) {
    Mat Vloc;

    if (Na->c && csize != 1) { /* scatter parallel vector to sequential */
      VecScatter sct;

      CHKERRQ(VecScatterCreateToAll(Na->c,&sct,&c));
      CHKERRQ(VecScatterBegin(sct,Na->c,c,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(sct,Na->c,c,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&sct));
      CHKERRQ(VecDestroy(&Na->c));
      CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)c));
      Na->c = c;
    }
    CHKERRQ(MatDenseGetLocalMatrix(Na->V,&Vloc));
    CHKERRQ(VecDuplicate(Na->work1,&Na->work2));
    CHKERRQ(MatCreateVecs(Vloc,NULL,&Na->xl));
    CHKERRQ(MatCreateVecs(Uloc,NULL,&Na->yl));
  }
  CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->work1));
  CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->work1));
  CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->xl));
  CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)Na->yl));

  /* Internally create a scaling vector if roottypes do not match */
  if (Na->c) {
    VecType rt1,rt2;

    CHKERRQ(VecGetRootType_Private(Na->work1,&rt1));
    CHKERRQ(VecGetRootType_Private(Na->c,&rt2));
    CHKERRQ(PetscStrcmp(rt1,rt2,&match));
    if (!match) {
      CHKERRQ(VecDuplicate(Na->c,&c));
      CHKERRQ(VecCopy(Na->c,c));
      CHKERRQ(VecDestroy(&Na->c));
      CHKERRQ(PetscLogObjectParent((PetscObject)*N,(PetscObject)c));
      Na->c = c;
    }
  }

  (*N)->ops->destroy       = MatDestroy_LRC;
  (*N)->ops->mult          = MatMult_LRC;
  (*N)->ops->multtranspose = MatMultTranspose_LRC;

  (*N)->assembled    = PETSC_TRUE;
  (*N)->preallocated = PETSC_TRUE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatLRCGetMats_C",MatLRCGetMats_LRC));
  CHKERRQ(MatSetUp(*N));
  PetscFunctionReturn(0);
}
