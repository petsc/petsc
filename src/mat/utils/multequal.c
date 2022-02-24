
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

static PetscErrorCode MatMultEqual_Private(Mat A,Mat B,PetscInt n,PetscBool *flg,PetscInt t,PetscBool add)
{
  Vec            Ax = NULL,Bx = NULL,s1 = NULL,s2 = NULL,Ay = NULL, By = NULL;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;
  const char*    sops[] = {"MatMult","MatMultAdd","MatMultTranspose","MatMultTransposeAdd","MatMultHermitianTranspose","MatMultHermitianTransposeAdd"};
  const char*    sop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscCheckSameComm(A,1,B,2);
  PetscValidLogicalCollectiveInt(A,n,3);
  PetscValidPointer(flg,4);
  PetscValidLogicalCollectiveInt(A,t,5);
  PetscValidLogicalCollectiveBool(A,add,6);
  CHKERRQ(MatGetLocalSize(A,&am,&an));
  CHKERRQ(MatGetLocalSize(B,&bm,&bn));
  PetscCheckFalse(am != bm || an != bn,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,bm,an,bn);
  sop  = sops[(add ? 1 : 0) + 2 * t]; /* t = 0 => no transpose, t = 1 => transpose, t = 2 => Hermitian transpose */
  CHKERRQ(PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  if (t) {
    CHKERRQ(MatCreateVecs(A,&s1,&Ax));
    CHKERRQ(MatCreateVecs(B,&s2,&Bx));
  } else {
    CHKERRQ(MatCreateVecs(A,&Ax,&s1));
    CHKERRQ(MatCreateVecs(B,&Bx,&s2));
  }
  if (add) {
    CHKERRQ(VecDuplicate(s1,&Ay));
    CHKERRQ(VecDuplicate(s2,&By));
  }

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    CHKERRQ(VecSetRandom(Ax,rctx));
    CHKERRQ(VecCopy(Ax,Bx));
    if (add) {
      CHKERRQ(VecSetRandom(Ay,rctx));
      CHKERRQ(VecCopy(Ay,By));
    }
    if (t == 1) {
      if (add) {
        CHKERRQ(MatMultTransposeAdd(A,Ax,Ay,s1));
        CHKERRQ(MatMultTransposeAdd(B,Bx,By,s2));
      } else {
        CHKERRQ(MatMultTranspose(A,Ax,s1));
        CHKERRQ(MatMultTranspose(B,Bx,s2));
      }
    } else if (t == 2) {
      if (add) {
        CHKERRQ(MatMultHermitianTransposeAdd(A,Ax,Ay,s1));
        CHKERRQ(MatMultHermitianTransposeAdd(B,Bx,By,s2));
      } else {
        CHKERRQ(MatMultHermitianTranspose(A,Ax,s1));
        CHKERRQ(MatMultHermitianTranspose(B,Bx,s2));
      }
    } else {
      if (add) {
        CHKERRQ(MatMultAdd(A,Ax,Ay,s1));
        CHKERRQ(MatMultAdd(B,Bx,By,s2));
      } else {
        CHKERRQ(MatMult(A,Ax,s1));
        CHKERRQ(MatMult(B,Bx,s2));
      }
    }
    CHKERRQ(VecNorm(s2,NORM_INFINITY,&r2));
    if (r2 < tol) {
      CHKERRQ(VecNorm(s1,NORM_INFINITY,&r1));
    } else {
      CHKERRQ(VecAXPY(s2,none,s1));
      CHKERRQ(VecNorm(s2,NORM_INFINITY,&r1));
      r1  /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      CHKERRQ(PetscInfo(A,"Error: %" PetscInt_FMT "-th %s() %g\n",k,sop,(double)r1));
      break;
    }
  }
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(VecDestroy(&Ax));
  CHKERRQ(VecDestroy(&Bx));
  CHKERRQ(VecDestroy(&Ay));
  CHKERRQ(VecDestroy(&By));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultEqual_Private(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg,PetscBool At,PetscBool Bt)
{
  Vec            Ax,Bx,Cx,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;
  const char*    sops[] = {"MatMatMult","MatTransposeMatMult","MatMatTransposeMult","MatTransposeMatTransposeMult"};
  const char*    sop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscCheckSameComm(A,1,B,2);
  PetscValidHeaderSpecific(C,MAT_CLASSID,3);
  PetscCheckSameComm(A,1,C,3);
  PetscValidLogicalCollectiveInt(A,n,4);
  PetscValidPointer(flg,5);
  PetscValidLogicalCollectiveBool(A,At,6);
  PetscValidLogicalCollectiveBool(B,Bt,7);
  CHKERRQ(MatGetLocalSize(A,&am,&an));
  CHKERRQ(MatGetLocalSize(B,&bm,&bn));
  CHKERRQ(MatGetLocalSize(C,&cm,&cn));
  if (At) { PetscInt tt = an; an = am; am = tt; };
  if (Bt) { PetscInt tt = bn; bn = bm; bm = tt; };
  PetscCheckFalse(an != bm || am != cm || bn != cn,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,an,bm,bn,cm,cn);

  sop  = sops[(At ? 1 : 0) + 2 * (Bt ? 1 : 0)];
  CHKERRQ(PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  if (Bt) {
    CHKERRQ(MatCreateVecs(B,&s1,&Bx));
  } else {
    CHKERRQ(MatCreateVecs(B,&Bx,&s1));
  }
  if (At) {
    CHKERRQ(MatCreateVecs(A,&s2,&Ax));
  } else {
    CHKERRQ(MatCreateVecs(A,&Ax,&s2));
  }
  CHKERRQ(MatCreateVecs(C,&Cx,&s3));

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    CHKERRQ(VecSetRandom(Bx,rctx));
    if (Bt) {
      CHKERRQ(MatMultTranspose(B,Bx,s1));
    } else {
      CHKERRQ(MatMult(B,Bx,s1));
    }
    CHKERRQ(VecCopy(s1,Ax));
    if (At) {
      CHKERRQ(MatMultTranspose(A,Ax,s2));
    } else {
      CHKERRQ(MatMult(A,Ax,s2));
    }
    CHKERRQ(VecCopy(Bx,Cx));
    CHKERRQ(MatMult(C,Cx,s3));

    CHKERRQ(VecNorm(s2,NORM_INFINITY,&r2));
    if (r2 < tol) {
      CHKERRQ(VecNorm(s3,NORM_INFINITY,&r1));
    } else {
      CHKERRQ(VecAXPY(s2,none,s3));
      CHKERRQ(VecNorm(s2,NORM_INFINITY,&r1));
      r1  /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      CHKERRQ(PetscInfo(A,"Error: %" PetscInt_FMT "-th %s %g\n",k,sop,(double)r1));
      break;
    }
  }
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(VecDestroy(&Ax));
  CHKERRQ(VecDestroy(&Bx));
  CHKERRQ(VecDestroy(&Cx));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(VecDestroy(&s3));
  PetscFunctionReturn(0);
}

/*@
   MatMultEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatMultEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,0,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatMultAddEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode  MatMultAddEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,0,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
   MatMultTransposeEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode  MatMultTransposeEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,1,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatMultTransposeAddEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode  MatMultTransposeAddEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,1,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
   MatMultHermitianTransposeEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode  MatMultHermitianTransposeEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,2,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatMultHermitianTransposeAddEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode  MatMultHermitianTransposeAddEqual(Mat A,Mat B,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultEqual_Private(A,B,n,flg,2,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
   MatMatMultEqual - Test A*B*x = C*x for n random vector x

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatMatMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultEqual_Private(A,B,C,n,flg,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatTransposeMatMultEqual - Test A^T*B*x = C*x for n random vector x

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatTransposeMatMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultEqual_Private(A,B,C,n,flg,PETSC_TRUE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatMatTransposeMultEqual - Test A*B^T*x = C*x for n random vector x

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatMatTransposeMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultEqual_Private(A,B,C,n,flg,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProjMultEqual_Private(Mat A,Mat B,Mat C,PetscInt n,PetscBool rart,PetscBool *flg)
{
  Vec            x,v1,v2,v3,v4,Cx,Bx;
  PetscReal      norm_abs,norm_rel,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       i,am,an,bm,bn,cm,cn;
  PetscRandom    rdm;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  CHKERRQ(MatGetLocalSize(A,&am,&an));
  CHKERRQ(MatGetLocalSize(B,&bm,&bn));
  if (rart) { PetscInt t = bm; bm = bn; bn = t; }
  CHKERRQ(MatGetLocalSize(C,&cm,&cn));
  PetscCheckFalse(an != bm || bn != cm || bn != cn,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,an,bm,bn,cm,cn);

  /* Create left vector of A: v2 */
  CHKERRQ(MatCreateVecs(A,&Bx,&v2));

  /* Create right vectors of B: x, v3, v4 */
  if (rart) {
    CHKERRQ(MatCreateVecs(B,&v1,&x));
  } else {
    CHKERRQ(MatCreateVecs(B,&x,&v1));
  }
  CHKERRQ(VecDuplicate(x,&v3));

  CHKERRQ(MatCreateVecs(C,&Cx,&v4));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  *flg = PETSC_TRUE;
  for (i=0; i<n; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(VecCopy(x,Cx));
    CHKERRQ(MatMult(C,Cx,v4));           /* v4 = C*x   */
    if (rart) {
      CHKERRQ(MatMultTranspose(B,x,v1));
    } else {
      CHKERRQ(MatMult(B,x,v1));
    }
    CHKERRQ(VecCopy(v1,Bx));
    CHKERRQ(MatMult(A,Bx,v2));          /* v2 = A*B*x */
    CHKERRQ(VecCopy(v2,v1));
    if (rart) {
      CHKERRQ(MatMult(B,v1,v3)); /* v3 = R*A*R^t*x */
    } else {
      CHKERRQ(MatMultTranspose(B,v1,v3)); /* v3 = Bt*A*B*x */
    }
    CHKERRQ(VecNorm(v4,NORM_2,&norm_abs));
    CHKERRQ(VecAXPY(v4,none,v3));
    CHKERRQ(VecNorm(v4,NORM_2,&norm_rel));

    if (norm_abs > tol) norm_rel /= norm_abs;
    if (norm_rel > tol) {
      *flg = PETSC_FALSE;
      CHKERRQ(PetscInfo(A,"Error: %" PetscInt_FMT "-th Mat%sMult() %g\n",i,rart ? "RARt" : "PtAP",(double)norm_rel));
      break;
    }
  }

  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&Bx));
  CHKERRQ(VecDestroy(&Cx));
  CHKERRQ(VecDestroy(&v1));
  CHKERRQ(VecDestroy(&v2));
  CHKERRQ(VecDestroy(&v3));
  CHKERRQ(VecDestroy(&v4));
  PetscFunctionReturn(0);
}

/*@
   MatPtAPMultEqual - Compares matrix-vector products of C = Bt*A*B

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatPtAPMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatProjMultEqual_Private(A,B,C,n,PETSC_FALSE,flg));
  PetscFunctionReturn(0);
}

/*@
   MatRARtMultEqual - Compares matrix-vector products of C = B*A*B^t

   Collective on Mat

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatRARtMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg)
{
  PetscFunctionBegin;
  CHKERRQ(MatProjMultEqual_Private(A,B,C,n,PETSC_TRUE,flg));
  PetscFunctionReturn(0);
}

/*@
   MatIsLinear - Check if a shell matrix A is a linear operator.

   Collective on Mat

   Input Parameters:
+  A - the shell matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the shell matrix is linear; PETSC_FALSE otherwise.

   Level: intermediate
@*/
PetscErrorCode MatIsLinear(Mat A,PetscInt n,PetscBool *flg)
{
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscScalar    a;
  PetscInt       k;
  PetscReal      norm,normA;
  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(PetscRandomCreate(comm,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  CHKERRQ(MatCreateVecs(A,&x,&s1));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(s1,&s2));

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    CHKERRQ(VecSetRandom(x,rctx));
    CHKERRQ(VecSetRandom(y,rctx));
    if (rank == 0) {
      CHKERRQ(PetscRandomGetValue(rctx,&a));
    }
    CHKERRMPI(MPI_Bcast(&a, 1, MPIU_SCALAR, 0, comm));

    /* s2 = a*A*x + A*y */
    CHKERRQ(MatMult(A,y,s2)); /* s2 = A*y */
    CHKERRQ(MatMult(A,x,s1)); /* s1 = A*x */
    CHKERRQ(VecAXPY(s2,a,s1)); /* s2 = a s1 + s2 */

    /* s1 = A * (a x + y) */
    CHKERRQ(VecAXPY(y,a,x)); /* y = a x + y */
    CHKERRQ(MatMult(A,y,s1));
    CHKERRQ(VecNorm(s1,NORM_INFINITY,&normA));

    CHKERRQ(VecAXPY(s2,-1.0,s1)); /* s2 = - s1 + s2 */
    CHKERRQ(VecNorm(s2,NORM_INFINITY,&norm));
    if (norm/normA > 100.*PETSC_MACHINE_EPSILON) {
      *flg = PETSC_FALSE;
      CHKERRQ(PetscInfo(A,"Error: %" PetscInt_FMT "-th |A*(ax+y) - (a*A*x+A*y)|/|A(ax+y)| %g > tol %g\n",k,(double)(norm/normA),(double)(100.*PETSC_MACHINE_EPSILON)));
      break;
    }
  }
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  PetscFunctionReturn(0);
}
