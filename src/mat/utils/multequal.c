
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

static PetscErrorCode MatMultEqual_Private(Mat A,Mat B,PetscInt n,PetscBool *flg,PetscInt t,PetscBool add)
{
  PetscErrorCode ierr;
  Vec            Ax = NULL,Bx = NULL,s1 = NULL,s2 = NULL,Ay = NULL, By = NULL;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;
  const char*    sops[] = {"MatMult","MatMultAdd","MatMultTranspose","MatMultTransposeAdd","MatMultHermitianTranspose","MatMultHermitianTranposeAdd"};
  const char*    sop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscCheckSameComm(A,1,B,2);
  PetscValidLogicalCollectiveInt(A,n,3);
  PetscValidPointer(flg,4);
  PetscValidLogicalCollectiveInt(A,t,5);
  PetscValidLogicalCollectiveBool(A,add,6);
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (PetscUnlikely(am != bm || an != bn)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,bm,an,bn);
  sop  = sops[(add ? 1 : 0) + 2 * t]; /* t = 0 => no transpose, t = 1 => transpose, t = 2 => Hermitian transpose */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  if (t) {
    ierr = MatCreateVecs(A,&s1,&Ax);CHKERRQ(ierr);
    ierr = MatCreateVecs(B,&s2,&Bx);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(A,&Ax,&s1);CHKERRQ(ierr);
    ierr = MatCreateVecs(B,&Bx,&s2);CHKERRQ(ierr);
  }
  if (add) {
    ierr = VecDuplicate(s1,&Ay);CHKERRQ(ierr);
    ierr = VecDuplicate(s2,&By);CHKERRQ(ierr);
  }

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(Ax,rctx);CHKERRQ(ierr);
    ierr = VecCopy(Ax,Bx);CHKERRQ(ierr);
    if (add) {
      ierr = VecSetRandom(Ay,rctx);CHKERRQ(ierr);
      ierr = VecCopy(Ay,By);CHKERRQ(ierr);
    }
    if (t == 1) {
      if (add) {
        ierr = MatMultTransposeAdd(A,Ax,Ay,s1);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(B,Bx,By,s2);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(A,Ax,s1);CHKERRQ(ierr);
        ierr = MatMultTranspose(B,Bx,s2);CHKERRQ(ierr);
      }
    } else if (t == 2) {
      if (add) {
        ierr = MatMultHermitianTransposeAdd(A,Ax,Ay,s1);CHKERRQ(ierr);
        ierr = MatMultHermitianTransposeAdd(B,Bx,By,s2);CHKERRQ(ierr);
      } else {
        ierr = MatMultHermitianTranspose(A,Ax,s1);CHKERRQ(ierr);
        ierr = MatMultHermitianTranspose(B,Bx,s2);CHKERRQ(ierr);
      }
    } else {
      if (add) {
        ierr = MatMultAdd(A,Ax,Ay,s1);CHKERRQ(ierr);
        ierr = MatMultAdd(B,Bx,By,s2);CHKERRQ(ierr);
      } else {
        ierr = MatMult(A,Ax,s1);CHKERRQ(ierr);
        ierr = MatMult(B,Bx,s2);CHKERRQ(ierr);
      }
    }
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol) {
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1  /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo(A,"Error: %" PetscInt_FMT "-th %s() %g\n",k,sop,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&Ay);CHKERRQ(ierr);
  ierr = VecDestroy(&By);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultEqual_Private(Mat A,Mat B,Mat C,PetscInt n,PetscBool *flg,PetscBool At,PetscBool Bt)
{
  PetscErrorCode ierr;
  Vec            Ax,Bx,Cx,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;
  const char*    sops[] = {"MatMatMult","MatTransposeMatMult","MatMatTransposeMult","MatTransposeMatTranposeMult"};
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
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (At) { PetscInt tt = an; an = am; am = tt; };
  if (Bt) { PetscInt tt = bn; bn = bm; bm = tt; };
  if (PetscUnlikely(an != bm || am != cm || bn != cn)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,an,bm,bn,cm,cn);

  sop  = sops[(At ? 1 : 0) + 2 * (Bt ? 1 : 0)];
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  if (Bt) {
    ierr = MatCreateVecs(B,&s1,&Bx);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(B,&Bx,&s1);CHKERRQ(ierr);
  }
  if (At) {
    ierr = MatCreateVecs(A,&s2,&Ax);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(A,&Ax,&s2);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(C,&Cx,&s3);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(Bx,rctx);CHKERRQ(ierr);
    if (Bt) {
      ierr = MatMultTranspose(B,Bx,s1);CHKERRQ(ierr);
    } else {
      ierr = MatMult(B,Bx,s1);CHKERRQ(ierr);
    }
    ierr = VecCopy(s1,Ax);CHKERRQ(ierr);
    if (At) {
      ierr = MatMultTranspose(A,Ax,s2);CHKERRQ(ierr);
    } else {
      ierr = MatMult(A,Ax,s2);CHKERRQ(ierr);
    }
    ierr = VecCopy(Bx,Cx);CHKERRQ(ierr);
    ierr = MatMult(C,Cx,s3);CHKERRQ(ierr);

    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol) {
      ierr = VecNorm(s3,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s3);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1  /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo(A,"Error: %" PetscInt_FMT "-th %s %g\n",k,sop,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&Cx);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&s3);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,0,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,0,PETSC_TRUE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,1,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,1,PETSC_TRUE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,2,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultEqual_Private(A,B,n,flg,2,PETSC_TRUE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultEqual_Private(A,B,C,n,flg,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultEqual_Private(A,B,C,n,flg,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultEqual_Private(A,B,C,n,flg,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProjMultEqual_Private(Mat A,Mat B,Mat C,PetscInt n,PetscBool rart,PetscBool *flg)
{
  PetscErrorCode ierr;
  Vec            x,v1,v2,v3,v4,Cx,Bx;
  PetscReal      norm_abs,norm_rel,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       i,am,an,bm,bn,cm,cn;
  PetscRandom    rdm;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (rart) { PetscInt t = bm; bm = bn; bn = t; }
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (PetscUnlikely(an != bm || bn != cm || bn != cn)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,am,an,bm,bn,cm,cn);

  /* Create left vector of A: v2 */
  ierr = MatCreateVecs(A,&Bx,&v2);CHKERRQ(ierr);

  /* Create right vectors of B: x, v3, v4 */
  if (rart) {
    ierr = MatCreateVecs(B,&v1,&x);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(B,&x,&v1);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x,&v3);CHKERRQ(ierr);

  ierr = MatCreateVecs(C,&Cx,&v4);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (i=0; i<n; i++) {
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = VecCopy(x,Cx);CHKERRQ(ierr);
    ierr = MatMult(C,Cx,v4);CHKERRQ(ierr);           /* v4 = C*x   */
    if (rart) {
      ierr = MatMultTranspose(B,x,v1);CHKERRQ(ierr);
    } else {
      ierr = MatMult(B,x,v1);CHKERRQ(ierr);
    }
    ierr = VecCopy(v1,Bx);CHKERRQ(ierr);
    ierr = MatMult(A,Bx,v2);CHKERRQ(ierr);          /* v2 = A*B*x */
    ierr = VecCopy(v2,v1);CHKERRQ(ierr);
    if (rart) {
      ierr = MatMult(B,v1,v3);CHKERRQ(ierr); /* v3 = R*A*R^t*x */
    } else {
      ierr = MatMultTranspose(B,v1,v3);CHKERRQ(ierr); /* v3 = Bt*A*B*x */
    }
    ierr = VecNorm(v4,NORM_2,&norm_abs);CHKERRQ(ierr);
    ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
    ierr = VecNorm(v4,NORM_2,&norm_rel);CHKERRQ(ierr);

    if (norm_abs > tol) norm_rel /= norm_abs;
    if (norm_rel > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo(A,"Error: %" PetscInt_FMT "-th Mat%sMult() %g\n",i,rart ? "RARt" : "PtAP",(double)norm_rel);CHKERRQ(ierr);
      break;
    }
  }

  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&Cx);CHKERRQ(ierr);
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = VecDestroy(&v3);CHKERRQ(ierr);
  ierr = VecDestroy(&v4);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatProjMultEqual_Private(A,B,C,n,PETSC_FALSE,flg);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatProjMultEqual_Private(A,B,C,n,PETSC_TRUE,flg);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscScalar    a;
  PetscInt       k;
  PetscReal      norm,normA;
  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = PetscRandomGetValue(rctx,&a);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(&a, 1, MPIU_SCALAR, 0, comm);CHKERRMPI(ierr);

    /* s2 = a*A*x + A*y */
    ierr = MatMult(A,y,s2);CHKERRQ(ierr); /* s2 = A*y */
    ierr = MatMult(A,x,s1);CHKERRQ(ierr); /* s1 = A*x */
    ierr = VecAXPY(s2,a,s1);CHKERRQ(ierr); /* s2 = a s1 + s2 */

    /* s1 = A * (a x + y) */
    ierr = VecAXPY(y,a,x);CHKERRQ(ierr); /* y = a x + y */
    ierr = MatMult(A,y,s1);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_INFINITY,&normA);CHKERRQ(ierr);

    ierr = VecAXPY(s2,-1.0,s1);CHKERRQ(ierr); /* s2 = - s1 + s2 */
    ierr = VecNorm(s2,NORM_INFINITY,&norm);CHKERRQ(ierr);
    if (norm/normA > 100.*PETSC_MACHINE_EPSILON) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo(A,"Error: %" PetscInt_FMT "-th |A*(ax+y) - (a*A*x+A*y)|/|A(ax+y)| %g > tol %g\n",k,(double)(norm/normA),(double)(100.*PETSC_MACHINE_EPSILON));CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
