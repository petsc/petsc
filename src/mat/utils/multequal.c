
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

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
PetscErrorCode MatMultEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr = MatMult(B,x,s2);CHKERRQ(ierr);
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
      ierr = PetscInfo2(A,"Error: %D-th MatMult() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
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
PetscErrorCode  MatMultAddEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&y);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(B,x,y,s2);CHKERRQ(ierr);
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
      ierr = PetscInfo2(A,"Error: %d-th MatMultAdd() %g\n",k,(double)r1);CHKERRQ(ierr);
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
PetscErrorCode  MatMultTransposeEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol= PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&s1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(B,x,s2);CHKERRQ(ierr);
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
      ierr = PetscInfo2(A,"Error: %d-th MatMultTranspose() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
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
PetscErrorCode  MatMultTransposeAddEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol = PETSC_SQRT_MACHINE_EPSILON; 
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)A),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&s1,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&y);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(B,x,y,s2);CHKERRQ(ierr);
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
      ierr = PetscInfo2(A,"Error: %d-th MatMultTransposeAdd() %g\n",k,(double)r1);CHKERRQ(ierr);
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
  Vec            x,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (an != bm || am != cm || bn != cn) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %D %D %D %D",am,an,bm,bn,cm, cn);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,&s1);CHKERRQ(ierr);
  ierr = MatCreateVecs(C,NULL,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s2,&s3);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(B,x,s1);CHKERRQ(ierr);
    ierr = MatMult(A,s1,s2);CHKERRQ(ierr);
    ierr = MatMult(C,x,s3);CHKERRQ(ierr);

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
      ierr = PetscInfo2(A,"Error: %D-th MatMatMult() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&s3);CHKERRQ(ierr);
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
  Vec            x,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (am != bm || an != cm || bn != cn) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %D %D %D %D",am,an,bm,bn,cm, cn);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,&s1);CHKERRQ(ierr);
  ierr = MatCreateVecs(C,NULL,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s2,&s3);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(B,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,s1,s2);CHKERRQ(ierr);
    ierr = MatMult(C,x,s3);CHKERRQ(ierr);

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
      ierr = PetscInfo2(A,"Error: %D-th MatTransposeMatMult() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&s3);CHKERRQ(ierr);
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
  Vec            x,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (an != bn || am != cm || bm != cn) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %D %D %D %D",am,an,bm,bn,cm, cn);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&s1,&x);CHKERRQ(ierr);
  ierr = MatCreateVecs(C,NULL,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s2,&s3);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMultTranspose(B,x,s1);CHKERRQ(ierr);
    ierr = MatMult(A,s1,s2);CHKERRQ(ierr);
    ierr = MatMult(C,x,s3);CHKERRQ(ierr);

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
      ierr = PetscInfo2(A,"Error: %D-th MatMatTransposeMult() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&s3);CHKERRQ(ierr);
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
  Vec            x,v1,v2,v3,v4;
  PetscReal      norm_abs,norm_rel,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       i,am,an,bm,bn,cm,cn;
  PetscRandom    rdm;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (an != bm || bn != cm || bn != cn) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %D %D %D %D %D %D",am,an,bm,bn,cm,cn);

  /* Create left vector of A: v2 */
  ierr = MatCreateVecs(A,NULL,&v2);CHKERRQ(ierr);

  /* Create right vectors of B: x, v3, v4 */
  ierr = MatCreateVecs(B,&x,&v1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&v3);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&v4);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (i=0; i<n; i++) {
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = MatMult(B,x,v1);CHKERRQ(ierr);
    ierr = MatMult(A,v1,v2);CHKERRQ(ierr);          /* v2 = A*B*x */

    ierr = MatMultTranspose(B,v2,v3);CHKERRQ(ierr); /* v3 = Bt*A*B*x */
    ierr = MatMult(C,x,v4);CHKERRQ(ierr);           /* v4 = C*x   */
    ierr = VecNorm(v4,NORM_2,&norm_abs);CHKERRQ(ierr);
    ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
    ierr = VecNorm(v4,NORM_2,&norm_rel);CHKERRQ(ierr);

    if (norm_abs > tol) norm_rel /= norm_abs;
    if (norm_rel > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %D-th MatPtAPMult() %g\n",i,(double)norm_rel);CHKERRQ(ierr);
      break;
    }
  }

  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = VecDestroy(&v3);CHKERRQ(ierr);
  ierr = VecDestroy(&v4);CHKERRQ(ierr);
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
PetscErrorCode MatIsLinear(Mat A,PetscInt n,PetscBool  *flg)
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
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRandomGetValue(rctx,&a);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(&a, 1, MPIU_SCALAR, 0, comm);CHKERRQ(ierr);

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
      ierr = PetscInfo3(A,"Error: %D-th |A*(ax+y) - (a*A*x+A*y)|/|A(ax+y)| %g > tol %g\n",k,(double)norm/normA,100.*PETSC_MACHINE_EPSILON);CHKERRQ(ierr);
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
