
#include <petsc-private/matimpl.h>  /*I   "petscmat.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "MatMultEqual"
/*@
   MatMultEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: matrices^equality between
@*/
PetscErrorCode  MatMultEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);

#if defined(PETSC_USE_REAL_SINGLE)
  tol  = 1.e-5;
#endif
  ierr = PetscRandomCreate(((PetscObject)A)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)A)->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)A)->comm,&s1);CHKERRQ(ierr);
  ierr = VecSetSizes(s1,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr = MatMult(B,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol){
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %D-th MatMult() %G\n",k,r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAddEqual"
/*@
   MatMultAddEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: matrices^equality between
@*/
PetscErrorCode  MatMultAddEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(((PetscObject)A)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)A)->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)A)->comm,&s1);CHKERRQ(ierr);
  ierr = VecSetSizes(s1,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&y);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(B,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol){
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %d-th MatMultAdd() %G\n",k,r1);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeEqual"
/*@
   MatMultTransposeEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: matrices^equality between
@*/
PetscErrorCode  MatMultTransposeEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(((PetscObject)A)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)A)->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)A)->comm,&s1);CHKERRQ(ierr);
  ierr = VecSetSizes(s1,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(B,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol){
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %d-th MatMultTranspose() %G\n",k,r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAddEqual"
/*@
   MatMultTransposeAddEqual - Compares matrix-vector products of two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - PETSC_TRUE if the products are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: matrices^equality between
@*/
PetscErrorCode  MatMultTransposeAddEqual(Mat A,Mat B,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(((PetscObject)A)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)A)->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)A)->comm,&s1);CHKERRQ(ierr);
  ierr = VecSetSizes(s1,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&y);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(B,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol){
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %d-th MatMultTransposeAdd() %G\n",k,r1);CHKERRQ(ierr);
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
