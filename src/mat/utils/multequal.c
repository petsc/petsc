
#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

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
PetscErrorCode MatMultEqual(Mat A,Mat B,PetscInt n,PetscTruth *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(A->comm,RANDOM_DEFAULT,&rctx);CHKERRQ(ierr);
  ierr = VecCreate(A->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  
  ierr = VecCreate(A->comm,&s1);CHKERRQ(ierr);
  ierr = VecSetSizes(s1,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);
  
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(rctx,x);CHKERRQ(ierr);
    ierr = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr = MatMult(B,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      *flg = PETSC_FALSE;
    } else {
      *flg = PETSC_TRUE;
    }
  }
  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
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
PetscErrorCode MatMultAddEqual(Mat A,Mat B,PetscInt n,PetscTruth *flg)
{
  PetscErrorCode ierr;
  Vec            x,y,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=1.e-10;
  PetscInt       am,an,bm,bn,k;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);
  ierr = PetscRandomCreate(A->comm,RANDOM_DEFAULT,&rctx);CHKERRQ(ierr);
  ierr = VecCreate(A->comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,an,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);
  
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(rctx,x);CHKERRQ(ierr);
    ierr = VecSetRandom(rctx,y);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(B,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      *flg = PETSC_FALSE;
    } else {
      *flg = PETSC_TRUE;
    }
  }
  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
