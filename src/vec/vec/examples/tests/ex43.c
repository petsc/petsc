
#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  Vec            *V,t;
  PetscInt       i,j,reps,n=15,k=6;
  PetscRandom    rctx;
  PetscScalar    *val;
  PetscBool      mdot;

  PetscInitialize(&argc,&argv,(char*)0,"");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-k",&k,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mdot",&mdot);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test with %D random vectors of length %D",k,n);CHKERRQ(ierr); 
  if (mdot) ierr = PetscPrintf(PETSC_COMM_WORLD,"(mdot)",k,n);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n",k,n);CHKERRQ(ierr); 
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);
  ierr = VecSetRandom(t,rctx);CHKERRQ(ierr);
  ierr = PetscMalloc(k*sizeof(PetscScalar),&val);CHKERRQ(ierr);
  for (i=0;i<k;i++) { ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr); }
  for (reps=0;reps<20;reps++) {
    for (i=1;i<k;i++) {
      if (mdot) {
        ierr = VecMDot(t,i,V,val);CHKERRQ(ierr);
      } else {
        for (j=0;j<i;j++) {
          ierr = VecDot(t,V[j],&val[j]);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree(val);CHKERRQ(ierr);
  ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
