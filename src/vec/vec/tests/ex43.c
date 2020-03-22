static char help[] = "Tests VecMDot(),VecDot(),VecMTDot(), and VecTDot()\n";


#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            *V,t;
  PetscInt       i,j,reps,n=15,k=6;
  PetscRandom    rctx;
  PetscScalar    *val_dot,*val_mdot,*tval_dot,*tval_mdot;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test with %D random vectors of length %D",k,n);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n",k,n);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);
  ierr = VecSetRandom(t,rctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(k,&val_dot);CHKERRQ(ierr);
  ierr = PetscMalloc1(k,&val_mdot);CHKERRQ(ierr);
  ierr = PetscMalloc1(k,&tval_dot);CHKERRQ(ierr);
  ierr = PetscMalloc1(k,&tval_mdot);CHKERRQ(ierr);
  for (i=0; i<k; i++) { ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr); }
  for (reps=0; reps<20; reps++) {
    for (i=1; i<k; i++) {
      ierr = VecMDot(t,i,V,val_mdot);CHKERRQ(ierr);
      ierr = VecMTDot(t,i,V,tval_mdot);CHKERRQ(ierr);
      for (j=0;j<i;j++) {
        ierr = VecDot(t,V[j],&val_dot[j]);CHKERRQ(ierr);
        ierr = VecTDot(t,V[j],&tval_dot[j]);CHKERRQ(ierr);
      }
      /* Check result */
      for (j=0;j<i;j++) {
        if (PetscAbsScalar(val_mdot[j] - val_dot[j])/PetscAbsScalar(val_dot[j]) > 1e-5) {
          ierr = PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%D, j=%D, val_mdot[j]=%g, val_dot[j]=%g\n",i,j,(double)PetscAbsScalar(val_mdot[j]), (double)PetscAbsScalar(val_dot[j]));CHKERRQ(ierr);
          break;
        }
        if (PetscAbsScalar(tval_mdot[j] - tval_dot[j])/PetscAbsScalar(tval_dot[j]) > 1e-5) {
          ierr = PetscPrintf(PETSC_COMM_WORLD, "[TEST FAILED] i=%D, j=%D, tval_mdot[j]=%g, tval_dot[j]=%g\n",i,j,(double)PetscAbsScalar(tval_mdot[j]), (double)PetscAbsScalar(tval_dot[j]));CHKERRQ(ierr);
          break;
        }
      }
    }
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test completed successfully!\n",k,n);CHKERRQ(ierr);
  ierr = PetscFree(val_dot);CHKERRQ(ierr);
  ierr = PetscFree(val_mdot);CHKERRQ(ierr);
  ierr = PetscFree(tval_dot);CHKERRQ(ierr);
  ierr = PetscFree(tval_mdot);CHKERRQ(ierr);
  ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

   test:
      suffix: cuda
      args: -vec_type cuda
      output_file: output/ex43_1.out
      requires: cuda

TEST*/
