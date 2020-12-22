static char help[] = "Time vector operations on GPU\n";
/* This program produces the results for Argonne Technical Report ANL-19/41.
   The technical report and resources for generating data can be found in the
   repository:  https://gitlab.com/hannah_mairs/summit-performance */

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            v,w,x;
  PetscInt       n=15;
  PetscScalar    val;
  PetscReal      norm1,norm2;
  PetscRandom    rctx;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  ierr = VecSetRandom(v,rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(w,rctx);CHKERRQ(ierr);

  /* create dummy vector to clear cache */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,10000000);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);

  /* send v to GPU */
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = VecNorm(v,NORM_1,&norm1);

  /* register a stage work on GPU */
  ierr = PetscLogStageRegister("Work on GPU", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_1,&norm1);CHKERRQ(ierr); /* send w to GPU */
  ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr); /* clear cache */
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = VecAXPY(w,1.0,v);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm1);CHKERRQ(ierr);
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = VecDot(w,v,&val);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm2);CHKERRQ(ierr);
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = VecCopy(v,w);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test completed successfully!\n");CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      output_file: output/performance_cuda.out

      test:
        suffix: cuda
        args: -vec_type mpicuda
        requires: cuda

      test:
        suffix: hip
        args: -vec_type mpihip
        requires: hip

TEST*/
