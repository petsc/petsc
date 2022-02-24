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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
  CHKERRQ(VecSetSizes(v,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(v));
  CHKERRQ(VecDuplicate(v,&w));
  CHKERRQ(VecSetRandom(v,rctx));
  CHKERRQ(VecSetRandom(w,rctx));

  /* create dummy vector to clear cache */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,10000000));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetRandom(x,rctx));

  /* send v to GPU */
  CHKERRQ(PetscBarrier(NULL));
  CHKERRQ(VecNorm(v,NORM_1,&norm1));

  /* register a stage work on GPU */
  CHKERRQ(PetscLogStageRegister("Work on GPU", &stage));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(VecNorm(w,NORM_1,&norm1)); /* send w to GPU */
  CHKERRQ(VecNorm(x,NORM_1,&norm1)); /* clear cache */
  CHKERRQ(PetscBarrier(NULL));
  CHKERRQ(VecAXPY(w,1.0,v));
  CHKERRQ(VecNorm(x,NORM_INFINITY,&norm1));
  CHKERRQ(PetscBarrier(NULL));
  CHKERRQ(VecDot(w,v,&val));
  CHKERRQ(VecNorm(x,NORM_1,&norm1));
  CHKERRQ(PetscBarrier(NULL));
  CHKERRQ(VecSet(v,0.0));
  CHKERRQ(VecNorm(x,NORM_2,&norm2));
  CHKERRQ(PetscBarrier(NULL));
  CHKERRQ(VecCopy(v,w));
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test completed successfully!\n"));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(PetscRandomDestroy(&rctx));
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
