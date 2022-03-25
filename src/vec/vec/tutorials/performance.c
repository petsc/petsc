static char help[] = "Time vector operations on GPU\n";
/* This program produces the results for Argonne Technical Report ANL-19/41.
   The technical report and resources for generating data can be found in the
   repository:  https://gitlab.com/hannah_mairs/summit-performance */

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec            v,w,x;
  PetscInt       n=15;
  PetscScalar    val;
  PetscReal      norm1,norm2;
  PetscRandom    rctx;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecSetSizes(v,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(v));
  PetscCall(VecDuplicate(v,&w));
  PetscCall(VecSetRandom(v,rctx));
  PetscCall(VecSetRandom(w,rctx));

  /* create dummy vector to clear cache */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,10000000));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetRandom(x,rctx));

  /* send v to GPU */
  PetscCall(PetscBarrier(NULL));
  PetscCall(VecNorm(v,NORM_1,&norm1));

  /* register a stage work on GPU */
  PetscCall(PetscLogStageRegister("Work on GPU", &stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(VecNorm(w,NORM_1,&norm1)); /* send w to GPU */
  PetscCall(VecNorm(x,NORM_1,&norm1)); /* clear cache */
  PetscCall(PetscBarrier(NULL));
  PetscCall(VecAXPY(w,1.0,v));
  PetscCall(VecNorm(x,NORM_INFINITY,&norm1));
  PetscCall(PetscBarrier(NULL));
  PetscCall(VecDot(w,v,&val));
  PetscCall(VecNorm(x,NORM_1,&norm1));
  PetscCall(PetscBarrier(NULL));
  PetscCall(VecSet(v,0.0));
  PetscCall(VecNorm(x,NORM_2,&norm2));
  PetscCall(PetscBarrier(NULL));
  PetscCall(VecCopy(v,w));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test completed successfully!\n"));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(PetscFinalize());
  return 0;
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
