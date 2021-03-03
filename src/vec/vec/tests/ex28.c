
static char help[] = "Tests repeated VecDotBegin()/VecDotEnd().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 25,i,row0 = 0;
  PetscScalar    two = 2.0,result1,result2,results[40],value,ten = 10.0;
  PetscScalar    result1a,result2a;
  PetscReal      result3,result4,result[2],result3a,result4a,resulta[2];
  Vec            x,y,vecs[40];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(x,NULL,"-x_view");CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);

  /*
        Test mixing dot products and norms that require sums
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr    = VecDotBegin(x,y,&result1);CHKERRQ(ierr);
  ierr    = VecDotBegin(y,x,&result2);CHKERRQ(ierr);
  ierr    = VecNormBegin(y,NORM_2,&result3);CHKERRQ(ierr);
  ierr    = VecNormBegin(x,NORM_1,&result4);CHKERRQ(ierr);
  ierr    = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)x));CHKERRQ(ierr);
  ierr    = VecDotEnd(x,y,&result1);CHKERRQ(ierr);
  ierr    = VecDotEnd(y,x,&result2);CHKERRQ(ierr);
  ierr    = VecNormEnd(y,NORM_2,&result3);CHKERRQ(ierr);
  ierr    = VecNormEnd(x,NORM_1,&result4);CHKERRQ(ierr);

  ierr = VecDot(x,y,&result1a);CHKERRQ(ierr);
  ierr = VecDot(y,x,&result2a);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&result3a);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&result4a);CHKERRQ(ierr);

  if (result1 != result1a || result2 != result2a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error dot: result1 %g result2 %g\n",(double)PetscRealPart(result1),(double)PetscRealPart(result2));CHKERRQ(ierr);
  }
  if (result3 != result3a || result4 != result4a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error 1,2 norms: result3 %g result4 %g\n",(double)result3,(double)result4);CHKERRQ(ierr);
  }

  /*
        Test norms that only require abs
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr = VecNormBegin(y,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormBegin(x,NORM_MAX,&result4);CHKERRQ(ierr);
  ierr = VecNormEnd(y,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result4);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_MAX,&result4a);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_MAX,&result3a);CHKERRQ(ierr);
  if (result3 != result3a || result4 != result4a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error max norm: result3 %g result4 %g\n",(double)result3,(double)result4);CHKERRQ(ierr);
  }

  /*
        Tests dot,  max, 1, norm
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr = VecSetValues(x,1,&row0,&ten,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecDotBegin(x,y,&result1);CHKERRQ(ierr);
  ierr = VecDotBegin(y,x,&result2);CHKERRQ(ierr);
  ierr = VecNormBegin(x,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormBegin(x,NORM_1,&result4);CHKERRQ(ierr);
  ierr = VecDotEnd(x,y,&result1);CHKERRQ(ierr);
  ierr = VecDotEnd(y,x,&result2);CHKERRQ(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormEnd(x,NORM_1,&result4);CHKERRQ(ierr);

  ierr = VecDot(x,y,&result1a);CHKERRQ(ierr);
  ierr = VecDot(y,x,&result2a);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_MAX,&result3a);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1,&result4a);CHKERRQ(ierr);

  if (result1 != result1a || result2 != result2a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error dot: result1 %g result2 %g\n",(double)PetscRealPart(result1),(double)PetscRealPart(result2));CHKERRQ(ierr);
  }
  if (result3 != result3a || result4 != result4a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error max 1 norms: result3 %g result4 %g\n",(double)result3,(double)result4);CHKERRQ(ierr);
  }

  /*
       tests 1_and_2 norm
  */
  ierr = VecNormBegin(x,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormBegin(x,NORM_1_AND_2,result);CHKERRQ(ierr);
  ierr = VecNormBegin(y,NORM_MAX,&result4);CHKERRQ(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result3);CHKERRQ(ierr);
  ierr = VecNormEnd(x,NORM_1_AND_2,result);CHKERRQ(ierr);
  ierr = VecNormEnd(y,NORM_MAX,&result4);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_MAX,&result3a);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_1_AND_2,resulta);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_MAX,&result4a);CHKERRQ(ierr);
  if (result3 != result3a || result4 != result4a) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error max: result1 %g result2 %g\n",(double)result3,(double)result4);CHKERRQ(ierr);
  }
  if (PetscAbsReal(result[0]-resulta[0]) > .01 || PetscAbsReal(result[1]-resulta[1]) > .01) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error 1 and 2 norms: result[0] %g result[1] %g\n",(double)result[0],(double)result[1]);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  /*
       Tests computing a large number of operations that require
    allocating a larger data structure internally
  */
  for (i=0; i<40; i++) {
    ierr  = VecCreate(PETSC_COMM_WORLD,vecs+i);CHKERRQ(ierr);
    ierr  = VecSetSizes(vecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr  = VecSetFromOptions(vecs[i]);CHKERRQ(ierr);
    value = (PetscReal)i;
    ierr  = VecSet(vecs[i],value);CHKERRQ(ierr);
  }
  for (i=0; i<39; i++) {
    ierr = VecDotBegin(vecs[i],vecs[i+1],results+i);CHKERRQ(ierr);
  }
  for (i=0; i<39; i++) {
    ierr = VecDotEnd(vecs[i],vecs[i+1],results+i);CHKERRQ(ierr);
    if (results[i] != 25.0*i*(i+1)) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"i %D expected %g got %g\n",i,25.0*i*(i+1),(double)PetscRealPart(results[i]));CHKERRQ(ierr);
    }
  }
  for (i=0; i<40; i++) {
    ierr = VecDestroy(&vecs[i]);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

   testset:
     nsize: 3
     output_file: output/ex28_1.out

     test:
        suffix: 2
        args: -splitreduction_async

     test:
        suffix: 2_cuda
        args: -splitreduction_async -vec_type cuda
        requires: cuda

     test:
        suffix: cuda
        args: -vec_type cuda
        requires: cuda

     test:
        suffix: 2_kokkos
        args: -splitreduction_async -vec_type kokkos
        requires: kokkos_kernels

     test:
        suffix: kokkos
        args: -vec_type kokkos
        requires: kokkos_kernels
TEST*/
