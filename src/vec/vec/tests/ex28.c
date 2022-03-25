
static char help[] = "Tests repeated VecDotBegin()/VecDotEnd().\n\n";

#include <petscvec.h>
#define CheckError(a,b,tol) do {\
    if (!PetscIsCloseAtTol(a,b,0,tol)) {\
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Real error at line %d, tol %g: %s %g %s %g diff %g\n",__LINE__,(double)tol,#a,(double)(a),#b,(double)(b),(double)((a)-(b)))); \
    }\
  } while (0)

#define CheckErrorScalar(a,b,tol) do {\
    if (!PetscIsCloseAtTol(PetscRealPart(a),PetscRealPart(b),0,tol)) {\
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Real error at line %d, tol %g: %s %g %s %g diff %g\n",__LINE__,(double)tol,#a,(double)PetscRealPart(a),#b,(double)PetscRealPart(b),(double)PetscRealPart((a)-(b)))); \
    }\
    if (!PetscIsCloseAtTol(PetscImaginaryPart(a),PetscImaginaryPart(b),0,PETSC_SMALL)) {\
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Imag error at line %d, tol %g: %s %g %s %g diff %g\n",__LINE__,(double)tol,#a,(double)PetscImaginaryPart(a),#b,(double)PetscImaginaryPart(b),(double)PetscImaginaryPart((a)-(b)))); \
    }\
  } while (0)

int main(int argc,char **argv)
{
  PetscInt       n = 25,i,row0 = 0;
  PetscScalar    two = 2.0,result1,result2,results[40],value,ten = 10.0;
  PetscScalar    result1a,result2a;
  PetscReal      result3,result4,result[2],result3a,result4a,resulta[2];
  Vec            x,y,vecs[40];
  PetscReal      tol = PETSC_SMALL;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecSetRandom(x,NULL));
  PetscCall(VecViewFromOptions(x,NULL,"-x_view"));
  PetscCall(VecSet(y,two));

  /*
        Test mixing dot products and norms that require sums
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  PetscCall(VecDotBegin(x,y,&result1));
  PetscCall(VecDotBegin(y,x,&result2));
  PetscCall(VecNormBegin(y,NORM_2,&result3));
  PetscCall(VecNormBegin(x,NORM_1,&result4));
  PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)x)));
  PetscCall(VecDotEnd(x,y,&result1));
  PetscCall(VecDotEnd(y,x,&result2));
  PetscCall(VecNormEnd(y,NORM_2,&result3));
  PetscCall(VecNormEnd(x,NORM_1,&result4));

  PetscCall(VecDot(x,y,&result1a));
  PetscCall(VecDot(y,x,&result2a));
  PetscCall(VecNorm(y,NORM_2,&result3a));
  PetscCall(VecNorm(x,NORM_1,&result4a));

  CheckErrorScalar(result1,result1a,tol);
  CheckErrorScalar(result2,result2a,tol);
  CheckError(result3,result3a,tol);
  CheckError(result4,result4a,tol);

  /*
        Test norms that only require abs
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  PetscCall(VecNormBegin(y,NORM_MAX,&result3));
  PetscCall(VecNormBegin(x,NORM_MAX,&result4));
  PetscCall(VecNormEnd(y,NORM_MAX,&result3));
  PetscCall(VecNormEnd(x,NORM_MAX,&result4));

  PetscCall(VecNorm(x,NORM_MAX,&result4a));
  PetscCall(VecNorm(y,NORM_MAX,&result3a));
  CheckError(result3,result3a,tol);
  CheckError(result4,result4a,tol);

  /*
        Tests dot,  max, 1, norm
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  PetscCall(VecSetValues(x,1,&row0,&ten,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecDotBegin(x,y,&result1));
  PetscCall(VecDotBegin(y,x,&result2));
  PetscCall(VecNormBegin(x,NORM_MAX,&result3));
  PetscCall(VecNormBegin(x,NORM_1,&result4));
  PetscCall(VecDotEnd(x,y,&result1));
  PetscCall(VecDotEnd(y,x,&result2));
  PetscCall(VecNormEnd(x,NORM_MAX,&result3));
  PetscCall(VecNormEnd(x,NORM_1,&result4));

  PetscCall(VecDot(x,y,&result1a));
  PetscCall(VecDot(y,x,&result2a));
  PetscCall(VecNorm(x,NORM_MAX,&result3a));
  PetscCall(VecNorm(x,NORM_1,&result4a));

  CheckErrorScalar(result1,result1a,tol);
  CheckErrorScalar(result2,result2a,tol);
  CheckError(result3,result3a,tol);
  CheckError(result4,result4a,tol);

  /*
       tests 1_and_2 norm
  */
  PetscCall(VecNormBegin(x,NORM_MAX,&result3));
  PetscCall(VecNormBegin(x,NORM_1_AND_2,result));
  PetscCall(VecNormBegin(y,NORM_MAX,&result4));
  PetscCall(VecNormEnd(x,NORM_MAX,&result3));
  PetscCall(VecNormEnd(x,NORM_1_AND_2,result));
  PetscCall(VecNormEnd(y,NORM_MAX,&result4));

  PetscCall(VecNorm(x,NORM_MAX,&result3a));
  PetscCall(VecNorm(x,NORM_1_AND_2,resulta));
  PetscCall(VecNorm(y,NORM_MAX,&result4a));

  CheckError(result3,result3a,tol);
  CheckError(result4,result4a,tol);
  CheckError(result[0],resulta[0],tol);
  CheckError(result[1],resulta[1],tol);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  /*
       Tests computing a large number of operations that require
    allocating a larger data structure internally
  */
  for (i=0; i<40; i++) {
    PetscCall(VecCreate(PETSC_COMM_WORLD,vecs+i));
    PetscCall(VecSetSizes(vecs[i],PETSC_DECIDE,n));
    PetscCall(VecSetFromOptions(vecs[i]));
    value = (PetscReal)i;
    PetscCall(VecSet(vecs[i],value));
  }
  for (i=0; i<39; i++) {
    PetscCall(VecDotBegin(vecs[i],vecs[i+1],results+i));
  }
  for (i=0; i<39; i++) {
    PetscScalar expected = 25.0*i*(i+1);
    PetscCall(VecDotEnd(vecs[i],vecs[i+1],results+i));
    CheckErrorScalar(results[i],expected,tol);
  }
  for (i=0; i<40; i++) {
    PetscCall(VecDestroy(&vecs[i]));
  }

  PetscCall(PetscFinalize());
  return 0;
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
        suffix: 2_hip
        args: -splitreduction_async -vec_type hip
        requires: hip

     test:
        suffix: hip
        args: -vec_type hip
        requires: hip

     test:
        suffix: 2_kokkos
        args: -splitreduction_async -vec_type kokkos
        requires: kokkos_kernels

     test:
        suffix: kokkos
        args: -vec_type kokkos
        requires: kokkos_kernels
TEST*/
