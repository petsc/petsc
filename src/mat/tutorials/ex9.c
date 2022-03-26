
static char help[] = "Tests MatCreateComposite()\n\n";

/*T
   Concepts: Mat^composite matrices
   Processors: n
T*/

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat              *A,B;           /* matrix */
  Vec              x,y,v,v2,z,z2;
  PetscReal        rnorm;
  PetscInt         n = 20;         /* size of the matrix */
  PetscInt         nmat = 3;       /* number of matrices */
  PetscInt         i;
  PetscRandom      rctx;
  MatCompositeType type;
  PetscScalar      scalings[5]={2,3,4,5,6};

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nmat",&nmat,NULL));

  /*
     Create random matrices
  */
  PetscCall(PetscMalloc1(nmat+3,&A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n/2,3,NULL,3,NULL,&A[0]));
  for (i = 1; i < nmat+1; i++) {
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,3,NULL,3,NULL,&A[i]));
  }
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n/2,n,3,NULL,3,NULL,&A[nmat+1]));
  for (i = 0; i < nmat+2; i++) {
    PetscCall(MatSetRandom(A[i],rctx));
  }

  PetscCall(MatCreateVecs(A[1],&x,&y));
  PetscCall(VecDuplicate(y,&z));
  PetscCall(VecDuplicate(z,&z2));
  PetscCall(MatCreateVecs(A[0],&v,NULL));
  PetscCall(VecDuplicate(v,&v2));

  /* Test MatMult of an ADDITIVE MatComposite B made up of A[1],A[2],A[3] with separate scalings */

  /* Do MatMult with A[1],A[2],A[3] by hand and store the result in z */
  PetscCall(VecSet(x,1.0));
  PetscCall(MatMult(A[1],x,z));
  PetscCall(VecScale(z,scalings[1]));
  for (i = 2; i < nmat+1; i++) {
    PetscCall(MatMult(A[i],x,z2));
    PetscCall(VecAXPY(z,scalings[i],z2));
  }

  /* Do MatMult using MatComposite and store the result in y */
  PetscCall(VecSet(y,0.0));
  PetscCall(MatCreateComposite(PETSC_COMM_WORLD,nmat,A+1,&B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatCompositeSetScalings(B,&scalings[1]));
  PetscCall(MatMultAdd(B,x,y,y));

  /* Diff y and z */
  PetscCall(VecAXPY(y,-1.0,z));
  PetscCall(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with composite add %g\n",(double)rnorm));
  }

  /* Test MatCompositeMerge on ADDITIVE MatComposite */
  PetscCall(MatCompositeSetMatStructure(B,DIFFERENT_NONZERO_PATTERN)); /* default */
  PetscCall(MatCompositeMerge(B));
  PetscCall(MatMult(B,x,y));
  PetscCall(MatDestroy(&B));
  PetscCall(VecAXPY(y,-1.0,z));
  PetscCall(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with composite add after merge %g\n",(double)rnorm));
  }

  /*
     Test n x n/2 multiplicative composite B made up of A[0],A[1],A[2] with separate scalings
  */

  /* Do MatMult with A[0],A[1],A[2] by hand and store the result in z */
  PetscCall(VecSet(v,1.0));
  PetscCall(MatMult(A[0],v,z));
  PetscCall(VecScale(z,scalings[0]));
  for (i = 1; i < nmat; i++) {
    PetscCall(MatMult(A[i],z,y));
    PetscCall(VecScale(y,scalings[i]));
    PetscCall(VecCopy(y,z));
  }

  /* Do MatMult using MatComposite and store the result in y */
  PetscCall(MatCreateComposite(PETSC_COMM_WORLD,nmat,A,&B));
  PetscCall(MatCompositeSetType(B,MAT_COMPOSITE_MULTIPLICATIVE));
  PetscCall(MatCompositeSetMergeType(B,MAT_COMPOSITE_MERGE_LEFT));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatCompositeSetScalings(B,&scalings[0]));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY)); /* do MatCompositeMerge() if -mat_composite_merge 1 */
  PetscCall(MatMult(B,v,y));
  PetscCall(MatDestroy(&B));

  /* Diff y and z */
  PetscCall(VecAXPY(y,-1.0,z));
  PetscCall(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with composite multiplicative %g\n",(double)rnorm));
  }

  /*
     Test n/2 x n multiplicative composite B made up of A[2], A[3], A[4] without separate scalings
  */
  PetscCall(VecSet(x,1.0));
  PetscCall(MatMult(A[2],x,z));
  for (i = 3; i < nmat+1; i++) {
    PetscCall(MatMult(A[i],z,y));
    PetscCall(VecCopy(y,z));
  }
  PetscCall(MatMult(A[nmat+1],z,v));

  PetscCall(MatCreateComposite(PETSC_COMM_WORLD,nmat,A+2,&B));
  PetscCall(MatCompositeSetType(B,MAT_COMPOSITE_MULTIPLICATIVE));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY)); /* do MatCompositeMerge() if -mat_composite_merge 1 */
  PetscCall(MatMult(B,x,v2));
  PetscCall(MatDestroy(&B));

  PetscCall(VecAXPY(v2,-1.0,v));
  PetscCall(VecNorm(v2,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with composite multiplicative %g\n",(double)rnorm));
  }

  /*
     Test get functions
  */
  PetscCall(MatCreateComposite(PETSC_COMM_WORLD,nmat,A,&B));
  PetscCall(MatCompositeGetNumberMat(B,&n));
  if (nmat != n) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with GetNumberMat %" PetscInt_FMT " != %" PetscInt_FMT "\n",nmat,n));
  }
  PetscCall(MatCompositeGetMat(B,0,&A[nmat+2]));
  if (A[0] != A[nmat+2]) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with GetMat\n"));
  }
  PetscCall(MatCompositeGetType(B,&type));
  if (type != MAT_COMPOSITE_ADDITIVE) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error with GetType\n"));
  }
  PetscCall(MatDestroy(&B));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&v2));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&z2));
  PetscCall(PetscRandomDestroy(&rctx));
  for (i = 0; i < nmat+2; i++) {
    PetscCall(MatDestroy(&A[i]));
  }
  PetscCall(PetscFree(A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: double
      args: -mat_composite_merge {{0 1}shared output} -mat_composite_merge_mvctx {{0 1}shared output}

TEST*/
