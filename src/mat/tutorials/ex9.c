
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nmat",&nmat,NULL));

  /*
     Create random matrices
  */
  CHKERRQ(PetscMalloc1(nmat+3,&A));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n/2,3,NULL,3,NULL,&A[0]));
  for (i = 1; i < nmat+1; i++) {
    CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,3,NULL,3,NULL,&A[i]));
  }
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n/2,n,3,NULL,3,NULL,&A[nmat+1]));
  for (i = 0; i < nmat+2; i++) {
    CHKERRQ(MatSetRandom(A[i],rctx));
  }

  CHKERRQ(MatCreateVecs(A[1],&x,&y));
  CHKERRQ(VecDuplicate(y,&z));
  CHKERRQ(VecDuplicate(z,&z2));
  CHKERRQ(MatCreateVecs(A[0],&v,NULL));
  CHKERRQ(VecDuplicate(v,&v2));

  /* Test MatMult of an ADDITIVE MatComposite B made up of A[1],A[2],A[3] with separate scalings */

  /* Do MatMult with A[1],A[2],A[3] by hand and store the result in z */
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A[1],x,z));
  CHKERRQ(VecScale(z,scalings[1]));
  for (i = 2; i < nmat+1; i++) {
    CHKERRQ(MatMult(A[i],x,z2));
    CHKERRQ(VecAXPY(z,scalings[i],z2));
  }

  /* Do MatMult using MatComposite and store the result in y */
  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(MatCreateComposite(PETSC_COMM_WORLD,nmat,A+1,&B));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatCompositeSetScalings(B,&scalings[1]));
  CHKERRQ(MatMultAdd(B,x,y,y));

  /* Diff y and z */
  CHKERRQ(VecAXPY(y,-1.0,z));
  CHKERRQ(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with composite add %g\n",(double)rnorm));
  }

  /* Test MatCompositeMerge on ADDITIVE MatComposite */
  CHKERRQ(MatCompositeSetMatStructure(B,DIFFERENT_NONZERO_PATTERN)); /* default */
  CHKERRQ(MatCompositeMerge(B));
  CHKERRQ(MatMult(B,x,y));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecAXPY(y,-1.0,z));
  CHKERRQ(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with composite add after merge %g\n",(double)rnorm));
  }

  /*
     Test n x n/2 multiplicative composite B made up of A[0],A[1],A[2] with separate scalings
  */

  /* Do MatMult with A[0],A[1],A[2] by hand and store the result in z */
  CHKERRQ(VecSet(v,1.0));
  CHKERRQ(MatMult(A[0],v,z));
  CHKERRQ(VecScale(z,scalings[0]));
  for (i = 1; i < nmat; i++) {
    CHKERRQ(MatMult(A[i],z,y));
    CHKERRQ(VecScale(y,scalings[i]));
    CHKERRQ(VecCopy(y,z));
  }

  /* Do MatMult using MatComposite and store the result in y */
  CHKERRQ(MatCreateComposite(PETSC_COMM_WORLD,nmat,A,&B));
  CHKERRQ(MatCompositeSetType(B,MAT_COMPOSITE_MULTIPLICATIVE));
  CHKERRQ(MatCompositeSetMergeType(B,MAT_COMPOSITE_MERGE_LEFT));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatCompositeSetScalings(B,&scalings[0]));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY)); /* do MatCompositeMerge() if -mat_composite_merge 1 */
  CHKERRQ(MatMult(B,v,y));
  CHKERRQ(MatDestroy(&B));

  /* Diff y and z */
  CHKERRQ(VecAXPY(y,-1.0,z));
  CHKERRQ(VecNorm(y,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with composite multiplicative %g\n",(double)rnorm));
  }

  /*
     Test n/2 x n multiplicative composite B made up of A[2], A[3], A[4] without separate scalings
  */
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A[2],x,z));
  for (i = 3; i < nmat+1; i++) {
    CHKERRQ(MatMult(A[i],z,y));
    CHKERRQ(VecCopy(y,z));
  }
  CHKERRQ(MatMult(A[nmat+1],z,v));

  CHKERRQ(MatCreateComposite(PETSC_COMM_WORLD,nmat,A+2,&B));
  CHKERRQ(MatCompositeSetType(B,MAT_COMPOSITE_MULTIPLICATIVE));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY)); /* do MatCompositeMerge() if -mat_composite_merge 1 */
  CHKERRQ(MatMult(B,x,v2));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(VecAXPY(v2,-1.0,v));
  CHKERRQ(VecNorm(v2,NORM_2,&rnorm));
  if (rnorm > 10000.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with composite multiplicative %g\n",(double)rnorm));
  }

  /*
     Test get functions
  */
  CHKERRQ(MatCreateComposite(PETSC_COMM_WORLD,nmat,A,&B));
  CHKERRQ(MatCompositeGetNumberMat(B,&n));
  if (nmat != n) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with GetNumberMat %" PetscInt_FMT " != %" PetscInt_FMT "\n",nmat,n));
  }
  CHKERRQ(MatCompositeGetMat(B,0,&A[nmat+2]));
  if (A[0] != A[nmat+2]) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with GetMat\n"));
  }
  CHKERRQ(MatCompositeGetType(B,&type));
  if (type != MAT_COMPOSITE_ADDITIVE) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error with GetType\n"));
  }
  CHKERRQ(MatDestroy(&B));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&v2));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(VecDestroy(&z2));
  CHKERRQ(PetscRandomDestroy(&rctx));
  for (i = 0; i < nmat+2; i++) {
    CHKERRQ(MatDestroy(&A[i]));
  }
  CHKERRQ(PetscFree(A));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: double
      args: -mat_composite_merge {{0 1}shared output} -mat_composite_merge_mvctx {{0 1}shared output}

TEST*/
