
static char help[] = "Tests MatCreateLRC()\n\n";

/*T
   Concepts: Low rank correction

   Processors: n
T*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,b,c=NULL;
  Mat            A,U,V,LR,X,LRe;
  PetscInt       M = 5, N = 7;
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the sparse matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetOptionsPrefix(A,"A_"));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetRandom(A,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the dense matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(MatSetSizes(U,PETSC_DECIDE,PETSC_DECIDE,M,3));
  CHKERRQ(MatSetType(U,MATDENSE));
  CHKERRQ(MatSetOptionsPrefix(U,"U_"));
  CHKERRQ(MatSetFromOptions(U));
  CHKERRQ(MatSetUp(U));
  CHKERRQ(MatSetRandom(U,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&V));
  CHKERRQ(MatSetSizes(V,PETSC_DECIDE,PETSC_DECIDE,N,3));
  CHKERRQ(MatSetType(V,MATDENSE));
  CHKERRQ(MatSetOptionsPrefix(V,"V_"));
  CHKERRQ(MatSetFromOptions(V));
  CHKERRQ(MatSetUp(V));
  CHKERRQ(MatSetRandom(V,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create a vector to hold the diagonal of C
         A sequential vector can be created as well on each process
         It is user responsibility to ensure the data in the vector
         is consistent across processors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-use_c",&flg));
  if (flg) {
    CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,3,&c));
    CHKERRQ(VecSetRandom(c,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create low rank correction matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-low_rank",&flg));
  if (flg) {
    /* create a low-rank matrix, with no A-matrix */
    CHKERRQ(MatCreateLRC(NULL,U,c,V,&LR));
    CHKERRQ(MatDestroy(&A));
  } else {
    CHKERRQ(MatCreateLRC(A,U,c,V,&LR));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the low rank correction matrix explicitly to check for
         correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatHermitianTranspose(V,MAT_INITIAL_MATRIX,&X));
  CHKERRQ(MatDiagonalScale(X,c,NULL));
  CHKERRQ(MatMatMult(U,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&LRe));
  CHKERRQ(MatDestroy(&X));
  if (A) {
    CHKERRQ(MatAYPX(LRe,1.0,A,DIFFERENT_NONZERO_PATTERN));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create test vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateVecs(LR,&x,&b));
  CHKERRQ(VecSetRandom(x,NULL));
  CHKERRQ(MatMult(LR,x,b));
  CHKERRQ(MatMultTranspose(LR,b,x));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Check correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatMultEqual(LR,LRe,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error in MatMult\n"));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatMultHermitianTransposeEqual(LR,LRe,10,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error in MatMultTranspose\n"));
#endif

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&LRe));
  CHKERRQ(MatDestroy(&U));
  CHKERRQ(MatDestroy(&V));
  CHKERRQ(VecDestroy(&c));
  CHKERRQ(MatDestroy(&LR));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/ex102_1.out
      nsize: {{1 2}}
      args: -low_rank {{0 1}} -use_c {{0 1}}
      test:
        suffix: standard
      test:
        suffix: cuda
        requires: cuda
        args: -A_mat_type aijcusparse -U_mat_type densecuda -V_mat_type densecuda

TEST*/
