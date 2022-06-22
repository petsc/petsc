
static char help[] = "Tests MatCreateLRC()\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,b,c=NULL;
  Mat            A,U,V,LR,X,LRe;
  PetscInt       M = 5, N = 7;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the sparse matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetOptionsPrefix(A,"A_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetRandom(A,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the dense matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&U));
  PetscCall(MatSetSizes(U,PETSC_DECIDE,PETSC_DECIDE,M,3));
  PetscCall(MatSetType(U,MATDENSE));
  PetscCall(MatSetOptionsPrefix(U,"U_"));
  PetscCall(MatSetFromOptions(U));
  PetscCall(MatSetUp(U));
  PetscCall(MatSetRandom(U,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&V));
  PetscCall(MatSetSizes(V,PETSC_DECIDE,PETSC_DECIDE,N,3));
  PetscCall(MatSetType(V,MATDENSE));
  PetscCall(MatSetOptionsPrefix(V,"V_"));
  PetscCall(MatSetFromOptions(V));
  PetscCall(MatSetUp(V));
  PetscCall(MatSetRandom(V,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create a vector to hold the diagonal of C
         A sequential vector can be created as well on each process
         It is user responsibility to ensure the data in the vector
         is consistent across processors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-use_c",&flg));
  if (flg) {
    PetscCall(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,3,&c));
    PetscCall(VecSetRandom(c,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create low rank correction matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-low_rank",&flg));
  if (flg) {
    /* create a low-rank matrix, with no A-matrix */
    PetscCall(MatCreateLRC(NULL,U,c,V,&LR));
    PetscCall(MatDestroy(&A));
  } else {
    PetscCall(MatCreateLRC(A,U,c,V,&LR));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the low rank correction matrix explicitly to check for
         correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatHermitianTranspose(V,MAT_INITIAL_MATRIX,&X));
  PetscCall(MatDiagonalScale(X,c,NULL));
  PetscCall(MatMatMult(U,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&LRe));
  PetscCall(MatDestroy(&X));
  if (A) {
    PetscCall(MatAYPX(LRe,1.0,A,DIFFERENT_NONZERO_PATTERN));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create test vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreateVecs(LR,&x,&b));
  PetscCall(VecSetRandom(x,NULL));
  PetscCall(MatMult(LR,x,b));
  PetscCall(MatMultTranspose(LR,b,x));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Check correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatMultEqual(LR,LRe,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error in MatMult\n"));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatMultHermitianTransposeEqual(LR,LRe,10,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error in MatMultTranspose\n"));
#endif

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&LRe));
  PetscCall(MatDestroy(&U));
  PetscCall(MatDestroy(&V));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&LR));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
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
