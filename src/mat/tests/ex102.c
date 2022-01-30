
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the sparse matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the dense matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = MatSetSizes(U,PETSC_DECIDE,PETSC_DECIDE,M,3);CHKERRQ(ierr);
  ierr = MatSetType(U,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(U,"U_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(U);CHKERRQ(ierr);
  ierr = MatSetUp(U);CHKERRQ(ierr);
  ierr = MatSetRandom(U,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&V);CHKERRQ(ierr);
  ierr = MatSetSizes(V,PETSC_DECIDE,PETSC_DECIDE,N,3);CHKERRQ(ierr);
  ierr = MatSetType(V,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(V,"V_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(V);CHKERRQ(ierr);
  ierr = MatSetUp(V);CHKERRQ(ierr);
  ierr = MatSetRandom(V,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create a vector to hold the diagonal of C
         A sequential vector can be created as well on each process
         It is user responsibility to ensure the data in the vector
         is consistent across processors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsHasName(NULL,NULL,"-use_c",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,3,&c);CHKERRQ(ierr);
    ierr = VecSetRandom(c,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create low rank correction matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsHasName(NULL,NULL,"-low_rank",&flg);CHKERRQ(ierr);
  if (flg) {
    /* create a low-rank matrix, with no A-matrix */
    ierr = MatCreateLRC(NULL,U,c,V,&LR);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  } else {
    ierr = MatCreateLRC(A,U,c,V,&LR);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the low rank correction matrix explicitly to check for
         correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatHermitianTranspose(V,MAT_INITIAL_MATRIX,&X);CHKERRQ(ierr);
  ierr = MatDiagonalScale(X,c,NULL);CHKERRQ(ierr);
  ierr = MatMatMult(U,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&LRe);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  if (A) {
    ierr = MatAYPX(LRe,1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create test vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(LR,&x,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  ierr = MatMult(LR,x,b);CHKERRQ(ierr);
  ierr = MatMultTranspose(LR,b,x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Check correctness
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatMultEqual(LR,LRe,10,&flg);CHKERRQ(ierr);
  if (!flg) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error in MatMult\n");CHKERRQ(ierr); }
#if !defined(PETSC_USE_COMPLEX)
  ierr = MatMultHermitianTransposeEqual(LR,LRe,10,&flg);CHKERRQ(ierr);
  if (!flg) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error in MatMultTranspose\n");CHKERRQ(ierr); }
#endif

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&LRe);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&V);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = MatDestroy(&LR);CHKERRQ(ierr);

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
