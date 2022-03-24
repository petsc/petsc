
static char help[] = "Tests SeqSBAIJ factorizations for different block sizes\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;
  Mat            A,A2;
  KSP            ksp;
  PetscRandom    rctx;
  PetscReal      norm;
  PetscInt       i,j,k,l,n = 27,its,bs = 2,Ii,J;
  PetscBool      test_hermitian = PETSC_FALSE, convert = PETSC_FALSE;
  PetscScalar    v;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-herm",&test_hermitian,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-conv",&convert,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,n*bs,n*bs,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSize(A,bs));
  CHKERRQ(MatSetType(A,MATSEQSBAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqSBAIJSetPreallocation(A,bs,n,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,bs,n,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,n*bs,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,n*bs,NULL,n*bs,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rctx));
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      CHKERRQ(PetscRandomGetValue(rctx,&v));
      if (PetscRealPart(v) < .1 || i == j) {
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            Ii = i*bs + k;
            J = j*bs + l;
            CHKERRQ(PetscRandomGetValue(rctx,&v));
            if (Ii == J) v = PetscRealPart(v+3*n*bs);
            CHKERRQ(MatSetValue(A,Ii,J,v,INSERT_VALUES));
            if (test_hermitian) {
              CHKERRQ(MatSetValue(A,J,Ii,PetscConj(v),INSERT_VALUES));
            } else {
              CHKERRQ(MatSetValue(A,J,Ii,v,INSERT_VALUES));
            }
          }
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* With complex numbers:
     - PETSc cholesky does not support hermitian matrices
     - CHOLMOD only supports hermitian matrices
     - SUPERLU_DIST seems supporting both
  */
  if (test_hermitian) {
    CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  }

  {
    Mat M;
    CHKERRQ(MatComputeOperator(A,MATAIJ,&M));
    CHKERRQ(MatViewFromOptions(M,NULL,"-expl_view"));
    CHKERRQ(MatDestroy(&M));
  }

  A2 = NULL;
  if (convert) {
    CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&A2));
  }

  CHKERRQ(VecCreate(PETSC_COMM_SELF,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,n*bs));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(b,&x));

  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));

  /*
     Set operators.
  */
  CHKERRQ(KSPSetOperators(ksp,A2 ? A2 : A,A));

  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm > 100*PETSC_SMALL) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Norm of residual %g iterations %D bs %D\n",(double)norm,its,bs));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(PetscRandomDestroy(&rctx));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type {{aij baij sbaij}} -bs {{1 2 3 4 5 6 7 8 9 10 11 12}} -pc_type cholesky -herm 0 -conv {{0 1}}

   test:
      nsize: {{1 4}}
      suffix: cholmod
      requires: suitesparse
      args: -mat_type {{aij sbaij}} -bs 1 -pc_type cholesky -pc_factor_mat_solver_type cholmod -herm -conv {{0 1}}

   test:
      nsize: {{1 4}}
      suffix: superlu_dist
      requires: superlu_dist
      output_file: output/ex49_cholmod.out
      args: -mat_type mpiaij -bs 3 -pc_type cholesky -pc_factor_mat_solver_type superlu_dist -herm {{0 1}} -conv {{0 1}}

   test:
      suffix: mkl_pardiso
      requires: mkl_pardiso
      output_file: output/ex49_1.out
      args: -bs {{1 3}} -pc_type cholesky -pc_factor_mat_solver_type mkl_pardiso

   test:
      suffix: cg
      requires: complex
      output_file: output/ex49_cg.out
      args: -herm -ksp_cg_type hermitian -mat_type aij -ksp_type cg -pc_type jacobi -ksp_rtol 4e-07

   test:
      suffix: pipecg2
      requires: complex
      output_file: output/ex49_pipecg2.out
      args: -herm -mat_type aij -ksp_type pipecg2 -pc_type jacobi -ksp_rtol 4e-07 -ksp_norm_type {{preconditioned unpreconditioned natural}}

TEST*/
