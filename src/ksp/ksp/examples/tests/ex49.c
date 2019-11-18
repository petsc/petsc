
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
  PetscErrorCode ierr;
  PetscBool      test_hermitian = PETSC_FALSE, convert = PETSC_FALSE;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-herm",&test_hermitian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-conv",&convert,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n*bs,n*bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,bs,n,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,bs,n,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,n*bs,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,n*bs,NULL,n*bs,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rctx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      ierr = PetscRandomGetValue(rctx,&v);CHKERRQ(ierr);
      if (PetscRealPart(v) < .1 || i == j) {
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            Ii = i*bs + k;
            J = j*bs + l;
            ierr = PetscRandomGetValue(rctx,&v);CHKERRQ(ierr);
            if (Ii == J) v = PetscRealPart(v+3*n*bs);
            ierr = MatSetValue(A,Ii,J,v,INSERT_VALUES);CHKERRQ(ierr);
            if (test_hermitian) {
              ierr = MatSetValue(A,J,Ii,PetscConj(v),INSERT_VALUES);CHKERRQ(ierr);
            } else {
              ierr = MatSetValue(A,J,Ii,v,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* With complex numbers:
     - PETSc cholesky does not support hermitian matrices
     - CHOLMOD only supports hermitian matrices
     - SUPERLU_DIST seems supporting both
  */
  if (test_hermitian) {
    ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  }

  {
    Mat M;
    ierr = MatComputeOperator(A,MATAIJ,&M);CHKERRQ(ierr);
    ierr = MatViewFromOptions(M,NULL,"-expl_view");CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
  }

  A2 = NULL;
  if (convert) {
    ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  }

  ierr = VecCreate(PETSC_COMM_SELF,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,n*bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);

  /*
     Set operators.
  */
  ierr = KSPSetOperators(ksp,A2 ? A2 : A,A);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm > 100*PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of residual %g iterations %D bs %D\n",(double)norm,its,bs);CHKERRQ(ierr);
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

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

TEST*/
