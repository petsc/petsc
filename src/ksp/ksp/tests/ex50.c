
static char help[] = "Tests SeqBAIJ point block Jacobi for different block sizes\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,k,l,n = 27,its,bs = 2,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n*bs,n*bs,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSize(A,bs));
  CHKERRQ(MatSetType(A,MATSEQBAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /*
     Don't bother to preallocate matrix
  */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rctx));
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      CHKERRQ(PetscRandomGetValue(rctx,&v));
      if (PetscRealPart(v) < .25 || i == j) {
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            CHKERRQ(PetscRandomGetValue(rctx,&v));
            Ii = i*bs + k;
            J = j*bs + l;
            if (Ii == J) v += 10.;
            CHKERRQ(MatSetValue(A,Ii,J,v,INSERT_VALUES));
          }
        }
      }
    }
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
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
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  CHKERRQ(KSPSetOperators(ksp,A,A));

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
  if (norm > .1) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of residual %g iterations %D bs %D\n",(double)norm,its,bs));
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
  CHKERRQ(PetscRandomDestroy(&rctx));

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
      args: -bs {{1 2 3 4 5 6 7}} -pc_type pbjacobi

   test:
      suffix: 2
      args: -bs {{8 9 10 11 12 13 14 15}} -pc_type ilu

TEST*/
