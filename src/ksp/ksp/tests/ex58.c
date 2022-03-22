
static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

/*T
   Concepts: KSP^solving a system of linear equations
   Processors: 1
T*/

/*
  Modified from ex1.c for testing matrix operations when matrix structure is changed.
  Contributed by Jose E. Roman, Feb. 2012.
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            A,B,C;        /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm;         /* norm of solution error */
  PetscInt       i,n = 20,col[3],its;
  PetscMPIInt    size;
  PetscScalar    one = 1.0,value[3];
  PetscBool      nonzeroguess = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-nonzero_guess",&nonzeroguess,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(PetscObjectSetName((PetscObject) x, "Solution"));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     jroman: added matrices
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  for (i=0; i<n; i++) {
    PetscCall(MatSetValue(B,i,i,value[1],INSERT_VALUES));
    if (n-i+n/3<n) {
      PetscCall(MatSetValue(B,n-i+n/3,i,value[0],INSERT_VALUES));
      PetscCall(MatSetValue(B,i,n-i+n/3,value[0],INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&C));
  PetscCall(MatAXPY(C,2.0,B,DIFFERENT_NONZERO_PATTERN));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  PetscCall(VecSet(u,one));
  PetscCall(MatMult(C,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp,C,C));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCJACOBI));
  PetscCall(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  if (nonzeroguess) {
    PetscScalar p = .5;
    PetscCall(VecSet(x,p));
    PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Solve linear system
  */
  PetscCall(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(KSPDestroy(&ksp));

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

   test:
      args: -mat_type aij
      output_file: output/ex58.out

   test:
      suffix: baij
      args: -mat_type baij
      output_file: output/ex58.out

   test:
      suffix: sbaij
      args: -mat_type sbaij
      output_file: output/ex58.out

TEST*/
