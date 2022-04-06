
/* Usage:  mpiexec ex16 [-help] [all PETSc options] */

static char help[] = "Solves a sequence of linear systems with different right-hand-side vectors.\n\
Input parameters include:\n\
  -ntimes <ntimes>  : number of linear systems to solve\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;  /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       ntimes,i,j,k,Ii,J,Istart,Iend;
  PetscInt       m   = 8,n = 7,its;
  PetscBool      flg = PETSC_FALSE;
  PetscScalar    v,one = 1.0,rhs;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix for use in solving a series of
         linear systems of the form, A x_i = b_i, for i=1,2,...
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
   */
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Create parallel vectors.
      - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
        we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,m*n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(b,&x));

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
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve several linear systems of the form  A x_i = b_i
       I.e., we retain the same matrix (A) for all systems, but
       change the right-hand-side vector (b_i) at each step.

       In this case, we simply call KSPSolve() multiple times.  The
       preconditioner setup operations (e.g., factorization for ILU)
       be done during the first call to KSPSolve() only; such operations
       will NOT be repeated for successive solves.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ntimes = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ntimes",&ntimes,NULL));
  for (k=1; k<ntimes+1; k++) {

    /*
       Set exact solution; then compute right-hand-side vector.  We use
       an exact solution of a vector with all elements equal to 1.0*k.
    */
    rhs  = one * (PetscReal)k;
    PetscCall(VecSet(u,rhs));
    PetscCall(MatMult(A,u,b));

    /*
       View the exact solution vector if desired
    */
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL));
    if (flg) PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(KSPSolve(ksp,b,x));

    /*
       Check the error
    */
    PetscCall(VecAXPY(x,-1.0,u));
    PetscCall(VecNorm(x,NORM_2,&norm));
    PetscCall(KSPGetIterationNumber(ksp,&its));
    /*
       Print convergence information.  PetscPrintf() produces a single
       print statement from all processes that share a communicator.
    */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g System %D: iterations %D\n",(double)norm,k,its));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));  PetscCall(MatDestroy(&A));

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
      nsize: 2
      args: -ntimes 4 -ksp_gmres_cgs_refinement_type refine_always

TEST*/
