
/* Usage:  mpiexec ex16 [-help] [all PETSc options] */

static char help[] = "Solves a sequence of linear systems with different right-hand-side vectors.\n\
Input parameters include:\n\
  -ntimes <ntimes>  : number of linear systems to solve\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

/*T
   Concepts: KSP^repeatedly solving linear systems;
   Concepts: KSP^Laplacian, 2d
   Concepts: Laplacian, 2d
   Processors: n
T*/

/* 
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;  /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PetscReal      norm;     /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       ntimes,i,j,k,Ii,J,Istart,Iend;
  PetscInt       m = 8,n = 7,its;
  PetscBool      flg = PETSC_FALSE;
  PetscScalar    v,one = 1.0,neg_one = -1.0,rhs;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

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
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  /* 
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.
   */
  for (Ii=Istart; Ii<Iend; Ii++) { 
    v = -1.0; i = Ii/n; j = Ii - i*n;  
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

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
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr); 
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A,SAME_PRECONDITIONER);CHKERRQ(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

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
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ntimes",&ntimes,PETSC_NULL);CHKERRQ(ierr);
  for (k=1; k<ntimes+1; k++) {

    /* 
       Set exact solution; then compute right-hand-side vector.  We use
       an exact solution of a vector with all elements equal to 1.0*k.
    */
    rhs = one * (PetscReal)k;
    ierr = VecSet(u,rhs);CHKERRQ(ierr);
    ierr = MatMult(A,u,b);CHKERRQ(ierr);

    /*
       View the exact solution vector if desired
    */
    ierr = PetscOptionsGetBool(PETSC_NULL,"-view_exact_sol",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    /* 
       Check the error
    */
    ierr = VecAXPY(x,neg_one,u);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    /*
       Print convergence information.  PetscPrintf() produces a single 
       print statement from all processes that share a communicator.
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G System %D: iterations %D\n",norm,k,its);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  ierr = PetscFinalize();
  return 0;
}
