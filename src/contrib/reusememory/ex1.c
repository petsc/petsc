
/*
      Demonstrates using MatReleaseValuesMemory() memory and MatRestoreValuesMemory()
   to allow reuse of memory for another calculation while a matrix is inactive.

   This is a hack job for a demonstration of how this may be done.
*/
#include "sles.h"

extern int MatRestoreValuesMemory(Mat);
extern int MatReleaseValuesMemory(Mat);

int main(int argc,char **args)
{
  Vec        x, b, u;      /* approx solution, RHS, exact solution */
  Mat        A;            /* linear system matrix */
  SLES       sles;         /* linear solver context */
  double     norm;         /* norm of solution error */
  int        i, j, I, J, Istart, Iend, ierr, m = 8, n = 7, its, flg,k;
  int        rank;
  Scalar     v, one = 1.0, none = -1.0;
  PLogDouble mem,memmax;

  PetscInitialize(&argc,&args,(char *)0,(char*)0);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,5,0,2,0,&A); 
    CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);

    /* 
       Create parallel vectors.
        - When using VecCreate(), we specify only the vector's global
          dimension; the parallel partitioning is determined at runtime. 
        - Note: We form 1 vector from scratch and then duplicate as needed.
    */
    ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
    ierr = VecSetFromOptions(u);CHKERRA(ierr); /* this sets the vector type seq or mpi */
    ierr = VecDuplicate(u,&b); CHKERRA(ierr); 
    ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /*
        Loop 20 times assembling the matrix and solving, reusing the
     data structures
  */
  for ( k=0; k<5; k++ ) {

    /* 
       Set matrix elements for the 2-D, five-point stencil in parallel.
        - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly). 
        - Always specify global row and columns of matrix entries.
     */
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);
    }

    /* 
       Assemble matrix, using the 2-step process:
         MatAssemblyBegin(), MatAssemblyEnd()
       Computations can be done while messages are in transition,
       by placing code between these two statements.
    */
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);


    /* 
       Set exact solution; then compute right-hand-side vector.
    */
    ierr = VecSet(&one,u); CHKERRA(ierr);
    ierr = MatMult(A,u,b); CHKERRA(ierr);


    /* 
       Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix.
    */
    ierr = SLESSetOperators(sles,A,A,SAME_NONZERO_PATTERN); CHKERRA(ierr);


    /* 
      Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      SLESSetFromOptions() is called _after_ any other customization
      routines.
    */
    ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* 
      Check the error
    */
    ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
    ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
    if (norm > 1.e-12)
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
    else 
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

    PetscMallocGetCurrentUsage(&mem);
    PetscMallocGetMaximumUsage(&mem);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Before release: memory usage %g high water %g\n",rank,mem,memmax);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
    ierr = MatReleaseValuesMemory(A); CHKERRQ(ierr);

    PetscMallocGetCurrentUsage(&mem);
    PetscMallocGetMaximumUsage(&mem);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] After release: memory usage %g high water %g\n",rank,mem,memmax);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);

    ierr = MatRestoreValuesMemory(A); CHKERRQ(ierr);

    PetscMallocGetCurrentUsage(&mem);
    PetscMallocGetMaximumUsage(&mem);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] After restore: memory usage %g high water %g\n",rank,mem,memmax);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);

  }

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

