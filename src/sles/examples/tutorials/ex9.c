#ifndef lint
static char vcid[] = "$Id: ex19.c,v 1.13 1996/08/14 02:10:09 curfman Exp curfman $";
#endif

static char help[] = "Illustrates the solution of 2 different linear systems\n\
with different linear solvers.  Also, this example illustrates the repeated\n\
solution of linear systems, while reusing matrix, vector, and solver data\n\
structures throughout the process.  Note the various stages of event logging.\n\n";

/*T
   Concepts: SLES, solving linear equations
   Routines: SLESCreate(), SLESSetOperators(), SLESSetFromOptions()
   Routines: SLESSolve(), SLESView()
   Routines: PLogEventRegister(), PLogEventBegin(), PLogEventEnd()
   Routines: PLogStageRegister(), PLogStagePush(), PLogStagePop()
   Multiprocessor code
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"
#include  <stdio.h>

/* 
   Declare user-defined routines
*/
int CheckError(Vec,Vec,Vec,int,int);
int MyKSPMonitor(KSP,int,double,void*);

int main(int argc,char **args)
{
  Vec    x1, b1, x2, b2; /* solution and RHS vectors for systems #1 and #2 */
  Vec    u;              /* exact solution vector */
  Mat    C1, C2;         /* matrices for systems #1 and #2 */
  SLES   sles1, sles2;   /* SLES contexts for systems #1 and #2 */
  KSP    ksp1;           /* KSP context for system #1 */
  int    ntimes = 3;     /* number of times to solve the linear systems */
  int    CHECK_ERROR;    /* event number for error checking */
  int    ldim, ierr, low, high, iglobal, Istart, Iend, Istart2, Iend2;
  int    I, J, i, j, m = 3, n = 2, rank, size, its, flg, t;
  Scalar v;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-t",&ntimes,&flg); CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  n = 2*size;

  /* 
     Register various stages for profiling
  */
  PLogStageRegister(0,"Prelim setup");
  PLogStageRegister(1,"Linear System 1");
  PLogStageRegister(2,"Linear System 2");

  /* 
     Register a user-defined event for logging (error checking).
  */
  CHECK_ERROR = 0;
  PLogEventRegister(&CHECK_ERROR,"Check Error     ","Red:");

  /* ---------------------- Stage 0: ---------------------------- */
  /*                    Preliminary Setup                         */
  /* ------------------------------------------------------------ */

  PLogStagePush(0);

  /* 
     Create data structures for first linear system.
      - Create parallel matrix, specifying only its global dimensions.
        When using MatCreate(), the matrix format can be specified at
        runtime. Also, the parallel partioning of the matrix is
        determined by PETSc at runtime.
      - Create parallel vectors.
        - When using VecCreate(), we specify only the vector's global
          dimension; the parallel partitioning is determined at runtime. 
        - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C1); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C1,&Istart,&Iend); CHKERRA(ierr);
  ierr = VecCreate(MPI_COMM_WORLD,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b1); CHKERRA(ierr);
  ierr = VecDuplicate(u,&x1); CHKERRA(ierr);

  /*
     Create first linear solver context.
     Set runtime options (e.g., -_pc_type <type>).
     Note that the first linear system uses the default option
     names, while the second linear systme uses a different
     options prefix.
  */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles1); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles1); CHKERRA(ierr);

  /* 
     Set user-defined monitoring routine for first linear system
  */
  ierr = OptionsHasName(PETSC_NULL,"-my_ksp_monitor",&flg); CHKERRA(ierr);
  ierr = SLESGetKSP(sles1,&ksp1); CHKERRA(ierr);
  if (flg) {ierr = KSPSetMonitor(ksp1,MyKSPMonitor,PETSC_NULL); CHKERRA(ierr);}

  /*
     Create data structures for second linear system
  */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&C2); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C2,&Istart2,&Iend2); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b2); CHKERRA(ierr);
  ierr = VecDuplicate(u,&x2); CHKERRA(ierr);

  /*
     Create second linear solver context
  */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles2); CHKERRA(ierr);

  /* 
     Set different options prefix for second linear system
     Set runtime options (e.g., -s2_pc_type <type>)
  */
  ierr = SLESAppendOptionsPrefix(sles2,"s2_"); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles2); CHKERRA(ierr);

  /* Set exact solution vector */
  ierr = VecGetLocalSize(u,&ldim); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,ADD_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  PLogFlops(2*ldim);
  PLogStagePop();

  /* ---------------- Linear solver loop  ---------------------- */

  /* Solve 2 different linear systems several times in succession */
  for (t=0; t<ntimes; t++) {

    /* ---------------------- Stage 1: ---------------------------- */
    /*           Assemble and solve first linear system             */
    /* ------------------------------------------------------------ */

    /* Assemble first matrix */
    PLogStagePush(1);
    ierr = MatZeroEntries(C1); CHKERRA(ierr);
    for ( I=Istart; I<Iend; I++ ) { 
      v = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 )   {J = I - n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(C1,1,&I,1,&I,&v,ADD_VALUES);
    }
    for ( I=Istart; I<Iend; I++ ) { /* Make matrix nonsymmetric */
      v = -1.0*(t+0.5); i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C1,1,&I,1,&J,&v,ADD_VALUES);}
    }
    PLogFlops(2*(Istart-Iend));
    ierr = MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive linear system matrices */
    ierr = MatSetOption(C1,MAT_NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C1,u,b1); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive preconditioner
       matrices by setting SAME_NONZERO_PATTERN below */
    ierr = SLESSetOperators(sles1,C1,C1,SAME_NONZERO_PATTERN); CHKERRA(ierr);

    /* Use the previous solution of linear system #1 as the initial guess
       for the next solve of linear system #1 */
    if (t>0) {
      ierr = KSPSetInitialGuessNonzero(ksp1); CHKERRA(ierr);
    }

    /* Solve first linear system */
    ierr = SLESSetUp(sles1,b1,x1); CHKERRA(ierr);
    ierr = SLESSolve(sles1,b1,x1,&its); CHKERRA(ierr);
    ierr = CheckError(u,x1,b1,its,CHECK_ERROR); CHKERRA(ierr); 

    /* ---------------------- Stage 2: ---------------------------- */
    /*           Assemble and solve second linear system            */
    /* ------------------------------------------------------------ */

    PLogStagePop();
    PLogStagePush(2);

    /* Assemble second matrix */
    ierr = MatZeroEntries(C2); CHKERRA(ierr);
    for ( i=0; i<m; i++ ) { 
      for ( j=2*rank; j<2*rank+2; j++ ) {
        v = -1.0;  I = j + n*i;
        if ( i>0 )   {J = I - n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( i<m-1 ) {J = I + n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( j>0 )   {J = I - 1; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        if ( j<n-1 ) {J = I + 1; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
        v = 6.0 + t*0.5; ierr = MatSetValues(C2,1,&I,1,&I,&v,ADD_VALUES); CHKERRA(ierr);
      }
    } 
    for ( I=Istart2; I<Iend2; I++ ) { /* Make matrix nonsymmetric */
      v = -1.0*(t+0.5); i = I/n;
      if ( i>0 )   {J = I - n; MatSetValues(C2,1,&I,1,&J,&v,ADD_VALUES);}
    }
    PLogFlops(2*(Istart-Iend));
    ierr = MatAssemblyBegin(C2,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
    ierr = MatAssemblyEnd(C2,MAT_FINAL_ASSEMBLY); CHKERRA(ierr); 

    /* Indicate same nonzero structure of successive linear system matrices */
    ierr = MatSetOption(C2,MAT_NO_NEW_NONZERO_LOCATIONS); CHKERRA(ierr);

    /* Compute right-hand-side */
    ierr = MatMult(C2,u,b2); CHKERRA(ierr);

    /* Indicate same nonzero structure of successive preconditioner
       matrices by setting SAME_NONZERO_PATTERN below */
    ierr = SLESSetOperators(sles2,C2,C2,SAME_NONZERO_PATTERN); CHKERRA(ierr);

    /* Solve second linear system */
    ierr = SLESSetUp(sles2,b2,x2); CHKERRA(ierr);
    ierr = SLESSolve(sles2,b2,x2,&its); CHKERRA(ierr);
    ierr = CheckError(u,x2,b2,its,CHECK_ERROR); CHKERRA(ierr); 
    PLogStagePop();
  }

  /* Free work space */
  ierr = SLESDestroy(sles1); CHKERRA(ierr); ierr = SLESDestroy(sles2); CHKERRA(ierr);
  ierr = VecDestroy(x1); CHKERRA(ierr);     ierr = VecDestroy(x2); CHKERRA(ierr);
  ierr = VecDestroy(b1); CHKERRA(ierr);     ierr = VecDestroy(b2); CHKERRA(ierr);
  ierr = MatDestroy(C1); CHKERRA(ierr);     ierr = MatDestroy(C2); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* ------------------------------------------------------------- */
/*
    CheckError - Checks the error of the solution.

    Input Parameters:
    u - exact solution
    x - approximate solution
    b - work vector
    its - number of iterations for convergence
    CHECK_ERROR - the event number for error checking
                  (for use with profiling)

    Notes:
    In order to profile this section of code separately from the
    rest of the program, we register it as an "event" with
    PLogEventRegister() in the main program.  Then, we indicate
    the start and end of this event by respectively calling
        PLogEventBegin(CHECK_ERROR,u,x,b,0);
        PLogEventEnd(CHECK_ERROR,u,x,b,0);
    Here, we specify the objects most closely associated with
    the event (the vectors u,x,b).  Such information is optional;
    we could instead just use 0 instead for all objects.
*/
int CheckError(Vec u,Vec x,Vec b,int its,int CHECK_ERROR)
{
  Scalar none = -1.0;
  double norm;
  int    ierr;

  PLogEventBegin(CHECK_ERROR,u,x,b,0);

  /*
     Compute error of the solution, using b as a work vector.
  */
  ierr = VecCopy(x,b); CHKERRQ(ierr);
  ierr = VecAXPY(&none,u,b); CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&norm); CHKERRQ(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);
  PLogEventEnd(CHECK_ERROR,u,x,b,0);
  return 0;
}
/* ------------------------------------------------------------- */
/*
   MyKSPMonitor - This is a user-defined routine for monitoring
   the SLES iterative solvers.

   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     dummy - optional user-defined monitor context (unused here)
*/
int MyKSPMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  Vec      x;
  int      ierr;

  /* 
     Build the solution vector
  */
  ierr = KSPBuildSolution(ksp,PETSC_NULL,&x); CHKERRQ(ierr);

  /*
     Write the solution vector and residual norm to stdout.
      - PetscPrintf() handles output for multiprocessor jobs 
        by printing from only one processor in the communicator.
      - The parallel viewer VIEWER_STDOUT_WORLD handles
        data from multiple processors so that the output
        is not jumbled.
  */
  PetscPrintf(MPI_COMM_WORLD,"iteration %d solution vector:\n",n);
  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(MPI_COMM_WORLD,"iteration %d KSP Residual norm %14.12e \n",n,rnorm);
  return 0;
}

