
static char help[] = "The solution of 2 different linear systems with different linear solvers.\n\
Also, this example illustrates the repeated\n\
solution of linear systems, while reusing matrix, vector, and solver data\n\
structures throughout the process.  Note the various stages of event logging.\n\n";

/*T
   Concepts: KSP^repeatedly solving linear systems;
   Concepts: PetscLog^profiling multiple stages of code;
   Concepts: PetscLog^user-defined event profiling;
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

/*
   Declare user-defined routines
*/
extern PetscErrorCode CheckError(Vec,Vec,Vec,PetscInt,PetscReal,PetscLogEvent);
extern PetscErrorCode MyKSPMonitor(KSP,PetscInt,PetscReal,void*);

int main(int argc,char **args)
{
  Vec            x1,b1,x2,b2; /* solution and RHS vectors for systems #1 and #2 */
  Vec            u;              /* exact solution vector */
  Mat            C1,C2;         /* matrices for systems #1 and #2 */
  KSP            ksp1,ksp2;   /* KSP contexts for systems #1 and #2 */
  PetscInt       ntimes = 3;     /* number of times to solve the linear systems */
  PetscLogEvent  CHECK_ERROR;    /* event number for error checking */
  PetscInt       ldim,low,high,iglobal,Istart,Iend,Istart2,Iend2;
  PetscInt       Ii,J,i,j,m = 3,n = 2,its,t;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE, unsym = PETSC_TRUE;
  PetscScalar    v;
  PetscMPIInt    rank,size;
#if defined(PETSC_USE_LOG)
  PetscLogStage stages[3];
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-t",&ntimes,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-unsym",&unsym,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = 2*size;

  /*
     Register various stages for profiling
  */
  ierr = PetscLogStageRegister("Prelim setup",&stages[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Linear System 1",&stages[1]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Linear System 2",&stages[2]);CHKERRQ(ierr);

  /*
     Register a user-defined event for profiling (error checking).
  */
  CHECK_ERROR = 0;
  ierr        = PetscLogEventRegister("Check Error",KSP_CLASSID,&CHECK_ERROR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - Stage 0: - - - - - - - - - - - - - -
                        Preliminary Setup
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);

  /*
     Create data structures for first linear system.
      - Create parallel matrix, specifying only its global dimensions.
        When using MatCreate(), the matrix format can be specified at
        runtime. Also, the parallel partitioning of the matrix is
        determined by PETSc at runtime.
      - Create parallel vectors.
        - When using VecSetSizes(), we specify only the vector's global
          dimension; the parallel partitioning is determined at runtime.
        - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&C1);CHKERRQ(ierr);
  ierr = MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C1);CHKERRQ(ierr);
  ierr = MatSetUp(C1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C1,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b1);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x1);CHKERRQ(ierr);

  /*
     Create first linear solver context.
     Set runtime options (e.g., -pc_type <type>).
     Note that the first linear system uses the default option
     names, while the second linear system uses a different
     options prefix.
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp1);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp1);CHKERRQ(ierr);

  /*
     Set user-defined monitoring routine for first linear system.
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-my_ksp_monitor",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = KSPMonitorSet(ksp1,MyKSPMonitor,NULL,0);CHKERRQ(ierr);}

  /*
     Create data structures for second linear system.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&C2);CHKERRQ(ierr);
  ierr = MatSetSizes(C2,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C2);CHKERRQ(ierr);
  ierr = MatSetUp(C2);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C2,&Istart2,&Iend2);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b2);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x2);CHKERRQ(ierr);

  /*
     Create second linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp2);CHKERRQ(ierr);

  /*
     Set different options prefix for second linear system.
     Set runtime options (e.g., -s2_pc_type <type>)
  */
  ierr = KSPAppendOptionsPrefix(ksp2,"s2_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp2);CHKERRQ(ierr);

  /*
     Assemble exact solution vector in parallel.  Note that each
     processor needs to set only its local part of the vector.
  */
  ierr = VecGetLocalSize(u,&ldim);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100*rank);
    ierr    = VecSetValues(u,1,&iglobal,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  /*
     Log the number of flops for computing vector entries
  */
  ierr = PetscLogFlops(2.0*ldim);CHKERRQ(ierr);

  /*
     End curent profiling stage
  */
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* --------------------------------------------------------------
                        Linear solver loop:
      Solve 2 different linear systems several times in succession
     -------------------------------------------------------------- */

  for (t=0; t<ntimes; t++) {

    /* - - - - - - - - - - - - Stage 1: - - - - - - - - - - - - - -
                 Assemble and solve first linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling stage #1
    */
    ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);

    /*
       Initialize all matrix entries to zero.  MatZeroEntries() retains
       the nonzero structure of the matrix for sparse formats.
    */
    if (t > 0) {ierr = MatZeroEntries(C1);CHKERRQ(ierr);}

    /*
       Set matrix entries in parallel.  Also, log the number of flops
       for computing matrix entries.
        - Each processor needs to insert only elements that it owns
          locally (but any non-local elements will be sent to the
          appropriate processor during matrix assembly).
        - Always specify global row and columns of matrix entries.
    */
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; i = Ii/n; j = Ii - i*n;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C1,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C1,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C1,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C1,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C1,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (unsym) {
      for (Ii=Istart; Ii<Iend; Ii++) { /* Make matrix nonsymmetric */
        v = -1.0*(t+0.5); i = Ii/n;
        if (i>0)   {J = Ii - n; ierr = MatSetValues(C1,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      }
      ierr = PetscLogFlops(2.0*(Iend-Istart));CHKERRQ(ierr);
    }

    /*
       Assemble matrix, using the 2-step process:
         MatAssemblyBegin(), MatAssemblyEnd()
       Computations can be done while messages are in transition
       by placing code between these two statements.
    */
    ierr = MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /*
       Indicate same nonzero structure of successive linear system matrices
    */
    ierr = MatSetOption(C1,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);CHKERRQ(ierr);

    /*
       Compute right-hand-side vector
    */
    ierr = MatMult(C1,u,b1);CHKERRQ(ierr);

    /*
       Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix.
    */
    ierr = KSPSetOperators(ksp1,C1,C1);CHKERRQ(ierr);

    /*
       Use the previous solution of linear system #1 as the initial
       guess for the next solve of linear system #1.  The user MUST
       call KSPSetInitialGuessNonzero() in indicate use of an initial
       guess vector; otherwise, an initial guess of zero is used.
    */
    if (t>0) {
      ierr = KSPSetInitialGuessNonzero(ksp1,PETSC_TRUE);CHKERRQ(ierr);
    }

    /*
       Solve the first linear system.  Here we explicitly call
       KSPSetUp() for more detailed performance monitoring of
       certain preconditioners, such as ICC and ILU.  This call
       is optional, ase KSPSetUp() will automatically be called
       within KSPSolve() if it hasn't been called already.
    */
    ierr = KSPSetUp(ksp1);CHKERRQ(ierr);
    ierr = KSPSolve(ksp1,b1,x1);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp1,&its);CHKERRQ(ierr);

    /*
       Check error of solution to first linear system
    */
    ierr = CheckError(u,x1,b1,its,1.e-4,CHECK_ERROR);CHKERRQ(ierr);

    /* - - - - - - - - - - - - Stage 2: - - - - - - - - - - - - - -
                 Assemble and solve second linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Conclude profiling stage #1; begin profiling stage #2
    */
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscLogStagePush(stages[2]);CHKERRQ(ierr);

    /*
       Initialize all matrix entries to zero
    */
    if (t > 0) {ierr = MatZeroEntries(C2);CHKERRQ(ierr);}

    /*
       Assemble matrix in parallel. Also, log the number of flops
       for computing matrix entries.
        - To illustrate the features of parallel matrix assembly, we
          intentionally set the values differently from the way in
          which the matrix is distributed across the processors.  Each
          entry that is not owned locally will be sent to the appropriate
          processor during MatAssemblyBegin() and MatAssemblyEnd().
        - For best efficiency the user should strive to set as many
          entries locally as possible.
     */
    for (i=0; i<m; i++) {
      for (j=2*rank; j<2*rank+2; j++) {
        v = -1.0;  Ii = j + n*i;
        if (i>0)   {J = Ii - n; ierr = MatSetValues(C2,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i<m-1) {J = Ii + n; ierr = MatSetValues(C2,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j>0)   {J = Ii - 1; ierr = MatSetValues(C2,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C2,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        v = 6.0 + t*0.5; ierr = MatSetValues(C2,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (unsym) {
      for (Ii=Istart2; Ii<Iend2; Ii++) { /* Make matrix nonsymmetric */
        v = -1.0*(t+0.5); i = Ii/n;
        if (i>0)   {J = Ii - n; ierr = MatSetValues(C2,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
      }
    }
    ierr = MatAssemblyBegin(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*(Iend-Istart));CHKERRQ(ierr);

    /*
       Indicate same nonzero structure of successive linear system matrices
    */
    ierr = MatSetOption(C2,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);

    /*
       Compute right-hand-side vector
    */
    ierr = MatMult(C2,u,b2);CHKERRQ(ierr);

    /*
       Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix.  Indicate same nonzero
       structure of successive preconditioner matrices by setting flag
       SAME_NONZERO_PATTERN.
    */
    ierr = KSPSetOperators(ksp2,C2,C2);CHKERRQ(ierr);

    /*
       Solve the second linear system
    */
    ierr = KSPSetUp(ksp2);CHKERRQ(ierr);
    ierr = KSPSolve(ksp2,b2,x2);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp2,&its);CHKERRQ(ierr);

    /*
       Check error of solution to second linear system
    */
    ierr = CheckError(u,x2,b2,its,1.e-4,CHECK_ERROR);CHKERRQ(ierr);

    /*
       Conclude profiling stage #2
    */
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  /* --------------------------------------------------------------
                       End of linear solver loop
     -------------------------------------------------------------- */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp1);CHKERRQ(ierr); ierr = KSPDestroy(&ksp2);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);   ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);   ierr = VecDestroy(&b2);CHKERRQ(ierr);
  ierr = MatDestroy(&C1);CHKERRQ(ierr);   ierr = MatDestroy(&C2);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------- */
/*
    CheckError - Checks the error of the solution.

    Input Parameters:
    u - exact solution
    x - approximate solution
    b - work vector
    its - number of iterations for convergence
    tol - tolerance
    CHECK_ERROR - the event number for error checking
                  (for use with profiling)

    Notes:
    In order to profile this section of code separately from the
    rest of the program, we register it as an "event" with
    PetscLogEventRegister() in the main program.  Then, we indicate
    the start and end of this event by respectively calling
        PetscLogEventBegin(CHECK_ERROR,u,x,b,0);
        PetscLogEventEnd(CHECK_ERROR,u,x,b,0);
    Here, we specify the objects most closely associated with
    the event (the vectors u,x,b).  Such information is optional;
    we could instead just use 0 instead for all objects.
*/
PetscErrorCode CheckError(Vec u,Vec x,Vec b,PetscInt its,PetscReal tol,PetscLogEvent CHECK_ERROR)
{
  PetscScalar    none = -1.0;
  PetscReal      norm;
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(CHECK_ERROR,u,x,b,0);CHKERRQ(ierr);

  /*
     Compute error of the solution, using b as a work vector.
  */
  ierr = VecCopy(x,b);CHKERRQ(ierr);
  ierr = VecAXPY(b,none,u);CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(CHECK_ERROR,u,x,b,0);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------- */
/*
   MyKSPMonitor - This is a user-defined routine for monitoring
   the KSP iterative solvers.

   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     dummy - optional user-defined monitor context (unused here)
*/
PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  Vec            x;
  PetscErrorCode ierr;

  /*
     Build the solution vector
  */
  ierr = KSPBuildSolution(ksp,NULL,&x);CHKERRQ(ierr);

  /*
     Write the solution vector and residual norm to stdout.
      - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
      - The parallel viewer PETSC_VIEWER_STDOUT_WORLD handles
        data from multiple processors so that the output
        is not jumbled.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration %D solution vector:\n",n);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration %D KSP Residual norm %14.12e \n",n,rnorm);CHKERRQ(ierr);
  return 0;
}



/*TEST

   test:
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_monitor_short

   test:
      requires: hpddm
      suffix: hpddm
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type {{gmres hpddm}} -s2_ksp_type {{gmres hpddm}} -s2_pc_type jacobi -s2_ksp_monitor_short

   test:
      requires: hpddm
      suffix: hpddm_2
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -s2_ksp_type hpddm -s2_ksp_hpddm_krylov_method gcrodr -s2_ksp_hpddm_recycle 10 -s2_pc_type jacobi -s2_ksp_monitor_short

   testset:
      requires: hpddm
      output_file: output/ex9_hpddm_cg.out
      args: -unsym 0 -t 2 -pc_type jacobi -ksp_monitor_short -s2_pc_type jacobi -s2_ksp_monitor_short -ksp_rtol 1.e-2 -s2_ksp_rtol 1.e-2
      test:
         suffix: hpddm_cg_p_p
         args: -ksp_type cg -s2_ksp_type cg
      test:
         suffix: hpddm_cg_p_h
         args: -ksp_type cg -s2_ksp_type hpddm -s2_ksp_hpddm_krylov_method cg
      test:
         suffix: hpddm_cg_h_h
         args: -ksp_type hpddm -ksp_hpddm_krylov_method cg -s2_ksp_type hpddm -s2_ksp_hpddm_krylov_method cg

TEST*/
