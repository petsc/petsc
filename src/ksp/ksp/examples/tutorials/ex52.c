
/* Program usage:  mpiexec -n <procs> ex52 [-help] [all PETSc options] */ 

static char help[] = "Solves a linear system in parallel with KSP. Modified from ex2.c \n\
                      Illustrate how to use external packages MUMPS and SUPERLU \n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its;
  PetscErrorCode ierr;
  PetscBool      flg,flg_ilu,flg_ch;
  PetscScalar    v;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,PETSC_NULL);CHKERRQ(ierr);

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

     Note: this uses the less common natural ordering that orders first
     all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
     instead of J = I +- m as you might expect. The more standard ordering
     would first do all variables for y = h, then y = 2h etc.

   */
  ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
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
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  /* 
     Create parallel vectors.
      - We form 1 vector from scratch and then duplicate as needed.
      - When using VecCreate(), VecSetSizes and VecSetFromOptions()
        in this example, we specify only the
        vector's global dimension; the parallel partitioning is determined
        at runtime. 
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.  
      - The user can alternatively specify the local vector and matrix
        dimensions when more sophisticated partitioning is needed
        (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
        below).
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr); 
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  ierr = PetscOptionsGetBool(PETSC_NULL,"-random_exact_sol",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,1.0);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /*
     View the exact solution vector if desired
  */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_exact_sol",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /*
    Example of how to use external package MUMPS 
    Note: runtime options 
          '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package mumps -mat_mumps_icntl_7 2' 
          are equivalent to these procedual calls 
  */
#ifdef PETSC_HAVE_MUMPS 
  flg    = PETSC_FALSE;
  flg_ch = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_mumps_lu",&flg,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_mumps_ch",&flg_ch,PETSC_NULL);CHKERRQ(ierr);
  if (flg || flg_ch){
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    PC       pc;
    Mat      F;
    PetscInt ival,icntl;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg){
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ch) {
      ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr); /* set MUMPS id%SYM=1 */
      ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    }
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    icntl=7; ival = 2;
    ierr = MatMumpsSetIcntl(F,icntl,ival);CHKERRQ(ierr);
  }
#endif

  /*
    Example of how to use external package SuperLU
    Note: runtime options 
          '-ksp_type preonly -pc_type ilu -pc_factor_mat_solver_package superlu -mat_superlu_ilu_droptol 1.e-8' 
          are equivalent to these procedual calls 
  */
#ifdef PETSC_HAVE_SUPERLU
  flg_ilu = PETSC_FALSE;
  flg     = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_superlu_lu",&flg,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_superlu_ilu",&flg_ilu,PETSC_NULL);CHKERRQ(ierr);
  if (flg || flg_ilu){
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    PC       pc;
    Mat      F;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg){
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ilu) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr); 
    }
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    ierr = MatSuperluSetILUDropTol(F,1.e-8);CHKERRQ(ierr);
  }
#endif

  /*
    Example of how to use procedural calls that are equivalent to 
          '-ksp_type preonly -pc_type lu/ilu -pc_factor_mat_solver_package petsc' 
  */
  flg     = PETSC_FALSE;
  flg_ilu = PETSC_FALSE;
  flg_ch  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_petsc_lu",&flg,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_petsc_ilu",&flg_ilu,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_petsc_ch",&flg_ch,PETSC_NULL);CHKERRQ(ierr);
  if (flg || flg_ilu || flg_ch){
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    PC       pc;
    Mat      F;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg){
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ilu) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr); 
    } else if (flg_ch) {
      ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr); 
    }
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERPETSC);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);

    /* Test MatGetDiagonal() */
    Vec diag;
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&diag);CHKERRQ(ierr);
    ierr = MatGetDiagonal(F,diag);CHKERRQ(ierr);
    ierr = VecView(diag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&diag);CHKERRQ(ierr); 
  }
  
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
  /* Scale the norm */
  /*  norm *= sqrt(1.0/((m+1)*(n+1))); */

  /*
     Print convergence information.  PetscPrintf() produces a single 
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G iterations %D\n",
                     norm,its);CHKERRQ(ierr);

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
