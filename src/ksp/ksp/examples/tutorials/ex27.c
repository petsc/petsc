
static char help[] = "Reads a PETSc matrix and vector from a file and solves the normal equations.\n\n";
/*T
   Concepts: KSP^solving a linear system
   Concepts: Normal equations
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


int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A,N;                /* matrix */
  Vec            x,b,r,Ab;          /* approx solution, RHS, residual */
  PetscViewer    fd;               /* viewer */
  char           file[PETSC_MAX_PATH_LEN]="";     /* input file name */
  char           file_x0[PETSC_MAX_PATH_LEN]="";  /* name of input file with initial guess */
  KSPType        ksptype;
  PetscErrorCode ierr,ierrp;
  PetscInt       its,n,m;
  PetscReal      norm;
  PetscBool      nonzero_guess=PETSC_TRUE;
  PetscBool      solve_normal=PETSC_TRUE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /*
     Determine files from which we read the linear system
     (matrix, right-hand-side and initial guess vector).
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f_x0",file_x0,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  /*
     Decide whether to solve the original system (-solve_normal 0)
     or the normal equation (-solve_normal 1).
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-solve_normal",&solve_normal,NULL);CHKERRQ(ierr);

  /* -----------------------------------------------------------
                  Beginning of linear solver loop
     ----------------------------------------------------------- */
  /*
     Loop through the linear solve 2 times.
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_view) can be done with the larger one (that actually
        is the system of interest).
  */
  PetscPreLoadBegin(PETSC_FALSE,"Load system");

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Load system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr  = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr  = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr  = MatLoad(A,fd);CHKERRQ(ierr);
  ierr  = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr  = PetscPushErrorHandler(PetscIgnoreErrorHandler,NULL);CHKERRQ(ierr);
  ierrp = VecLoad(b,fd);
  ierr  = PetscPopErrorHandler();CHKERRQ(ierr);
  ierr  = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (ierrp) {   /* if file contains no RHS, then use a vector of all ones */
    PetscScalar one = 1.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Failed to load RHS, so use a vector of all ones.\n");CHKERRQ(ierr);
    ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = VecSet(b,one);CHKERRQ(ierr);
  }

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* load file_x0 if it is specified, otherwise try to reuse file */
  if (file_x0[0]) {
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  }
  ierr = PetscPushErrorHandler(PetscIgnoreErrorHandler,NULL);CHKERRQ(ierr);
  ierrp = VecLoad(x,fd);
  ierr = PetscPopErrorHandler();CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  if (ierrp) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Failed to load initial guess, so use a vector of all zeros.\n");CHKERRQ(ierr);
    ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    nonzero_guess=PETSC_FALSE;
  }

  ierr = VecDuplicate(x,&Ab);CHKERRQ(ierr);

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Conclude profiling last stage; begin profiling next stage.
  */
  PetscPreLoadStage("KSPSetUp");

  ierr = MatCreateNormal(A,&N);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,b,Ab);CHKERRQ(ierr);

  /*
     Create linear solver; set operators; set runtime options.
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  if (solve_normal) {
    ierr = KSPSetOperators(ksp,N,N);CHKERRQ(ierr);
  } else {
    PC pc;
    ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  }
  ierr = KSPSetInitialGuessNonzero(ksp,nonzero_guess);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
  */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  /*
                         Solve system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Begin profiling next stage
  */
  PetscPreLoadStage("KSPSolve");

  /*
     Solve linear system
  */
  if (solve_normal) {
    ierr = KSPSolve(ksp,Ab,x);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }

  /*
      Conclude profiling this stage
   */
  PetscPreLoadStage("Cleanup");

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
          Check error, print output, free data structures.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check error
  */
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = MatMult(A,x,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP type: %s\n",ksptype);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&N);CHKERRQ(ierr); ierr = VecDestroy(&Ab);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      suffix: 1
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/medium -ksp_view -ksp_monitor_short -ksp_max_it 100

   test:
      suffix: 2
      nsize: 2
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -ksp_view -ksp_monitor_short -ksp_max_it 100

   # Test handling failing VecLoad without abort
   test:
      suffix: 3
      nsize: {{1 2}separate output}
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
      args: -f_x0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_x0
      args: -ksp_type cg -ksp_view -ksp_converged_reason -ksp_monitor_short -ksp_max_it 10
   test:
      suffix: 3a
      nsize: {{1 2}separate output}
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
      args: -f_x0 NONEXISTING_FILE
      args: -ksp_type cg -ksp_view -ksp_converged_reason -ksp_monitor_short -ksp_max_it 10
   test:
      suffix: 3b
      nsize: {{1 2}separate output}
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_with_x0  # this file includes all A, b and x0
      args: -ksp_type cg -ksp_view -ksp_converged_reason -ksp_monitor_short -ksp_max_it 10

   # Test least-square algorithms
   test:
      suffix: 4
      nsize: {{1 2 4}}
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
      args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 100
      args: -solve_normal 1 -ksp_type cg
   test:
      suffix: 4a
      nsize: {{1 2 4}}
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
      args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 100
      args: -solve_normal 0 -ksp_type {{cgls lsqr}separate output}
   test:
      # Test KSPLSQR-specific options
      suffix: 4b
      nsize: 2
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
      args: -ksp_converged_reason -ksp_rtol 1e-3 -ksp_max_it 200 -ksp_view
      args: -solve_normal 0 -ksp_type lsqr -ksp_convergence_test lsqr -ksp_lsqr_monitor -ksp_lsqr_compute_standard_error -ksp_lsqr_exact_mat_norm {{0 1}separate output}

   # Test for correct cgls convergence reason
   test:
      suffix: 5
      nsize: 1
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
      args: -ksp_converged_reason -ksp_rtol 1e-2 -ksp_max_it 100
      args: -solve_normal 0 -ksp_type cgls

TEST*/
