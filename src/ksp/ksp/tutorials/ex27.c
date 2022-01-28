
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
#include <petscviewerhdf5.h>

static PetscErrorCode VecLoadIfExists_Private(Vec b,PetscViewer fd,PetscBool *has)
{
  PetscBool      hdf5=PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)fd,PETSCVIEWERHDF5,&hdf5);CHKERRQ(ierr);
  if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerHDF5HasObject(fd,(PetscObject)b,has);CHKERRQ(ierr);
    if (*has) {ierr = VecLoad(b,fd);CHKERRQ(ierr);}
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    PetscErrorCode ierrp;
    ierr  = PetscPushErrorHandler(PetscReturnErrorHandler,NULL);CHKERRQ(ierr);
    ierrp = VecLoad(b,fd);
    ierr  = PetscPopErrorHandler();CHKERRQ(ierr);
    *has  = ierrp ? PETSC_FALSE : PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A,N;                /* matrix */
  Vec            x,b,r,Ab;          /* approx solution, RHS, residual */
  PetscViewer    fd;               /* viewer */
  char           file[PETSC_MAX_PATH_LEN]="";     /* input file name */
  char           file_x0[PETSC_MAX_PATH_LEN]="";  /* name of input file with initial guess */
  char           A_name[128]="A",b_name[128]="b",x0_name[128]="x0";  /* name of the matrix, RHS and initial guess */
  KSPType        ksptype;
  PetscErrorCode ierr;
  PetscBool      has;
  PetscInt       its,n,m;
  PetscReal      norm;
  PetscBool      nonzero_guess=PETSC_TRUE;
  PetscBool      solve_normal=PETSC_FALSE;
  PetscBool      hdf5=PETSC_FALSE;
  PetscBool      test_custom_layout=PETSC_FALSE;
  PetscMPIInt    rank,size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  /*
     Determine files from which we read the linear system
     (matrix, right-hand-side and initial guess vector).
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f_x0",file_x0,sizeof(file_x0),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-A_name",A_name,sizeof(A_name),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-b_name",b_name,sizeof(b_name),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-x0_name",x0_name,sizeof(x0_name),NULL);CHKERRQ(ierr);
  /*
     Decide whether to solve the original system (-solve_normal 0)
     or the normal equation (-solve_normal 1).
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-solve_normal",&solve_normal,NULL);CHKERRQ(ierr);
  /*
     Decide whether to use the HDF5 reader.
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-hdf5",&hdf5,NULL);CHKERRQ(ierr);
  /*
     Decide whether custom matrix layout will be tested.
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_custom_layout",&test_custom_layout,NULL);CHKERRQ(ierr);

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
  if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(fd,PETSC_VIEWER_HDF5_MAT);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  }

  /*
     Load the matrix.
     Matrix type is set automatically but you can override it by MatSetType() prior to MatLoad().
     Do that only if you really insist on the given type.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,A_name);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  if (test_custom_layout && size > 1) {
    /* Perturb the local sizes and create the matrix anew */
    PetscInt m1,n1;
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    m = rank ? m-1 : m+size-1;
    n = (rank == size-1) ? n+size-1 : n-1;
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A,A_name);CHKERRQ(ierr);
    ierr = MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A,fd);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
    if (m1 != m || n1 != n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"resulting sizes differ from demanded ones: %D %D != %D %D",m1,n1,m,n);
  }
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);

  /*
     Load the RHS vector if it is present in the file, otherwise use a vector of all ones.
  */
  ierr = MatCreateVecs(A, &x, &b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)b,b_name);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecLoadIfExists_Private(b,fd,&has);CHKERRQ(ierr);
  if (!has) {
    PetscScalar one = 1.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Failed to load RHS, so use a vector of all ones.\n");CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = VecSet(b,one);CHKERRQ(ierr);
  }

  /*
     Load the initial guess vector if it is present in the file, otherwise use a vector of all zeros.
  */
  ierr = PetscObjectSetName((PetscObject)x,x0_name);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  /* load file_x0 if it is specified, otherwise try to reuse file */
  if (file_x0[0]) {
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
      ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd);CHKERRQ(ierr);
#endif
    } else {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    }
  }
  ierr = VecLoadIfExists_Private(x,fd,&has);CHKERRQ(ierr);
  if (!has) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Failed to load initial guess, so use a vector of all zeros.\n");CHKERRQ(ierr);
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    nonzero_guess=PETSC_FALSE;
  }
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = VecDuplicate(x,&Ab);CHKERRQ(ierr);

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Conclude profiling last stage; begin profiling next stage.
  */
  PetscPreLoadStage("KSPSetUp");

  ierr = MatCreateNormalHermitian(A,&N);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(A,b,Ab);CHKERRQ(ierr);

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
    ierr = KSPSetOperators(ksp,A,N);CHKERRQ(ierr);
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
  ierr = PetscObjectSetName((PetscObject)x,"x");CHKERRQ(ierr);

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
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/medium -ksp_view -ksp_monitor_short -ksp_max_it 100 -solve_normal

   test:
      suffix: 2
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -ksp_view -ksp_monitor_short -ksp_max_it 100 -solve_normal

   # Test handling failing VecLoad without abort
   testset:
     requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
     args: -ksp_type cg -ksp_view -ksp_converged_reason -ksp_monitor_short -ksp_max_it 10
     test:
        suffix: 3
        nsize: {{1 2}separate output}
        args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
        args: -f_x0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_x0
     test:
        suffix: 3a
        nsize: {{1 2}separate output}
        args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
        args: -f_x0 NONEXISTING_FILE
     test:
        suffix: 3b
        nsize: {{1 2}separate output}
        args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_with_x0  # this file includes all A, b and x0
     test:
        # Load square matrix, RHS and initial guess from HDF5 (Version 7.3 MAT-File)
        suffix: 3b_hdf5
        requires: hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
        nsize: {{1 2}separate output}
        args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_with_x0.mat -hdf5

   # Test least-square algorithms
   testset:
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
     args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
     test:
        suffix: 4
        nsize: {{1 2 4}}
        args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 100
        args: -solve_normal -ksp_type cg
     test:
        suffix: 4a
        nsize: {{1 2 4}}
        args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 100
        args: -ksp_type {{cgls lsqr}separate output}
     test:
        # Test KSPLSQR-specific options
        suffix: 4b
        nsize: 2
        args: -ksp_converged_reason -ksp_rtol 1e-3 -ksp_max_it 200 -ksp_view
        args: -ksp_type lsqr -ksp_convergence_test lsqr -ksp_lsqr_monitor -ksp_lsqr_compute_standard_error -ksp_lsqr_exact_mat_norm {{0 1}separate output}

   test:
      # Load rectangular matrix from HDF5 (Version 7.3 MAT-File)
      suffix: 4a_lsqr_hdf5
      nsize: {{1 2 4 8}}
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/rectangular_ultrasound_4889x841.mat -hdf5
      args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 100
      args: -ksp_type lsqr
      args: -test_custom_layout {{0 1}}

   # Test for correct cgls convergence reason
   test:
      suffix: 5
      nsize: 1
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/rectangular_ultrasound_4889x841
      args: -ksp_converged_reason -ksp_rtol 1e-2 -ksp_max_it 100
      args: -ksp_type cgls

   # Load a matrix, RHS and solution from HDF5 (Version 7.3 MAT-File). Test immediate convergence.
   testset:
     nsize: {{1 2 4 8}}
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
     args: -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-5 -ksp_max_it 10
     args: -ksp_type lsqr
     args: -test_custom_layout {{0 1}}
     args: -hdf5 -x0_name x
     test:
        suffix: 6_hdf5
        args: -f ${DATAFILESPATH}/matrices/matlab/small.mat
     test:
        suffix: 6_hdf5_rect
        args: -f ${DATAFILESPATH}/matrices/matlab/small_rect.mat
     test:
        suffix: 6_hdf5_dense
        args: -f ${DATAFILESPATH}/matrices/matlab/small_dense.mat -mat_type dense
     test:
        suffix: 6_hdf5_rect_dense
        args: -f ${DATAFILESPATH}/matrices/matlab/small_rect_dense.mat -mat_type dense

   # Test correct handling of local dimensions in PCApply
   testset:
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
     requires: hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
     nsize: 3
     suffix: 7
     args: -f ${DATAFILESPATH}/matrices/matlab/small.mat -hdf5 -test_custom_layout 1 -ksp_type lsqr -pc_type jacobi

   # Test complex matrices
   testset:
     requires: double complex !defined(PETSC_USE_64BIT_INDICES)
     args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/nh-complex-int32-float64
     output_file: output/ex27_8.out
     filter: grep -v "KSP type"
     test:
       suffix: 8
       args: -solve_normal 0 -ksp_type {{lsqr cgls}}
     test:
       suffix: 8_normal
       args: -solve_normal 1 -ksp_type {{cg bicg}}

   testset:
     requires: double suitesparse !defined(PETSC_USE_64BIT_INDICES)
     args: -solve_normal {{0 1}shared output} -pc_type qr
     output_file: output/ex27_9.out
     filter: grep -v "KSP type"
     test:
       suffix: 9_real
       requires: !complex
       args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
     test:
       suffix: 9_complex
       requires: complex
       args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/nh-complex-int32-float64

TEST*/
