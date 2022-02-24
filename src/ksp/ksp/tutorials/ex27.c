
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

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)fd,PETSCVIEWERHDF5,&hdf5));
  if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(PetscViewerHDF5HasObject(fd,(PetscObject)b,has));
    if (*has) CHKERRQ(VecLoad(b,fd));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    PetscErrorCode ierrp;
    CHKERRQ(PetscPushErrorHandler(PetscReturnErrorHandler,NULL));
    ierrp = VecLoad(b,fd);
    CHKERRQ(PetscPopErrorHandler());
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  /*
     Determine files from which we read the linear system
     (matrix, right-hand-side and initial guess vector).
  */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f_x0",file_x0,sizeof(file_x0),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-A_name",A_name,sizeof(A_name),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-b_name",b_name,sizeof(b_name),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-x0_name",x0_name,sizeof(x0_name),NULL));
  /*
     Decide whether to solve the original system (-solve_normal 0)
     or the normal equation (-solve_normal 1).
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-solve_normal",&solve_normal,NULL));
  /*
     Decide whether to use the HDF5 reader.
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-hdf5",&hdf5,NULL));
  /*
     Decide whether custom matrix layout will be tested.
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_custom_layout",&test_custom_layout,NULL));

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
    CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
    CHKERRQ(PetscViewerPushFormat(fd,PETSC_VIEWER_HDF5_MAT));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  }

  /*
     Load the matrix.
     Matrix type is set automatically but you can override it by MatSetType() prior to MatLoad().
     Do that only if you really insist on the given type.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,A_name));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,fd));
  if (test_custom_layout && size > 1) {
    /* Perturb the local sizes and create the matrix anew */
    PetscInt m1,n1;
    CHKERRQ(MatGetLocalSize(A,&m,&n));
    m = rank ? m-1 : m+size-1;
    n = (rank == size-1) ? n+size-1 : n-1;
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(PetscObjectSetName((PetscObject)A,A_name));
    CHKERRQ(MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatLoad(A,fd));
    CHKERRQ(MatGetLocalSize(A,&m1,&n1));
    PetscCheckFalse(m1 != m || n1 != n,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"resulting sizes differ from demanded ones: %D %D != %D %D",m1,n1,m,n);
  }
  CHKERRQ(MatGetLocalSize(A,&m,&n));

  /*
     Load the RHS vector if it is present in the file, otherwise use a vector of all ones.
  */
  CHKERRQ(MatCreateVecs(A, &x, &b));
  CHKERRQ(PetscObjectSetName((PetscObject)b,b_name));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecLoadIfExists_Private(b,fd,&has));
  if (!has) {
    PetscScalar one = 1.0;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Failed to load RHS, so use a vector of all ones.\n"));
    CHKERRQ(VecSetFromOptions(b));
    CHKERRQ(VecSet(b,one));
  }

  /*
     Load the initial guess vector if it is present in the file, otherwise use a vector of all zeros.
  */
  CHKERRQ(PetscObjectSetName((PetscObject)x,x0_name));
  CHKERRQ(VecSetFromOptions(x));
  /* load file_x0 if it is specified, otherwise try to reuse file */
  if (file_x0[0]) {
    CHKERRQ(PetscViewerDestroy(&fd));
    if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
      CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd));
#endif
    } else {
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd));
    }
  }
  CHKERRQ(VecLoadIfExists_Private(x,fd,&has));
  if (!has) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Failed to load initial guess, so use a vector of all zeros.\n"));
    CHKERRQ(VecSet(x,0.0));
    nonzero_guess=PETSC_FALSE;
  }
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(VecDuplicate(x,&Ab));

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Conclude profiling last stage; begin profiling next stage.
  */
  PetscPreLoadStage("KSPSetUp");

  CHKERRQ(MatCreateNormalHermitian(A,&N));
  CHKERRQ(MatMultHermitianTranspose(A,b,Ab));

  /*
     Create linear solver; set operators; set runtime options.
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  if (solve_normal) {
    CHKERRQ(KSPSetOperators(ksp,N,N));
  } else {
    PC pc;
    CHKERRQ(KSPSetType(ksp,KSPLSQR));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
    CHKERRQ(KSPSetOperators(ksp,A,N));
  }
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,nonzero_guess));
  CHKERRQ(KSPSetFromOptions(ksp));

  /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
  */
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSetUpOnBlocks(ksp));

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
    CHKERRQ(KSPSolve(ksp,Ab,x));
  } else {
    CHKERRQ(KSPSolve(ksp,b,x));
  }
  CHKERRQ(PetscObjectSetName((PetscObject)x,"x"));

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
  CHKERRQ(VecDuplicate(b,&r));
  CHKERRQ(MatMult(A,x,r));
  CHKERRQ(VecAXPY(r,-1.0,b));
  CHKERRQ(VecNorm(r,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(KSPGetType(ksp,&ksptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP type: %s\n",ksptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(MatDestroy(&A)); CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&N)); CHKERRQ(VecDestroy(&Ab));
  CHKERRQ(VecDestroy(&r)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));
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
