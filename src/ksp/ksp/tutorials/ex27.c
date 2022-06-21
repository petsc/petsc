
static char help[] = "Reads a PETSc matrix and vector from a file and solves the normal equations.\n\n";

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
  PetscCall(PetscObjectTypeCompare((PetscObject)fd,PETSCVIEWERHDF5,&hdf5));
  if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(PetscViewerHDF5HasObject(fd,(PetscObject)b,has));
    if (*has) PetscCall(VecLoad(b,fd));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    PetscErrorCode ierrp;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,NULL));
    ierrp = VecLoad(b,fd);
    PetscCall(PetscPopErrorHandler());
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
  PetscBool      has;
  PetscInt       its,n,m;
  PetscReal      norm;
  PetscBool      nonzero_guess=PETSC_TRUE;
  PetscBool      solve_normal=PETSC_FALSE;
  PetscBool      truncate=PETSC_FALSE;
  PetscBool      hdf5=PETSC_FALSE;
  PetscBool      test_custom_layout=PETSC_FALSE;
  PetscMPIInt    rank,size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  /*
     Determine files from which we read the linear system
     (matrix, right-hand-side and initial guess vector).
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-truncate",&truncate,NULL));
  if (!truncate) PetscCall(PetscOptionsGetString(NULL,NULL,"-f_x0",file_x0,sizeof(file_x0),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-A_name",A_name,sizeof(A_name),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-b_name",b_name,sizeof(b_name),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-x0_name",x0_name,sizeof(x0_name),NULL));
  /*
     Decide whether to solve the original system (-solve_normal 0)
     or the normal equation (-solve_normal 1).
  */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-solve_normal",&solve_normal,NULL));
  /*
     Decide whether to use the HDF5 reader.
  */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-hdf5",&hdf5,NULL));
  /*
     Decide whether custom matrix layout will be tested.
  */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_custom_layout",&test_custom_layout,NULL));

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
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
    PetscCall(PetscViewerPushFormat(fd,PETSC_VIEWER_HDF5_MAT));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"PETSc must be configured with HDF5 to use this feature");
#endif
  } else {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  }

  /*
     Load the matrix.
     Matrix type is set automatically but you can override it by MatSetType() prior to MatLoad().
     Do that only if you really insist on the given type.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(PetscObjectSetName((PetscObject)A,A_name));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));
  if (truncate) {
    Mat      P,B;
    PetscInt M,N;
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,N/1.5,1,NULL,1,NULL,&P));
    PetscCall(MatGetOwnershipRangeColumn(P,&m,&n));
    for (; m < n; ++m) PetscCall(MatSetValue(P,m,m,1.0,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(P,1.0));
    PetscCall(MatMatMult(A,P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&A));
    A = B;
  }
  if (test_custom_layout && size > 1) {
    /* Perturb the local sizes and create the matrix anew */
    PetscInt m1,n1;
    PetscCall(MatGetLocalSize(A,&m,&n));
    m = rank ? m-1 : m+size-1;
    n = (rank == size-1) ? n+size-1 : n-1;
    PetscCall(MatDestroy(&A));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(PetscObjectSetName((PetscObject)A,A_name));
    PetscCall(MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A,fd));
    PetscCall(MatGetLocalSize(A,&m1,&n1));
    PetscCheck(m1 == m && n1 == n,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"resulting sizes differ from requested ones: %" PetscInt_FMT " %" PetscInt_FMT " != %" PetscInt_FMT " %" PetscInt_FMT,m1,n1,m,n);
  }
  PetscCall(MatGetLocalSize(A,&m,&n));

  /*
     Load the RHS vector if it is present in the file, otherwise use a vector of all ones.
  */
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(PetscObjectSetName((PetscObject)b,b_name));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecLoadIfExists_Private(b,fd,&has));
  if (!has) {
    PetscScalar one = 1.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Failed to load RHS, so use a vector of all ones.\n"));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSet(b,one));
  }

  /*
     Load the initial guess vector if it is present in the file, otherwise use a vector of all zeros.
  */
  PetscCall(PetscObjectSetName((PetscObject)x,x0_name));
  PetscCall(VecSetFromOptions(x));
  if (!truncate) {
    /* load file_x0 if it is specified, otherwise try to reuse file */
    if (file_x0[0]) {
      PetscCall(PetscViewerDestroy(&fd));
      if (hdf5) {
#if defined(PETSC_HAVE_HDF5)
        PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd));
#endif
      } else {
        PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file_x0,FILE_MODE_READ,&fd));
      }
    }
    PetscCall(VecLoadIfExists_Private(x,fd,&has));
  } else has = PETSC_FALSE;
  if (truncate || !has) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Failed to load initial guess, so use a vector of all zeros.\n"));
    PetscCall(VecSet(x,0.0));
    nonzero_guess=PETSC_FALSE;
  }
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(VecDuplicate(x,&Ab));

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Conclude profiling last stage; begin profiling next stage.
  */
  PetscPreLoadStage("KSPSetUp");

  PetscCall(MatCreateNormalHermitian(A,&N));
  PetscCall(MatMultHermitianTranspose(A,b,Ab));

  /*
     Create linear solver; set operators; set runtime options.
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  if (solve_normal) {
    PetscCall(KSPSetOperators(ksp,N,N));
  } else {
    PC pc;
    PetscCall(KSPSetType(ksp,KSPLSQR));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
    PetscCall(KSPSetOperators(ksp,A,N));
  }
  PetscCall(KSPSetInitialGuessNonzero(ksp,nonzero_guess));
  PetscCall(KSPSetFromOptions(ksp));

  /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
  */
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetUpOnBlocks(ksp));

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
    PetscCall(KSPSolve(ksp,Ab,x));
  } else {
    PetscCall(KSPSolve(ksp,b,x));
  }
  PetscCall(PetscObjectSetName((PetscObject)x,"x"));

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
  PetscCall(VecDuplicate(b,&r));
  PetscCall(MatMult(A,x,r));
  PetscCall(VecAXPY(r,-1.0,b));
  PetscCall(VecNorm(r,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(KSPGetType(ksp,&ksptype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSP type: %s\n",ksptype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3" PetscInt_FMT "\n",its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&N)); PetscCall(VecDestroy(&Ab));
  PetscCall(VecDestroy(&r)); PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  PetscCall(PetscFinalize());
  return 0;
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
      args: -f ${DATAFILESPATH}/matrices/shallow_water1 -ksp_view -ksp_monitor_short -ksp_max_it 100 -solve_normal -pc_type none

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
        suffix: 4c
        nsize: 4
        requires: hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
        filter: grep -v "shared subdomain KSP between SLEPc and PETSc" | grep -v "total: nonzeros="
        args: -ksp_converged_reason -ksp_rtol 1e-5 -ksp_max_it 100 -ksp_view
        args: -ksp_type lsqr -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_1_st_share_sub_ksp {{false true}shared output}
        args: -pc_hpddm_levels_1_pc_asm_sub_mat_type aij -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_sub_pc_type cholesky
     test:
        suffix: 4d
        nsize: 4
        requires: hpddm slepc suitesparse defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
        filter: grep -v "shared subdomain KSP between SLEPc and PETSc"
        args: -ksp_converged_reason -ksp_rtol 1e-5 -ksp_max_it 100 -ksp_view
        args: -ksp_type lsqr -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_1_st_share_sub_ksp {{false true}shared output} -pc_hpddm_levels_1_st_pc_type qr
        args: -pc_hpddm_levels_1_pc_asm_sub_mat_type normalh -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_sub_pc_type qr

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

   test:
     suffix: 10
     requires: !complex double suitesparse !defined(PETSC_USE_64BIT_INDICES)
     nsize: 2
     args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -pc_type bjacobi -sub_pc_type qr

   test:
     suffix: 11
     nsize: 4
     requires: datafilespath double complex !defined(PETSC_USE_64BIT_INDICES) hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
     args: -f ${DATAFILESPATH}/matrices/farzad_B_rhs -truncate
     args: -ksp_converged_reason -ksp_rtol 1e-5 -ksp_max_it 100
     args: -ksp_type lsqr -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_1_eps_threshold 1e-6
     args: -pc_hpddm_levels_1_pc_asm_sub_mat_type aij -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_coarse_pc_type lu

TEST*/
