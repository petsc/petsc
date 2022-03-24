
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n\
  -nearnulldim <0> : number of vectors in the near-null space immediately following matrix\n\n\
  -trans  : solve transpose system instead\n\n";
/*
  This code can be used to test PETSc interface to other packages.\n\
  Examples of command line options:       \n\
   ./ex72 -f0 <datafile> -ksp_type preonly  \n\
        -help -ksp_view                  \n\
        -num_numfac <num_numfac> -num_rhs <num_rhs> \n\
        -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu or superlu_dist or mumps \n\
        -ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_type mumps \n\
   mpiexec -n <np> ./ex72 -f0 <datafile> -ksp_type cg -pc_type asm -pc_asm_type basic -sub_pc_type icc -mat_type sbaij
 \n\n";
*/
/*T
   Concepts: KSP^solving a linear system
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
  Mat            A;               /* matrix */
  Vec            x,b,u;           /* approx solution, RHS, exact solution */
  PetscViewer    viewer;          /* viewer */
  char           file[4][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      table     =PETSC_FALSE,flg,trans=PETSC_FALSE,initialguess = PETSC_FALSE;
  PetscBool      outputSoln=PETSC_FALSE,constantnullspace = PETSC_FALSE;
  PetscInt       its,num_numfac,m,n,M,p,nearnulldim = 0;
  PetscReal      norm;
  PetscBool      preload=PETSC_TRUE,isSymmetric,cknorm=PETSC_FALSE,initialguessfile = PETSC_FALSE;
  PetscMPIInt    rank;
  char           initialguessfilename[PETSC_MAX_PATH_LEN];

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-constantnullspace",&constantnullspace,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-initialguess",&initialguess,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-output_solution",&outputSoln,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-initialguessfilename",initialguessfilename,sizeof(initialguessfilename),&initialguessfile));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nearnulldim",&nearnulldim,NULL));

  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg));
  if (flg) {
    CHKERRQ(PetscStrcpy(file[1],file[0]));
    preload = PETSC_FALSE;
  } else {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg));
    if (!flg) preload = PETSC_FALSE;   /* don't bother with second system */
  }

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
  PetscPreLoadBegin(preload,"Load system");

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Load system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&viewer));

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  if (nearnulldim) {
    MatNullSpace nullsp;
    Vec          *nullvecs;
    PetscInt     i;
    CHKERRQ(PetscMalloc1(nearnulldim,&nullvecs));
    for (i=0; i<nearnulldim; i++) {
      CHKERRQ(VecCreate(PETSC_COMM_WORLD,&nullvecs[i]));
      CHKERRQ(VecLoad(nullvecs[i],viewer));
    }
    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,nearnulldim,nullvecs,&nullsp));
    CHKERRQ(MatSetNearNullSpace(A,nullsp));
    for (i=0; i<nearnulldim; i++) CHKERRQ(VecDestroy(&nullvecs[i]));
    CHKERRQ(PetscFree(nullvecs));
    CHKERRQ(MatNullSpaceDestroy(&nullsp));
  }
  if (constantnullspace) {
    MatNullSpace constant;
    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&constant));
    CHKERRQ(MatSetNullSpace(A,constant));
    CHKERRQ(MatNullSpaceDestroy(&constant));
  }
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-rhs",file[2],sizeof(file[2]),&flg));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  if (flg) {   /* rhs is stored in a separate file */
    if (file[2][0] == '0' || file[2][0] == 0) {
      PetscInt    m;
      PetscScalar one = 1.0;
      CHKERRQ(PetscInfo(0,"Using vector of ones for RHS\n"));
      CHKERRQ(MatGetLocalSize(A,&m,NULL));
      CHKERRQ(VecSetSizes(b,m,PETSC_DECIDE));
      CHKERRQ(VecSetFromOptions(b));
      CHKERRQ(VecSet(b,one));
    } else {
      CHKERRQ(PetscViewerDestroy(&viewer));
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&viewer));
      CHKERRQ(VecSetFromOptions(b));
      CHKERRQ(VecLoad(b,viewer));
    }
  } else {   /* rhs is stored in the same file as matrix */
    CHKERRQ(VecSetFromOptions(b));
    CHKERRQ(VecLoad(b,viewer));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* Make A singular for testing zero-pivot of ilu factorization */
  /* Example: ./ex72 -f0 <datafile> -test_zeropivot -pc_factor_shift_type <shift_type> */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-test_zeropivot", &flg,NULL));
  if (flg) { /* set a row as zeros */
    PetscInt          row=0;
    CHKERRQ(MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
    CHKERRQ(MatZeroRows(A,1,&row,0.0,NULL,NULL));
  }

  /* Check whether A is symmetric, then set A->symmetric option */
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-check_symmetry", &flg,NULL));
  if (flg) {
    CHKERRQ(MatIsSymmetric(A,0.0,&isSymmetric));
    if (!isSymmetric) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: A is non-symmetric \n"));
    }
  }

  /*
     If the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector.
  */

  CHKERRQ(MatGetLocalSize(A,NULL,&n));
  CHKERRQ(MatGetSize(A,&M,NULL));
  CHKERRQ(VecGetSize(b,&m));
  CHKERRQ(VecGetLocalSize(b,&p));
  preload = (PetscBool)(M != m || p != n); /* Global or local dimension mismatch */
  CHKERRMPI(MPIU_Allreduce(&preload,&flg,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A)));
  if (flg) { /* Create a new vector b by padding the old one */
    PetscInt    j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&tmp));
    CHKERRQ(VecSetSizes(tmp,n,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(tmp));
    CHKERRQ(VecGetOwnershipRange(b,&start,&end));
    CHKERRQ(VecGetLocalSize(b,&mvec));
    CHKERRQ(VecGetArray(b,&bold));
    for (j=0; j<mvec; j++) {
      indx = start+j;
      CHKERRQ(VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArray(b,&bold));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecAssemblyBegin(tmp));
    CHKERRQ(VecAssemblyEnd(tmp));
    b    = tmp;
  }

  CHKERRQ(MatCreateVecs(A,&x,NULL));
  CHKERRQ(VecDuplicate(b,&u));
  if (initialguessfile) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,initialguessfilename,FILE_MODE_READ,&viewer));
    CHKERRQ(VecLoad(x,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    initialguess = PETSC_TRUE;
  } else if (initialguess) {
    CHKERRQ(VecSet(x,1.0));
  } else {
    CHKERRQ(VecSet(x,0.0));
  }

  /* Check scaling in A */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-check_scaling", &flg,NULL));
  if (flg) {
    Vec       max, min;
    PetscInt  idx;
    PetscReal val;

    CHKERRQ(VecDuplicate(x, &max));
    CHKERRQ(VecDuplicate(x, &min));
    CHKERRQ(MatGetRowMaxAbs(A, max, NULL));
    CHKERRQ(MatGetRowMinAbs(A, min, NULL));
    {
      CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "max.data", &viewer));
      CHKERRQ(VecView(max, viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
      CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "min.data", &viewer));
      CHKERRQ(VecView(min, viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
    CHKERRQ(VecView(max, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(VecMax(max, &idx, &val));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Largest max row element %g at row %D\n", (double)val, idx));
    CHKERRQ(VecView(min, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(VecMin(min, &idx, &val));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Smallest min row element %g at row %D\n", (double)val, idx));
    CHKERRQ(VecMin(max, &idx, &val));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Smallest max row element %g at row %D\n", (double)val, idx));
    CHKERRQ(VecPointwiseDivide(max, max, min));
    CHKERRQ(VecMax(max, &idx, &val));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Largest row ratio %g at row %D\n", (double)val, idx));
    CHKERRQ(VecView(max, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(VecDestroy(&max));
    CHKERRQ(VecDestroy(&min));
  }

  /*  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD)); */
  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Conclude profiling last stage; begin profiling next stage.
  */
  PetscPreLoadStage("KSPSetUpSolve");

  /*
     Create linear solver; set operators; set runtime options.
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,initialguess));
  num_numfac = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL));
  while (num_numfac--) {
    PC        pc;
    PetscBool lsqr,isbddc,ismatis;
    char      str[32];

    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-ksp_type",str,sizeof(str),&lsqr));
    if (lsqr) {
      CHKERRQ(PetscStrcmp("lsqr",str,&lsqr));
    }
    if (lsqr) {
      Mat BtB;
      CHKERRQ(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,4,&BtB));
      CHKERRQ(KSPSetOperators(ksp,A,BtB));
      CHKERRQ(MatDestroy(&BtB));
    } else {
      CHKERRQ(KSPSetOperators(ksp,A,A));
    }
    CHKERRQ(KSPSetFromOptions(ksp));

    /* if we test BDDC, make sure pmat is of type MATIS */
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));
    if (isbddc && !ismatis) {
      Mat J;

      CHKERRQ(MatConvert(A,MATIS,MAT_INITIAL_MATRIX,&J));
      CHKERRQ(KSPSetOperators(ksp,A,J));
      CHKERRQ(MatDestroy(&J));
    }

    /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
    */
    CHKERRQ(KSPSetUp(ksp));
    CHKERRQ(KSPSetUpOnBlocks(ksp));

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Solve system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
     Solve linear system;
    */
    if (trans) {
      CHKERRQ(KSPSolveTranspose(ksp,b,x));
      CHKERRQ(KSPGetIterationNumber(ksp,&its));
    } else {
      PetscInt num_rhs=1;
      CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-num_rhs",&num_rhs,NULL));
      cknorm = PETSC_FALSE;
      CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-cknorm",&cknorm,NULL));
      while (num_rhs--) {
        if (num_rhs == 1) VecSet(x,0.0);
        CHKERRQ(KSPSolve(ksp,b,x));
      }
      CHKERRQ(KSPGetIterationNumber(ksp,&its));
      if (cknorm) {     /* Check error for each rhs */
        if (trans) {
          CHKERRQ(MatMultTranspose(A,x,u));
        } else {
          CHKERRQ(MatMult(A,x,u));
        }
        CHKERRQ(VecAXPY(u,-1.0,b));
        CHKERRQ(VecNorm(u,NORM_2,&norm));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its));
        if (!PetscIsNanScalar(norm)) {
          if (norm < 1.e-12) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual norm < 1.e-12\n"));
          } else {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)norm));
          }
        }
      }
    }   /* while (num_rhs--) */

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
          Check error, print output, free data structures.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Check error
    */
    if (trans) {
      CHKERRQ(MatMultTranspose(A,x,u));
    } else {
      CHKERRQ(MatMult(A,x,u));
    }
    CHKERRQ(VecAXPY(u,-1.0,b));
    CHKERRQ(VecNorm(u,NORM_2,&norm));
    /*
     Write output (optinally using table for solver details).
      - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
      - KSPView() prints information about the linear solver.
    */
    if (table) {
      char        *matrixname,kspinfo[120];

      /*
       Open a string viewer; then write info to it.
      */
      CHKERRQ(PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer));
      CHKERRQ(KSPView(ksp,viewer));
      CHKERRQ(PetscStrrchr(file[PetscPreLoadIt],'/',&matrixname));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n",matrixname,its,norm,kspinfo));

      /*
        Destroy the viewer
      */
      CHKERRQ(PetscViewerDestroy(&viewer));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
      if (!PetscIsNanScalar(norm)) {
        if (norm < 1.e-12 && !PetscIsNanScalar((PetscScalar)norm)) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual norm < 1.e-12\n"));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));
        }
      }
    }
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-solution",file[3],sizeof(file[3]),&flg));
    if (flg) {
      Vec         xstar;
      PetscReal   norm;

      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer));
      CHKERRQ(VecCreate(PETSC_COMM_WORLD,&xstar));
      CHKERRQ(VecLoad(xstar,viewer));
      CHKERRQ(VecAXPY(xstar, -1.0, x));
      CHKERRQ(VecNorm(xstar, NORM_2, &norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Error norm %g\n", (double)norm));
      CHKERRQ(VecDestroy(&xstar));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
    if (outputSoln) {
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution.petsc",FILE_MODE_WRITE,&viewer));
      CHKERRQ(VecView(x, viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }

    flg  = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL));
    if (flg) {
      KSPConvergedReason reason;
      CHKERRQ(KSPGetConvergedReason(ksp,&reason));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason));
    }

  }   /* while (num_numfac--) */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(MatDestroy(&A)); CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   testset:
      suffix: 1
      nsize: 2
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int@PETSC_INDEX_SIZE@-float@PETSC_SCALAR_SIZE@
      requires: !__float128

   testset:
      suffix: 1a
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int@PETSC_INDEX_SIZE@-float@PETSC_SCALAR_SIZE@
      requires: !__float128

   testset:
      nsize: 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args:  -ksp_type bicg
      test:
         suffix: 2

   testset:
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args: -ksp_type bicg
      test:
         suffix: 4
         args: -pc_type lu
      test:
         suffix: 5

   testset:
      suffix: 6
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/fem1
      args: -pc_factor_levels 2 -pc_factor_fill 1.73 -ksp_gmres_cgs_refinement_type refine_always

   testset:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 7
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args: -viewer_binary_skip_info -mat_type seqbaij
      args: -matload_block_size {{2 3 4 5 6 7 8}separate output}
      args: -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always
      args: -ksp_rtol 1.0e-15 -ksp_monitor_short
      test:
         suffix: a
      test:
         suffix: b
         args: -pc_factor_mat_ordering_type nd
      test:
         suffix: c
         args: -pc_factor_levels 1
      test:
         requires: metis
         suffix: d
         args: -pc_factor_mat_ordering_type metisnd

   testset:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 7_d
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args: -viewer_binary_skip_info -mat_type seqbaij
      args: -matload_block_size {{2 3 4 5 6 7 8}shared output}
      args: -ksp_type preonly -pc_type lu

   testset:
      suffix: 8
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args: -ksp_diagonal_scale -pc_type eisenstat -ksp_monitor_short -ksp_diagonal_scale_fix -ksp_gmres_cgs_refinement_type refine_always -mat_no_inode

   testset:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 9
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium
      args: -viewer_binary_skip_info  -matload_block_size {{1 2 3 4 5 6 7}separate output} -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol 1.0e-15 -ksp_monitor_short
      test:
         suffix: a
         args: -mat_type seqbaij
      test:
         suffix: b
         args: -mat_type seqbaij -trans
      test:
         suffix: c
         nsize: 2
         args: -mat_type mpibaij
      test:
         suffix: d
         nsize: 2
         args: -mat_type mpibaij -trans
      test:
         suffix: e
         nsize: 3
         args: -mat_type mpibaij
      test:
         suffix: f
         nsize: 3
         args: -mat_type mpibaij -trans

   testset:
      suffix: 10
      nsize: 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -ksp_type fgmres -pc_type ksp -f0 ${DATAFILESPATH}/matrices/medium -ksp_fgmres_modifypcksp -ksp_monitor_short

   testset:
      suffix: 12
      requires: matlab
      args: -pc_type lu -pc_factor_mat_solver_type matlab -f0 ${DATAFILESPATH}/matrices/arco1

   testset:
      suffix: 13
      requires: lusol
      args: -f0 ${DATAFILESPATH}/matrices/arco1
      args: -mat_type lusol -pc_type lu

   testset:
      nsize: 3
      args: -f0 ${DATAFILESPATH}/matrices/medium
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: 14
         requires: spai
         args: -pc_type spai
      test:
         suffix: 15
         requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
         args: -pc_type hypre -pc_hypre_type pilut
      test:
         suffix: 16
         requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
         args: -pc_type hypre -pc_hypre_type parasails
      test:
         suffix: 17
         requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
         args: -pc_type hypre -pc_hypre_type boomeramg
      test:
         suffix: 18
         requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
         args: -pc_type hypre -pc_hypre_type euclid

   testset:
      suffix: 19
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/poisson1
      args: -ksp_type cg -pc_type icc
      args: -pc_factor_levels {{0 2 4}separate output}
      test:
      test:
         args: -mat_type seqsbaij

   testset:
      suffix: ILU
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/small
      args: -pc_factor_levels 1
      test:
      test:
         # This is tested against regular ILU (used to be denoted ILUBAIJ)
         args: -mat_type baij

   testset:
      suffix: aijcusparse
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) cuda
      args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_monitor_short -ksp_view -mat_view ascii::ascii_info -mat_type aijcusparse -pc_factor_mat_solver_type cusparse -pc_type ilu -vec_type cuda

   testset:
      TODO: No output file. Need to determine if deprecated
      suffix: asm_viennacl
      nsize: 2
      requires: viennacl
      args: -pc_type asm -pc_asm_sub_mat_type aijviennacl -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int${PETSC_INDEX_SIZE}-float${PETSC_SCALAR_SIZE}

   testset:
      nsize: 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -f0 ${DATAFILESPATH}/matrices/poisson2.gz -ksp_monitor_short -ksp_rtol 1.E-9 -pc_type hypre -pc_hypre_type boomeramg
      test:
         suffix: boomeramg_euclid
         args: -pc_hypre_boomeramg_smooth_type Euclid -pc_hypre_boomeramg_smooth_num_levels 2 -pc_hypre_boomeramg_eu_level 1 -pc_hypre_boomeramg_eu_droptolerance 0.01
         TODO: Need to determine if deprecated
      test:
         suffix: boomeramg_euclid_bj
         args: -pc_hypre_boomeramg_smooth_type Euclid -pc_hypre_boomeramg_smooth_num_levels 2 -pc_hypre_boomeramg_eu_level 1 -pc_hypre_boomeramg_eu_droptolerance 0.01 -pc_hypre_boomeramg_eu_bj
         TODO: Need to determine if deprecated
      test:
         suffix: boomeramg_parasails
         args: -pc_hypre_boomeramg_smooth_type ParaSails -pc_hypre_boomeramg_smooth_num_levels 2
      test:
         suffix: boomeramg_pilut
         args: -pc_hypre_boomeramg_smooth_type Pilut -pc_hypre_boomeramg_smooth_num_levels 2
      test:
         suffix: boomeramg_schwarz
         args: -pc_hypre_boomeramg_smooth_type Schwarz-smoothers

   testset:
      suffix: cg_singlereduction
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/small
      args: -mat_type mpisbaij -ksp_type cg -pc_type eisenstat -ksp_monitor_short -ksp_converged_reason
      test:
      test:
         args: -ksp_cg_single_reduction

   testset:
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/poisson2.gz
      args: -ksp_monitor_short -pc_type icc
      test:
         suffix: cr
         args: -ksp_type cr
      test:
         suffix: lcd
         args: -ksp_type lcd

   testset:
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/small
      args: -ksp_monitor_short -ksp_view -mat_view ascii::ascii_info
      test:
         suffix: seqaijcrl
         args: -mat_type seqaijcrl
      test:
         suffix: seqaijperm
         args: -mat_type seqaijperm

   testset:
      nsize: 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/small
      args: -ksp_monitor_short -ksp_view
      # Different output files
      test:
         suffix: mpiaijcrl
         args: -mat_type mpiaijcrl
      test:
         suffix: mpiaijperm
         args: -mat_type mpiaijperm

   testset:
      nsize: 4
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !defined(PETSC_HAVE_I_MPI_NUMVERSION)
      args: -ksp_monitor_short -ksp_view
      test:
         suffix: xxt
         args: -f0 ${DATAFILESPATH}/matrices/poisson1 -check_symmetry -ksp_type cg -pc_type tfs
      test:
         suffix: xyt
         args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_type gmres -pc_type tfs

   testset:
      # The output file here is the same as mumps
      suffix: mumps_cholesky
      output_file: output/ex72_mumps.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_type mumps -num_numfac 2 -num_rhs 2
      nsize: {{1 2}}
      test:
         args: -mat_type sbaij -mat_ignore_lower_triangular
      test:
         args: -mat_type aij
      test:
         args: -mat_type aij -matload_spd

   testset:
      # The output file here is the same as mumps
      suffix: mumps_lu
      output_file: output/ex72_mumps.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -num_numfac 2 -num_rhs 2
      test:
         args: -mat_type seqaij
      test:
         nsize: 2
         args: -mat_type mpiaij
      test:
         args: -mat_type seqbaij -matload_block_size 2
      test:
         nsize: 2
         args: -mat_type mpibaij -matload_block_size 2
      test:
         args: -mat_type aij -mat_mumps_icntl_7 5
         TODO: Need to determine if deprecated

   test:
      suffix: mumps_lu_parmetis
      output_file: output/ex72_mumps.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps parmetis
      nsize: 2
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -num_numfac 2 -num_rhs 2 -mat_type mpiaij -mat_mumps_icntl_28 2 -mat_mumps_icntl_29 2

   test:
      suffix: mumps_lu_ptscotch
      output_file: output/ex72_mumps.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps ptscotch
      nsize: 2
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -num_numfac 2 -num_rhs 2 -mat_type mpiaij -mat_mumps_icntl_28 2 -mat_mumps_icntl_29 1

   testset:
      # The output file here is the same as mumps
      suffix: mumps_redundant
      output_file: output/ex72_mumps_redundant.out
      nsize: 8
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps
      args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_type preonly -pc_type redundant -pc_redundant_number {{8 7 6 5 4 3 2 1}} -redundant_pc_factor_mat_solver_type mumps -num_numfac 2 -num_rhs 2

   testset:
      suffix: pastix_cholesky
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) pastix
      output_file: output/ex72_mumps.out
      nsize: {{1 2}}
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_factor_mat_solver_type pastix -num_numfac 2 -num_rhs 2 -pc_type cholesky -mat_type sbaij -mat_ignore_lower_triangular

   testset:
      suffix: pastix_lu
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) pastix
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type pastix -num_numfac 2 -num_rhs 2
      output_file: output/ex72_mumps.out
      test:
         args: -mat_type seqaij
      test:
         nsize: 2
         args: -mat_type mpiaij

   testset:
      suffix: pastix_redundant
      output_file: output/ex72_mumps_redundant.out
      nsize: 8
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) pastix
      args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_type preonly -pc_type redundant -pc_redundant_number {{8 7 6 5 4 3 2 1}} -redundant_pc_factor_mat_solver_type pastix -num_numfac 2 -num_rhs 2

   testset:
      suffix: superlu_dist_lu
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) superlu_dist
      output_file: output/ex72_mumps.out
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -num_numfac 2 -num_rhs 2
      nsize: {{1 2}}

   testset:
      suffix: superlu_dist_redundant
      nsize: 8
      output_file: output/ex72_mumps_redundant.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) superlu_dist
      args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_type preonly -pc_type redundant -pc_redundant_number {{8 7 6 5 4 3 2 1}} -redundant_pc_factor_mat_solver_type superlu_dist -num_numfac 2 -num_rhs 2

   testset:
      suffix: superlu_lu
      output_file: output/ex72_mumps.out
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) superlu
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu -num_numfac 2 -num_rhs 2

   testset:
      suffix: umfpack
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) suitesparse
      args: -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -mat_type seqaij -pc_factor_mat_solver_type umfpack -num_numfac 2 -num_rhs 2

   testset:
      suffix: zeropivot
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps
      args: -f0 ${DATAFILESPATH}/matrices/small -test_zeropivot -ksp_converged_reason -ksp_type fgmres -pc_type ksp
      test:
         nsize: 3
         args: -ksp_pc_type bjacobi
      test:
         nsize: 2
         args: -ksp_ksp_type cg -ksp_pc_type bjacobi -ksp_pc_bjacobi_blocks 1
      #test:
         #nsize: 3
         #args: -ksp_ksp_converged_reason -ksp_pc_type bjacobi -ksp_sub_ksp_converged_reason
         #TODO: Need to determine if deprecated

   testset:
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -ksp_type fgmres
      test:
         suffix: bddc_seq
         nsize: 1
         args: -pc_type bddc
      test:
         suffix: bddc_par
         nsize: 2
         args: -pc_type bddc
      test:
         requires: parmetis
         suffix: bddc_par_nd_parmetis
         filter: sed -e "s/Number of iterations =   [0-9]/Number of iterations = 9/g"
         nsize: 4
         args: -ksp_error_if_not_converged -pc_type bddc -mat_is_disassemble_l2g_type nd -mat_partitioning_type parmetis
      test:
         requires: ptscotch defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
         suffix: bddc_par_nd_ptscotch
         filter: sed -e "s/Number of iterations =   [0-9]/Number of iterations = 9/g"
         nsize: 4
         args: -ksp_error_if_not_converged -pc_type bddc -mat_is_disassemble_l2g_type nd -mat_partitioning_type ptscotch

   testset:
      requires: !__float128 hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      test:
         suffix: hpddm_mat
         output_file: output/ex72_bddc_seq.out
         filter: sed -e "s/Number of iterations =   2/Number of iterations =   1/g"
         nsize: 2
         args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int@PETSC_INDEX_SIZE@-float@PETSC_SCALAR_SIZE@ -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_st_pc_type mat
      test:
         requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES)
         suffix: hpddm_gen_non_hermitian
         output_file: output/ex72_2.out
         nsize: 4
         args: -f0 ${DATAFILESPATH}/matrices/arco1 -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_1_st_share_sub_ksp -pc_hpddm_levels_1_eps_gen_non_hermitian -pc_hpddm_coarse_mat_type baij -pc_hpddm_block_splitting -pc_hpddm_levels_1_eps_threshold 0.7 -pc_hpddm_coarse_pc_type lu -ksp_pc_side right
      test:
         requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) mumps !defined(PETSCTEST_VALGRIND)
         suffix: hpddm_gen_non_hermitian_baij
         output_file: output/ex72_5.out
         nsize: 4
         timeoutfactor: 2
         args: -f0 ${DATAFILESPATH}/matrices/arco6 -pc_type hpddm -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 30 -pc_hpddm_levels_1_st_share_sub_ksp -pc_hpddm_levels_1_eps_gen_non_hermitian -pc_hpddm_coarse_mat_type baij -pc_hpddm_block_splitting -pc_hpddm_levels_1_eps_threshold 0.8 -pc_hpddm_coarse_pc_type lu -ksp_pc_side right -mat_type baij -pc_hpddm_levels_1_sub_pc_factor_mat_solver_type mumps -pc_hpddm_levels_1_eps_tol 1.0e-2
TEST*/
