
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
It is copied and intended to move dirty codes from ksp/tutorials/ex10.c and simplify ex10.c.\n\
  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n\
  -trans  : solve transpose system instead\n\n";
/*
  This code  can be used to test PETSc interface to other packages.\n\
  Examples of command line options:       \n\
   ex30 -f0 <datafile> -ksp_type preonly  \n\
        -help -ksp_view                  \n\
        -num_numfac <num_numfac> -num_rhs <num_rhs> \n\
        -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu or superlu_dist or mumps \n\
        -ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_type mumps \n\
   mpiexec -n <np> ex30 -f0 <datafile> -ksp_type cg -pc_type asm -pc_asm_type basic -sub_pc_type icc -mat_type sbaij

   ./ex30 -f0 $D/small -mat_sigma -3.999999999999999 -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type superlu -mat_superlu_conditionnumber -ckerror -mat_superlu_diagpivotthresh 0
   ./ex30 -f0 $D/small -mat_sigma -3.999999999999999 -ksp_type fgmres -pc_type hypre -pc_hypre_type boomeramg -ksp_type fgmres -ckError
   ./ex30 -f0 $D/small -mat_sigma -3.999999999999999 -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type petsc -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 -ckerror
 \n\n";
*/
/*T
   Concepts: KSP solving a linear system
   Processors: n
T*/

#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;
  Mat            A,B;
  Vec            x,b,u,b2;        /* approx solution, RHS, exact solution */
  PetscViewer    fd;              /* viewer */
  char           file[4][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      table     = PETSC_FALSE,flg,flgB=PETSC_FALSE,trans=PETSC_FALSE,partition=PETSC_FALSE,initialguess = PETSC_FALSE;
  PetscBool      outputSoln=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       its,num_numfac;
  PetscReal      rnorm,enorm;
  PetscBool      preload=PETSC_TRUE,diagonalscale,isSymmetric,ckrnorm=PETSC_TRUE,Test_MatDuplicate=PETSC_FALSE,ckerror=PETSC_FALSE;
  PetscMPIInt    rank;
  PetscScalar    sigma;
  PetscInt       m;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-initialguess",&initialguess,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-output_solution",&outputSoln,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ckrnorm",&ckrnorm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ckerror",&ckerror,NULL));

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
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&fd));

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,fd));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-rhs",file[2],sizeof(file[2]),&flg));
  if (flg) {   /* rhs is stored in a separate file */
    CHKERRQ(PetscViewerDestroy(&fd));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecLoad(b,fd));
  } else {
    /* if file contains no RHS, then use a vector of all ones */
    CHKERRQ(PetscInfo(0,"Using vector of ones for RHS\n"));
    CHKERRQ(MatGetLocalSize(A,&m,NULL));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecSetSizes(b,m,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(b));
    CHKERRQ(VecSet(b,1.0));
    CHKERRQ(PetscObjectSetName((PetscObject)b, "Rhs vector"));
  }
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Test MatDuplicate() */
  if (Test_MatDuplicate) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
    CHKERRQ(MatEqual(A,B,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  A != B \n"));
    }
    CHKERRQ(MatDestroy(&B));
  }

  /* Add a shift to A */
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-mat_sigma",&sigma,&flg));
  if (flg) {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fB",file[2],sizeof(file[2]),&flgB));
    if (flgB) {
      /* load B to get A = A + sigma*B */
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd));
      CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
      CHKERRQ(MatSetOptionsPrefix(B,"B_"));
      CHKERRQ(MatLoad(B,fd));
      CHKERRQ(PetscViewerDestroy(&fd));
      CHKERRQ(MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN));   /* A <- sigma*B + A */
    } else {
      CHKERRQ(MatShift(A,sigma));
    }
  }

  /* Make A singular for testing zero-pivot of ilu factorization        */
  /* Example: ./ex30 -f0 <datafile> -test_zeropivot -set_row_zero -pc_factor_shift_nonzero */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-test_zeropivot", &flg,NULL));
  if (flg) {
    PetscInt          row,ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscBool         flg1=PETSC_FALSE;
    PetscScalar       *zeros;
    row  = 0;
    CHKERRQ(MatGetRow(A,row,&ncols,&cols,&vals));
    CHKERRQ(PetscCalloc1(ncols+1,&zeros));
    flg1 = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-set_row_zero", &flg1,NULL));
    if (flg1) {   /* set entire row as zero */
      CHKERRQ(MatSetValues(A,1,&row,ncols,cols,zeros,INSERT_VALUES));
    } else {   /* only set (row,row) entry as zero */
      CHKERRQ(MatSetValues(A,1,&row,1,&row,zeros,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }

  /* Check whether A is symmetric */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-check_symmetry", &flg,NULL));
  if (flg) {
    Mat Atrans;
    CHKERRQ(MatTranspose(A, MAT_INITIAL_MATRIX,&Atrans));
    CHKERRQ(MatEqual(A, Atrans, &isSymmetric));
    if (isSymmetric) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"A is symmetric \n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"A is non-symmetric \n"));
    }
    CHKERRQ(MatDestroy(&Atrans));
  }

  CHKERRQ(VecDuplicate(b,&b2));
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(PetscObjectSetName((PetscObject)x, "Solution vector"));
  CHKERRQ(VecDuplicate(b,&u));
  CHKERRQ(PetscObjectSetName((PetscObject)u, "True Solution vector"));
  CHKERRQ(VecSet(x,0.0));

  if (ckerror) {   /* Set true solution */
    CHKERRQ(VecSet(u,1.0));
    CHKERRQ(MatMult(A,u,b));
  }

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                    Setup solve for system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (partition) {
    MatPartitioning mpart;
    IS              mis,nis,is;
    PetscInt        *count;
    PetscMPIInt     size;
    Mat             BB;
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    CHKERRQ(PetscMalloc1(size,&count));
    CHKERRQ(MatPartitioningCreate(PETSC_COMM_WORLD, &mpart));
    CHKERRQ(MatPartitioningSetAdjacency(mpart, A));
    /* CHKERRQ(MatPartitioningSetVertexWeights(mpart, weight)); */
    CHKERRQ(MatPartitioningSetFromOptions(mpart));
    CHKERRQ(MatPartitioningApply(mpart, &mis));
    CHKERRQ(MatPartitioningDestroy(&mpart));
    CHKERRQ(ISPartitioningToNumbering(mis,&nis));
    CHKERRQ(ISPartitioningCount(mis,size,count));
    CHKERRQ(ISDestroy(&mis));
    CHKERRQ(ISInvertPermutation(nis, count[rank], &is));
    CHKERRQ(PetscFree(count));
    CHKERRQ(ISDestroy(&nis));
    CHKERRQ(ISSort(is));
    CHKERRQ(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB));

    /* need to move the vector also */
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(MatDestroy(&A));
    A    = BB;
  }

  /*
     Create linear solver; set operators; set runtime options.
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,initialguess));
  num_numfac = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL));
  while (num_numfac--) {

    CHKERRQ(KSPSetOperators(ksp,A,A));
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
     Tests "diagonal-scaling of preconditioned residual norm" as used
     by many ODE integrator codes including SUNDIALS. Note this is different
     than diagonally scaling the matrix before computing the preconditioner
    */
    diagonalscale = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-diagonal_scale",&diagonalscale,NULL));
    if (diagonalscale) {
      PC       pc;
      PetscInt j,start,end,n;
      Vec      scale;

      CHKERRQ(KSPGetPC(ksp,&pc));
      CHKERRQ(VecGetSize(x,&n));
      CHKERRQ(VecDuplicate(x,&scale));
      CHKERRQ(VecGetOwnershipRange(scale,&start,&end));
      for (j=start; j<end; j++) {
        CHKERRQ(VecSetValue(scale,j,((PetscReal)(j+1))/((PetscReal)n),INSERT_VALUES));
      }
      CHKERRQ(VecAssemblyBegin(scale));
      CHKERRQ(VecAssemblyEnd(scale));
      CHKERRQ(PCSetDiagonalScale(pc,scale));
      CHKERRQ(VecDestroy(&scale));
    }

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

      while (num_rhs--) {
        CHKERRQ(KSPSolve(ksp,b,x));
      }
      CHKERRQ(KSPGetIterationNumber(ksp,&its));
      if (ckrnorm) {     /* Check residual for each rhs */
        if (trans) {
          CHKERRQ(MatMultTranspose(A,x,b2));
        } else {
          CHKERRQ(MatMult(A,x,b2));
        }
        CHKERRQ(VecAXPY(b2,-1.0,b));
        CHKERRQ(VecNorm(b2,NORM_2,&rnorm));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)rnorm));
      }
      if (ckerror && !trans) {    /* Check error for each rhs */
        /* CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD)); */
        CHKERRQ(VecAXPY(u,-1.0,x));
        CHKERRQ(VecNorm(u,NORM_2,&enorm));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g\n",(double)enorm));
      }

    }   /* while (num_rhs--) */

    /*
     Write output (optinally using table for solver details).
      - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
      - KSPView() prints information about the linear solver.
    */
    if (table && ckrnorm) {
      char        *matrixname,kspinfo[120];
      PetscViewer viewer;

      /*
        Open a string viewer; then write info to it.
      */
      CHKERRQ(PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer));
      CHKERRQ(KSPView(ksp,viewer));
      CHKERRQ(PetscStrrchr(file[PetscPreLoadIt],'/',&matrixname));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n", matrixname,its,rnorm,kspinfo));

      /*
        Destroy the viewer
      */
      CHKERRQ(PetscViewerDestroy(&viewer));
    }

    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-solution",file[3],sizeof(file[3]),&flg));
    if (flg) {
      PetscViewer viewer;
      Vec         xstar;

      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer));
      CHKERRQ(VecCreate(PETSC_COMM_WORLD,&xstar));
      CHKERRQ(VecLoad(xstar,viewer));
      CHKERRQ(VecAXPY(xstar, -1.0, x));
      CHKERRQ(VecNorm(xstar, NORM_2, &enorm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Error norm %g\n", (double)enorm));
      CHKERRQ(VecDestroy(&xstar));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
    if (outputSoln) {
      PetscViewer viewer;

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
  CHKERRQ(VecDestroy(&b2));
  CHKERRQ(KSPDestroy(&ksp));
  if (flgB) CHKERRQ(MatDestroy(&B));
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type ilu -pc_factor_mat_ordering_type natural -num_numfac 2 -pc_factor_reuse_fill
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30.out

    test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 2
      args: -f ${DATAFILESPATH}/matrices/arco1 -mat_type baij -matload_block_size 3 -ksp_type preonly -pc_type ilu -pc_factor_mat_ordering_type natural -num_numfac 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftnz
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftpd
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type POSITIVE_DEFINITE
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shift_cholesky_aij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftnz.out

    test:
      suffix: shiftpd_2
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shift_cholesky_sbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 -mat_type sbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftnz.out

    test:
      suffix: shiftpd_2_sbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE -mat_type sbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftpd_2.out

    test:
      suffix: shiftinblocks
      args:  -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type INBLOCKS
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftinblocks2
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftinblocks.out

    test:
      suffix: shiftinblockssbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS -mat_type sbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftinblocks.out

TEST*/
