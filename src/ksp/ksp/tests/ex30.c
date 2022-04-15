
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
  PetscInt       its,num_numfac;
  PetscReal      rnorm,enorm;
  PetscBool      preload=PETSC_TRUE,diagonalscale,isSymmetric,ckrnorm=PETSC_TRUE,Test_MatDuplicate=PETSC_FALSE,ckerror=PETSC_FALSE;
  PetscMPIInt    rank;
  PetscScalar    sigma;
  PetscInt       m;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-initialguess",&initialguess,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-output_solution",&outputSoln,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-ckrnorm",&ckrnorm,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-ckerror",&ckerror,NULL));

  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg));
  if (flg) {
    PetscCall(PetscStrcpy(file[1],file[0]));
    preload = PETSC_FALSE;
  } else {
    PetscCall(PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
    PetscCall(PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg));
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
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&fd));

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetString(NULL,NULL,"-rhs",file[2],sizeof(file[2]),&flg));
  if (flg) {   /* rhs is stored in a separate file */
    PetscCall(PetscViewerDestroy(&fd));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd));
    PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
    PetscCall(VecLoad(b,fd));
  } else {
    /* if file contains no RHS, then use a vector of all ones */
    PetscCall(PetscInfo(0,"Using vector of ones for RHS\n"));
    PetscCall(MatGetLocalSize(A,&m,NULL));
    PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b,m,PETSC_DECIDE));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSet(b,1.0));
    PetscCall(PetscObjectSetName((PetscObject)b, "Rhs vector"));
  }
  PetscCall(PetscViewerDestroy(&fd));

  /* Test MatDuplicate() */
  if (Test_MatDuplicate) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
    PetscCall(MatEqual(A,B,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  A != B \n"));
    }
    PetscCall(MatDestroy(&B));
  }

  /* Add a shift to A */
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-mat_sigma",&sigma,&flg));
  if (flg) {
    PetscCall(PetscOptionsGetString(NULL,NULL,"-fB",file[2],sizeof(file[2]),&flgB));
    if (flgB) {
      /* load B to get A = A + sigma*B */
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd));
      PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
      PetscCall(MatSetOptionsPrefix(B,"B_"));
      PetscCall(MatLoad(B,fd));
      PetscCall(PetscViewerDestroy(&fd));
      PetscCall(MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN));   /* A <- sigma*B + A */
    } else {
      PetscCall(MatShift(A,sigma));
    }
  }

  /* Make A singular for testing zero-pivot of ilu factorization        */
  /* Example: ./ex30 -f0 <datafile> -test_zeropivot -set_row_zero -pc_factor_shift_nonzero */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-test_zeropivot", &flg,NULL));
  if (flg) {
    PetscInt          row,ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscBool         flg1=PETSC_FALSE;
    PetscScalar       *zeros;
    row  = 0;
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    PetscCall(PetscCalloc1(ncols+1,&zeros));
    flg1 = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL, "-set_row_zero", &flg1,NULL));
    if (flg1) {   /* set entire row as zero */
      PetscCall(MatSetValues(A,1,&row,ncols,cols,zeros,INSERT_VALUES));
    } else {   /* only set (row,row) entry as zero */
      PetscCall(MatSetValues(A,1,&row,1,&row,zeros,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }

  /* Check whether A is symmetric */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-check_symmetry", &flg,NULL));
  if (flg) {
    Mat Atrans;
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX,&Atrans));
    PetscCall(MatEqual(A, Atrans, &isSymmetric));
    if (isSymmetric) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"A is symmetric \n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"A is non-symmetric \n"));
    }
    PetscCall(MatDestroy(&Atrans));
  }

  PetscCall(VecDuplicate(b,&b2));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution vector"));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(PetscObjectSetName((PetscObject)u, "True Solution vector"));
  PetscCall(VecSet(x,0.0));

  if (ckerror) {   /* Set true solution */
    PetscCall(VecSet(u,1.0));
    PetscCall(MatMult(A,u,b));
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
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    PetscCall(PetscMalloc1(size,&count));
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &mpart));
    PetscCall(MatPartitioningSetAdjacency(mpart, A));
    /* PetscCall(MatPartitioningSetVertexWeights(mpart, weight)); */
    PetscCall(MatPartitioningSetFromOptions(mpart));
    PetscCall(MatPartitioningApply(mpart, &mis));
    PetscCall(MatPartitioningDestroy(&mpart));
    PetscCall(ISPartitioningToNumbering(mis,&nis));
    PetscCall(ISPartitioningCount(mis,size,count));
    PetscCall(ISDestroy(&mis));
    PetscCall(ISInvertPermutation(nis, count[rank], &is));
    PetscCall(PetscFree(count));
    PetscCall(ISDestroy(&nis));
    PetscCall(ISSort(is));
    PetscCall(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB));

    /* need to move the vector also */
    PetscCall(ISDestroy(&is));
    PetscCall(MatDestroy(&A));
    A    = BB;
  }

  /*
     Create linear solver; set operators; set runtime options.
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp,initialguess));
  num_numfac = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL));
  while (num_numfac--) {

    PetscCall(KSPSetOperators(ksp,A,A));
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
     Tests "diagonal-scaling of preconditioned residual norm" as used
     by many ODE integrator codes including SUNDIALS. Note this is different
     than diagonally scaling the matrix before computing the preconditioner
    */
    diagonalscale = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-diagonal_scale",&diagonalscale,NULL));
    if (diagonalscale) {
      PC       pc;
      PetscInt j,start,end,n;
      Vec      scale;

      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(VecGetSize(x,&n));
      PetscCall(VecDuplicate(x,&scale));
      PetscCall(VecGetOwnershipRange(scale,&start,&end));
      for (j=start; j<end; j++) {
        PetscCall(VecSetValue(scale,j,((PetscReal)(j+1))/((PetscReal)n),INSERT_VALUES));
      }
      PetscCall(VecAssemblyBegin(scale));
      PetscCall(VecAssemblyEnd(scale));
      PetscCall(PCSetDiagonalScale(pc,scale));
      PetscCall(VecDestroy(&scale));
    }

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Solve system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
     Solve linear system;
    */
    if (trans) {
      PetscCall(KSPSolveTranspose(ksp,b,x));
      PetscCall(KSPGetIterationNumber(ksp,&its));
    } else {
      PetscInt num_rhs=1;
      PetscCall(PetscOptionsGetInt(NULL,NULL,"-num_rhs",&num_rhs,NULL));

      while (num_rhs--) {
        PetscCall(KSPSolve(ksp,b,x));
      }
      PetscCall(KSPGetIterationNumber(ksp,&its));
      if (ckrnorm) {     /* Check residual for each rhs */
        if (trans) {
          PetscCall(MatMultTranspose(A,x,b2));
        } else {
          PetscCall(MatMult(A,x,b2));
        }
        PetscCall(VecAXPY(b2,-1.0,b));
        PetscCall(VecNorm(b2,NORM_2,&rnorm));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3" PetscInt_FMT "\n",its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)rnorm));
      }
      if (ckerror && !trans) {    /* Check error for each rhs */
        /* PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD)); */
        PetscCall(VecAXPY(u,-1.0,x));
        PetscCall(VecNorm(u,NORM_2,&enorm));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g\n",(double)enorm));
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
      PetscCall(PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer));
      PetscCall(KSPView(ksp,viewer));
      PetscCall(PetscStrrchr(file[PetscPreLoadIt],'/',&matrixname));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3" PetscInt_FMT " %2.0e %s \n", matrixname,its,(double)rnorm,kspinfo));

      /*
        Destroy the viewer
      */
      PetscCall(PetscViewerDestroy(&viewer));
    }

    PetscCall(PetscOptionsGetString(NULL,NULL,"-solution",file[3],sizeof(file[3]),&flg));
    if (flg) {
      PetscViewer viewer;
      Vec         xstar;

      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer));
      PetscCall(VecCreate(PETSC_COMM_WORLD,&xstar));
      PetscCall(VecLoad(xstar,viewer));
      PetscCall(VecAXPY(xstar, -1.0, x));
      PetscCall(VecNorm(xstar, NORM_2, &enorm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error norm %g\n", (double)enorm));
      PetscCall(VecDestroy(&xstar));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    if (outputSoln) {
      PetscViewer viewer;

      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution.petsc",FILE_MODE_WRITE,&viewer));
      PetscCall(VecView(x, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }

    flg  = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL));
    if (flg) {
      KSPConvergedReason reason;
      PetscCall(KSPGetConvergedReason(ksp,&reason));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %s\n", KSPConvergedReasons[reason]));
    }

  }   /* while (num_numfac--) */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u)); PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b2));
  PetscCall(KSPDestroy(&ksp));
  if (flgB) PetscCall(MatDestroy(&B));
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  PetscCall(PetscFinalize());
  return 0;
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
