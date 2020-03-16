
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
It is copied and intended to move dirty codes from ksp/examples/tutorials/ex10.c and simplify ex10.c.\n\
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-initialguess",&initialguess,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-output_solution",&outputSoln,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ckrnorm",&ckrnorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ckerror",&ckerror,NULL);CHKERRQ(ierr);

  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr    = PetscStrcpy(file[1],file[0]);CHKERRQ(ierr);
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(NULL,NULL,"-f0",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
    ierr = PetscOptionsGetString(NULL,NULL,"-f1",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
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
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetString(NULL,NULL,"-rhs",file[2],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {   /* rhs is stored in a separate file */
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
  } else {
    /* if file contains no RHS, then use a vector of all ones */
    ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)b, "Rhs vector");CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Test MatDuplicate() */
  if (Test_MatDuplicate) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatEqual(A,B,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  A != B \n");CHKERRQ(ierr);
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  /* Add a shift to A */
  ierr = PetscOptionsGetScalar(NULL,NULL,"-mat_sigma",&sigma,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsGetString(NULL,NULL,"-fB",file[2],PETSC_MAX_PATH_LEN,&flgB);CHKERRQ(ierr);
    if (flgB) {
      /* load B to get A = A + sigma*B */
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(B,"B_");CHKERRQ(ierr);
      ierr = MatLoad(B,fd);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
      ierr = MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);   /* A <- sigma*B + A */
    } else {
      ierr = MatShift(A,sigma);CHKERRQ(ierr);
    }
  }

  /* Make A singular for testing zero-pivot of ilu factorization        */
  /* Example: ./ex30 -f0 <datafile> -test_zeropivot -set_row_zero -pc_factor_shift_nonzero */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-test_zeropivot", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscInt          row,ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscBool         flg1=PETSC_FALSE;
    PetscScalar       *zeros;
    row  = 0;
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = PetscCalloc1(ncols+1,&zeros);CHKERRQ(ierr);
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL, "-set_row_zero", &flg1,NULL);CHKERRQ(ierr);
    if (flg1) {   /* set entire row as zero */
      ierr = MatSetValues(A,1,&row,ncols,cols,zeros,INSERT_VALUES);CHKERRQ(ierr);
    } else {   /* only set (row,row) entry as zero */
      ierr = MatSetValues(A,1,&row,1,&row,zeros,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Check whether A is symmetric */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-check_symmetry", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Mat Atrans;
    ierr = MatTranspose(A, MAT_INITIAL_MATRIX,&Atrans);CHKERRQ(ierr);
    ierr = MatEqual(A, Atrans, &isSymmetric);CHKERRQ(ierr);
    if (isSymmetric) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"A is symmetric \n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"A is non-symmetric \n");CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Atrans);CHKERRQ(ierr);
  }

  ierr = VecDuplicate(b,&b2);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x, "Solution vector");CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "True Solution vector");CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  if (ckerror) {   /* Set true solution */
    ierr = VecSet(u,1.0);CHKERRQ(ierr);
    ierr = MatMult(A,u,b);CHKERRQ(ierr);
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
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&count);CHKERRQ(ierr);
    ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &mpart);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(mpart, A);CHKERRQ(ierr);
    /* ierr = MatPartitioningSetVertexWeights(mpart, weight);CHKERRQ(ierr); */
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = MatPartitioningApply(mpart, &mis);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
    ierr = ISPartitioningToNumbering(mis,&nis);CHKERRQ(ierr);
    ierr = ISPartitioningCount(mis,size,count);CHKERRQ(ierr);
    ierr = ISDestroy(&mis);CHKERRQ(ierr);
    ierr = ISInvertPermutation(nis, count[rank], &is);CHKERRQ(ierr);
    ierr = PetscFree(count);CHKERRQ(ierr);
    ierr = ISDestroy(&nis);CHKERRQ(ierr);
    ierr = ISSort(is);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB);CHKERRQ(ierr);

    /* need to move the vector also */
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = BB;
  }

  /*
     Create linear solver; set operators; set runtime options.
  */
  ierr       = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr       = KSPSetInitialGuessNonzero(ksp,initialguess);CHKERRQ(ierr);
  num_numfac = 1;
  ierr       = PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL);CHKERRQ(ierr);
  while (num_numfac--) {

    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
    */
    ierr   = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr   = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

    /*
     Tests "diagonal-scaling of preconditioned residual norm" as used
     by many ODE integrator codes including SUNDIALS. Note this is different
     than diagonally scaling the matrix before computing the preconditioner
    */
    diagonalscale = PETSC_FALSE;
    ierr          = PetscOptionsGetBool(NULL,NULL,"-diagonal_scale",&diagonalscale,NULL);CHKERRQ(ierr);
    if (diagonalscale) {
      PC       pc;
      PetscInt j,start,end,n;
      Vec      scale;

      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = VecGetSize(x,&n);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&scale);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(scale,&start,&end);CHKERRQ(ierr);
      for (j=start; j<end; j++) {
        ierr = VecSetValue(scale,j,((PetscReal)(j+1))/((PetscReal)n),INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin(scale);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(scale);CHKERRQ(ierr);
      ierr = PCSetDiagonalScale(pc,scale);CHKERRQ(ierr);
      ierr = VecDestroy(&scale);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Solve system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
     Solve linear system; 
    */
    if (trans) {
      ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    } else {
      PetscInt num_rhs=1;
      ierr = PetscOptionsGetInt(NULL,NULL,"-num_rhs",&num_rhs,NULL);CHKERRQ(ierr);

      while (num_rhs--) {
        ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      }
      ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
      if (ckrnorm) {     /* Check residual for each rhs */
        if (trans) {
          ierr = MatMultTranspose(A,x,b2);CHKERRQ(ierr);
        } else {
          ierr = MatMult(A,x,b2);CHKERRQ(ierr);
        }
        ierr = VecAXPY(b2,-1.0,b);CHKERRQ(ierr);
        ierr = VecNorm(b2,NORM_2,&rnorm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)rnorm);CHKERRQ(ierr);
      }
      if (ckerror && !trans) {    /* Check error for each rhs */
        /* ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
        ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);
        ierr = VecNorm(u,NORM_2,&enorm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm %g\n",(double)enorm);CHKERRQ(ierr);
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
      ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer);CHKERRQ(ierr);
      ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
      ierr = PetscStrrchr(file[PetscPreLoadIt],'/',&matrixname);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n", matrixname,its,rnorm,kspinfo);CHKERRQ(ierr);

      /*
        Destroy the viewer
      */
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    ierr = PetscOptionsGetString(NULL,NULL,"-solution",file[3],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      Vec         xstar;

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&xstar);CHKERRQ(ierr);
      ierr = VecLoad(xstar,viewer);CHKERRQ(ierr);
      ierr = VecAXPY(xstar, -1.0, x);CHKERRQ(ierr);
      ierr = VecNorm(xstar, NORM_2, &enorm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Error norm %g\n", (double)enorm);CHKERRQ(ierr);
      ierr = VecDestroy(&xstar);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    if (outputSoln) {
      PetscViewer viewer;

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution.petsc",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(x, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL);CHKERRQ(ierr);
    if (flg) {
      KSPConvergedReason reason;
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);CHKERRQ(ierr);
    }

  }   /* while (num_numfac--) */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b2);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  if (flgB) { ierr = MatDestroy(&B);CHKERRQ(ierr); }
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
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30.out

    test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 2
      args: -f ${DATAFILESPATH}/matrices/arco1 -mat_type baij -matload_block_size 3 -ksp_type preonly -pc_type ilu -pc_factor_mat_ordering_type natural -num_numfac 2
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftnz
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftpd
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type POSITIVE_DEFINITE
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shift_cholesky_aij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftnz.out

    test:
      suffix: shiftpd_2
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shift_cholesky_sbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 -mat_type sbaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftnz.out

    test:
      suffix: shiftpd_2_sbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE -mat_type sbaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftpd_2.out

    test:
      suffix: shiftinblocks
      args:  -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type INBLOCKS
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)

    test:
      suffix: shiftinblocks2
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftinblocks.out

    test:
      suffix: shiftinblockssbaij
      args: -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS -mat_type sbaij
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex30_shiftinblocks.out

TEST*/
