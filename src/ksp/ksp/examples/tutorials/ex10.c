
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
   ./ex10 -f0 <datafile> -ksp_type preonly  \n\
        -help -ksp_view                  \n\
        -num_numfac <num_numfac> -num_rhs <num_rhs> \n\
        -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu or superlu_dist or mumps \n\
        -ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_package mumps \n\
   mpiexec -n <np> ./ex10 -f0 <datafile> -ksp_type cg -pc_type asm -pc_asm_type basic -sub_pc_type icc -mat_type sbaij
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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;               /* matrix */
  Vec            x,b,u;           /* approx solution, RHS, exact solution */
  PetscViewer    fd;              /* viewer */
  char           file[4][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      table     =PETSC_FALSE,flg,trans=PETSC_FALSE,initialguess = PETSC_FALSE;
  PetscBool      outputSoln=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       its,num_numfac,m,n,M,nearnulldim = 0;
  PetscReal      norm;
  PetscBool      preload=PETSC_TRUE,isSymmetric,cknorm=PETSC_FALSE,initialguessfile = PETSC_FALSE;
  PetscMPIInt    rank;
  char           initialguessfilename[PETSC_MAX_PATH_LEN];

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-trans",&trans,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-initialguess",&initialguess,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-output_solution",&outputSoln,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-initialguessfilename",initialguessfilename,PETSC_MAX_PATH_LEN,&initialguessfile);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nearnulldim",&nearnulldim,NULL);CHKERRQ(ierr);

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
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f0 or -f option");
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
        -log_summary) can be done with the larger one (that actually
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
  if (nearnulldim) {
    MatNullSpace nullsp;
    Vec          *nullvecs;
    PetscInt     i;
    ierr = PetscMalloc1(nearnulldim,&nullvecs);CHKERRQ(ierr);
    for (i=0; i<nearnulldim; i++) {
      ierr = VecCreate(PETSC_COMM_WORLD,&nullvecs[i]);CHKERRQ(ierr);
      ierr = VecLoad(nullvecs[i],fd);CHKERRQ(ierr);
    }
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,nearnulldim,nullvecs,&nullsp);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(A,nullsp);CHKERRQ(ierr);
    for (i=0; i<nearnulldim; i++) {ierr = VecDestroy(&nullvecs[i]);CHKERRQ(ierr);}
    ierr = PetscFree(nullvecs);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetString(NULL,NULL,"-rhs",file[2],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  if (flg) {   /* rhs is stored in a separate file */
    if (file[2][0] == '0' || file[2][0] == 0) {
      PetscInt    m;
      PetscScalar one = 1.0;
      ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
      ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
      ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(b);CHKERRQ(ierr);
      ierr = VecSet(b,one);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = VecSetFromOptions(b);CHKERRQ(ierr);
      ierr = VecLoad(b,fd);CHKERRQ(ierr);
    }
  } else {   /* rhs is stored in the same file as matrix */
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Make A singular for testing zero-pivot of ilu factorization        */
  /* Example: ./ex10 -f0 <datafile> -test_zeropivot -set_row_zero -pc_factor_shift_type <shift_type> */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-test_zeropivot", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscInt          row=0;
    PetscBool         flg1=PETSC_FALSE;
    PetscScalar       zero=0.0;

    ierr = PetscOptionsGetBool(NULL,NULL, "-set_row_zero", &flg1,NULL);CHKERRQ(ierr);
    if (flg1) {   /* set a row as zeros */
      ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatZeroRows(A,1,&row,0.0,NULL,NULL);CHKERRQ(ierr);
    } else if (!rank) {
      /* set (row,row) entry as zero */
      ierr = MatSetValues(A,1,&row,1,&row,&zero,INSERT_VALUES);CHKERRQ(ierr); 
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Check whether A is symmetric */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-check_symmetry", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Mat Atrans;
    ierr = MatTranspose(A, MAT_INITIAL_MATRIX,&Atrans);
    ierr = MatEqual(A, Atrans, &isSymmetric);
    if (isSymmetric) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"Warning: A is non-symmetric \n");CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Atrans);CHKERRQ(ierr);
  }

  /*
     If the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector.
  */

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  /*  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);*/
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = VecGetSize(b,&m);CHKERRQ(ierr);
  if (M != m) {   /* Create a new vector b by padding the old one */
    PetscInt    j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
    ierr = VecSetSizes(tmp,n,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(tmp);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&mvec);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
    for (j=0; j<mvec; j++) {
      indx = start+j;
      ierr = VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
    b    = tmp;
  }

  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  if (initialguessfile) {
    PetscViewer viewer2;
    ierr         = PetscViewerBinaryOpen(PETSC_COMM_WORLD,initialguessfilename,FILE_MODE_READ,&viewer2);CHKERRQ(ierr);
    ierr         = VecLoad(x,viewer2);CHKERRQ(ierr);
    ierr         = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
    initialguess = PETSC_TRUE;
  } else if (initialguess) {
    ierr = VecSet(x,1.0);CHKERRQ(ierr);
  } else {
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
  }


  /* Check scaling in A */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-check_scaling", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Vec       max, min;
    PetscInt  idx;
    PetscReal val;

    ierr = VecDuplicate(x, &max);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &min);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(A, max, NULL);CHKERRQ(ierr);
    ierr = MatGetRowMinAbs(A, min, NULL);CHKERRQ(ierr);
    {
      PetscViewer viewer;

      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "max.data", &viewer);CHKERRQ(ierr);
      ierr = VecView(max, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "min.data", &viewer);CHKERRQ(ierr);
      ierr = VecView(min, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = VecView(max, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecMax(max, &idx, &val);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Largest max row element %g at row %D\n", (double)val, idx);CHKERRQ(ierr);
    ierr = VecView(min, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecMin(min, &idx, &val);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Smallest min row element %g at row %D\n", (double)val, idx);CHKERRQ(ierr);
    ierr = VecMin(max, &idx, &val);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Smallest max row element %g at row %D\n", (double)val, idx);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(max, max, min);CHKERRQ(ierr);
    ierr = VecMax(max, &idx, &val);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Largest row ratio %g at row %D\n", (double)val, idx);CHKERRQ(ierr);
    ierr = VecView(max, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&max);CHKERRQ(ierr);
    ierr = VecDestroy(&min);CHKERRQ(ierr);
  }

  /*  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
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
  ierr       = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr       = KSPSetInitialGuessNonzero(ksp,initialguess);CHKERRQ(ierr);
  num_numfac = 1;
  ierr       = PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL);CHKERRQ(ierr);
  while (num_numfac--) {
    PetscBool lsqr;
    char      str[32];
    ierr = PetscOptionsGetString(NULL,NULL,"-ksp_type",str,32,&lsqr);CHKERRQ(ierr);
    if (lsqr) {
      ierr = PetscStrcmp("lsqr",str,&lsqr);CHKERRQ(ierr);
    }
    if (lsqr) {
      Mat BtB;
      ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,4,&BtB);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,A,BtB);CHKERRQ(ierr);
      ierr = MatDestroy(&BtB);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /*
     Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
     enable more precise profiling of setting up the preconditioner.
     These calls are optional, since both will be called within
     KSPSolve() if they haven't been called already.
    */
    ierr   = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr   = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

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
      ierr   = PetscOptionsGetInt(NULL,NULL,"-num_rhs",&num_rhs,NULL);CHKERRQ(ierr);
      cknorm = PETSC_FALSE;
      ierr   = PetscOptionsGetBool(NULL,NULL,"-cknorm",&cknorm,NULL);CHKERRQ(ierr);
      while (num_rhs--) {
        if (num_rhs == 1) VecSet(x,0.0);
        ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      }
      ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
      if (cknorm) {     /* Check error for each rhs */
        if (trans) {
          ierr = MatMultTranspose(A,x,u);CHKERRQ(ierr);
        } else {
          ierr = MatMult(A,x,u);CHKERRQ(ierr);
        }
        ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
        ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its);CHKERRQ(ierr);
        if (!PetscIsNanScalar(norm)) {
          if (norm < 1.e-12) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm < 1.e-12\n");CHKERRQ(ierr);
          } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %g\n",(double)norm);CHKERRQ(ierr);
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
      ierr = MatMultTranspose(A,x,u);CHKERRQ(ierr);
    } else {
      ierr = MatMult(A,x,u);CHKERRQ(ierr);
    }
    ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
    /*
     Write output (optinally using table for solver details).
      - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
      - KSPView() prints information about the linear solver.
    */
    if (table) {
      char        *matrixname,kspinfo[120];
      PetscViewer viewer;

      /*
       Open a string viewer; then write info to it.
      */
      ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,120,&viewer);CHKERRQ(ierr);
      ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
      ierr = PetscStrrchr(file[PetscPreLoadIt],'/',&matrixname);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n",matrixname,its,norm,kspinfo);CHKERRQ(ierr);

      /*
        Destroy the viewer
      */
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
      if (!PetscIsNanScalar(norm)) {
        if (norm < 1.e-12 && !PetscIsNanScalar((PetscScalar)norm)) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm < 1.e-12\n");CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscOptionsGetString(NULL,NULL,"-solution",file[3],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      Vec         xstar;
      PetscReal   norm;

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&xstar);CHKERRQ(ierr);
      ierr = VecLoad(xstar,viewer);CHKERRQ(ierr);
      ierr = VecAXPY(xstar, -1.0, x);CHKERRQ(ierr);
      ierr = VecNorm(xstar, NORM_2, &norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Error norm %g\n", (double)norm);CHKERRQ(ierr);
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
      PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
    }

  }   /* while (num_numfac--) */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  PetscPreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  ierr = PetscFinalize();
  return 0;
}

