
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
It is copied and intended to move dirty codes from ksp/examples/tutorials/ex10.c and simplify ex10.c.\n\
  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n\
  -trans  : solve transpose system instead\n\n";
/*
  This code  can be used to test PETSc interface to other packages.\n\
  Examples of command line options:       \n\
   ex10 -f0 <datafile> -ksp_type preonly  \n\
        -help -ksp_view                  \n\
        -num_numfac <num_numfac> -num_rhs <num_rhs> \n\
        -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package spooles or superlu or superlu_dist or mumps \n\
        -ksp_type preonly -pc_type cholesky -pc_factor_mat_solver_package spooles or dscpack or mumps \n\
   mpiexec -n <np> ex10 -f0 <datafile> -ksp_type cg -pc_type asm -pc_asm_type basic -sub_pc_type icc -mat_type sbaij
 \n\n";
*/
/*T
   Concepts: KSP solving a linear system
   Processors: n
T*/

#include "petscksp.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A,B;            /* matrix */
  Vec            x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;               /* viewer */
  char           file[4][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscTruth     table = PETSC_FALSE,flg,flgB=PETSC_FALSE,trans=PETSC_FALSE,partition=PETSC_FALSE,initialguess = PETSC_FALSE;
  PetscTruth     outputSoln=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       its,num_numfac,n,M;
  PetscReal      norm;
  PetscLogDouble tsetup,tsetup1,tsetup2,tsolve,tsolve1,tsolve2;
  PetscTruth     preload=PETSC_TRUE,diagonalscale,isSymmetric,cknorm=PETSC_FALSE,Test_MatDuplicate=PETSC_FALSE;
  PetscMPIInt    rank;
  PetscScalar    sigma;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-table",&table,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-trans",&trans,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-partition",&partition,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-initialguess",&initialguess,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-output_solution",&outputSoln,PETSC_NULL);CHKERRQ(ierr);

  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file[0],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcpy(file[1],file[0]);CHKERRQ(ierr);
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(1,"Must indicate binary file with the -f0 or -f option");
    ierr = PetscOptionsGetString(PETSC_NULL,"-f1",file[1],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (!flg) {preload = PETSC_FALSE;} /* don't bother with second system */
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
  PreLoadBegin(preload,"Load system");

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Load system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* 
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
    */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PreLoadIt],FILE_MODE_READ,&fd);CHKERRQ(ierr);
    
    /*
       Load the matrix and vector; then destroy the viewer.
    */
    ierr = MatLoad(fd,MATAIJ,&A);CHKERRQ(ierr);
    
    if (!preload){
      flg = PETSC_FALSE;
      ierr = PetscOptionsGetString(PETSC_NULL,"-rhs",file[2],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
      if (flg){ /* rhs is stored in a separate file */
        ierr = PetscViewerDestroy(fd);CHKERRQ(ierr); 
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
      }
    }
    /* if file contains no RHS, then use a vector of all ones */
      PetscInt    m;
      PetscScalar one = 1.0;
      ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
      ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
      ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(b);CHKERRQ(ierr);
      ierr = VecSet(b,one);CHKERRQ(ierr);
    
    ierr = PetscViewerDestroy(fd);CHKERRQ(ierr); 

    /* Test MatDuplicate() */
    if (Test_MatDuplicate){
      ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
      ierr = MatEqual(A,B,&flg);CHKERRQ(ierr);
      if (!flg){
        PetscPrintf(PETSC_COMM_WORLD,"  A != B \n");CHKERRQ(ierr);
      } 
      ierr = MatDestroy(B);CHKERRQ(ierr); 
    }

    /* Add a shift to A */
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-mat_sigma",&sigma,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscOptionsGetString(PETSC_NULL,"-fB",file[2],PETSC_MAX_PATH_LEN-1,&flgB);CHKERRQ(ierr);
      if (flgB){
        /* load B to get A = A + sigma*B */
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
        ierr  = MatLoad(fd,MATAIJ,&B);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
        ierr = MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* A <- sigma*B + A */  
      } else {
        ierr = MatShift(A,sigma);CHKERRQ(ierr); 
      }
    }

    /* Make A singular for testing zero-pivot of ilu factorization        */
    /* Example: ./ex10 -f0 <datafile> -test_zeropivot -set_row_zero -pc_factor_shift_nonzero */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-test_zeropivot", &flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscInt          row,ncols;
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscTruth        flg1=PETSC_FALSE;
      PetscScalar       *zeros;
      row = 0;      
      ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);     
      ierr = PetscMalloc(sizeof(PetscScalar)*(ncols+1),&zeros);
      ierr = PetscMemzero(zeros,(ncols+1)*sizeof(PetscScalar));CHKERRQ(ierr);
      flg1 = PETSC_FALSE;
      ierr = PetscOptionsGetTruth(PETSC_NULL, "-set_row_zero", &flg1,PETSC_NULL);CHKERRQ(ierr);
      if (flg1){ /* set entire row as zero */
        ierr = MatSetValues(A,1,&row,ncols,cols,zeros,INSERT_VALUES);CHKERRQ(ierr);
      } else { /* only set (row,row) entry as zero */
        ierr = MatSetValues(A,1,&row,1,&row,zeros,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }

    /* Check whether A is symmetric */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL, "-check_symmetry", &flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      Mat Atrans;
      ierr = MatTranspose(A, MAT_INITIAL_MATRIX,&Atrans);
      ierr = MatEqual(A, Atrans, &isSymmetric);
      if (isSymmetric) {
        PetscPrintf(PETSC_COMM_WORLD,"A is symmetric \n");CHKERRQ(ierr);
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"A is non-symmetric \n");CHKERRQ(ierr);
      }
      ierr = MatDestroy(Atrans);CHKERRQ(ierr);
    }

    /* 
       If the loaded matrix is larger than the vector (due to being padded 
       to match the block size of the system), then create a new padded vector.
    */
    
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    if (m != n) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
    }
    ierr = MatGetSize(A,&M,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetSize(b,&m);CHKERRQ(ierr);
    if (M != m) { /* Create a new vector b by padding the old one */
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
        ierr  = VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
      ierr = VecDestroy(b);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
      b = tmp;
    }
    ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
    ierr = VecSet(x,0.0);CHKERRQ(ierr);

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
      ierr = PetscMalloc(size*sizeof(PetscInt),&count);CHKERRQ(ierr);
      ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &mpart);CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency(mpart, A);CHKERRQ(ierr);
      /* ierr = MatPartitioningSetVertexWeights(mpart, weight);CHKERRQ(ierr); */
      ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
      ierr = MatPartitioningApply(mpart, &mis);CHKERRQ(ierr);
      ierr = MatPartitioningDestroy(mpart);CHKERRQ(ierr);
      ierr = ISPartitioningToNumbering(mis,&nis);CHKERRQ(ierr);
      ierr = ISPartitioningCount(mis,size,count);CHKERRQ(ierr);
      ierr = ISDestroy(mis);CHKERRQ(ierr);
      ierr = ISInvertPermutation(nis, count[rank], &is);CHKERRQ(ierr);
      ierr = PetscFree(count);CHKERRQ(ierr);
      ierr = ISDestroy(nis);CHKERRQ(ierr);
      ierr = ISSort(is);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB);CHKERRQ(ierr);

      /* need to move the vector also */
      ierr = ISDestroy(is);CHKERRQ(ierr);
      ierr = MatDestroy(A);CHKERRQ(ierr);
      A    = BB;
    }

    /*
       We also explicitly time this stage via PetscGetTime()
    */
    ierr = PetscGetTime(&tsetup1);CHKERRQ(ierr);

    /*
       Create linear solver; set operators; set runtime options.
    */
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,initialguess);CHKERRQ(ierr);
    num_numfac = 1;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-num_numfac",&num_numfac,PETSC_NULL);CHKERRQ(ierr);
    while ( num_numfac-- ){
     

      ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

      /* 
       Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       KSPSolve() if they haven't been called already.
      */
      ierr = KSPSetUp(ksp);CHKERRQ(ierr);
      ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
      ierr = PetscGetTime(&tsetup2);CHKERRQ(ierr);
      tsetup = tsetup2 - tsetup1;

      /*
       Tests "diagonal-scaling of preconditioned residual norm" as used 
       by many ODE integrator codes including SUNDIALS. Note this is different
       than diagonally scaling the matrix before computing the preconditioner
      */
      diagonalscale = PETSC_FALSE;
      ierr = PetscOptionsGetTruth(PETSC_NULL,"-diagonal_scale",&diagonalscale,PETSC_NULL);CHKERRQ(ierr);
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
        ierr = PCDiagonalScaleSet(pc,scale);CHKERRQ(ierr);
        ierr = VecDestroy(scale);CHKERRQ(ierr);
      }

      /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Solve system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      /*
       Solve linear system; we also explicitly time this stage.
      */
      ierr = PetscGetTime(&tsolve1);CHKERRQ(ierr);
      if (trans) {
        ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
      } else {
        PetscInt  num_rhs=1;
        ierr = PetscOptionsGetInt(PETSC_NULL,"-num_rhs",&num_rhs,PETSC_NULL);CHKERRQ(ierr);
        cknorm = PETSC_FALSE;
        ierr = PetscOptionsGetTruth(PETSC_NULL,"-cknorm",&cknorm,PETSC_NULL);CHKERRQ(ierr);
        while ( num_rhs-- ) {
          ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
        }
        ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
        if (cknorm){   /* Check error for each rhs */
          if (trans) {
            ierr = MatMultTranspose(A,x,u);CHKERRQ(ierr);
          } else {
            ierr = MatMult(A,x,u);CHKERRQ(ierr);
          }
          ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
          ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of iterations = %3D\n",its);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"  Residual norm %A\n",norm);CHKERRQ(ierr);
        }
      } /* while ( num_rhs-- ) */
      ierr = PetscGetTime(&tsolve2);CHKERRQ(ierr);
      tsolve = tsolve2 - tsolve1;

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
        ierr = PetscStrrchr(file[PreLoadIt],'/',&matrixname);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %2.1e %2.1e %2.1e %s \n",
                matrixname,its,norm,tsetup+tsolve,tsetup,tsolve,kspinfo);CHKERRQ(ierr);

        /*
          Destroy the viewer
        */
        ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %A\n",norm);CHKERRQ(ierr);
      }
      ierr = PetscOptionsGetString(PETSC_NULL,"-solution",file[3],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
      if (flg) {
        PetscViewer viewer;
        Vec         xstar;
        PetscReal   norm;

        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = VecLoad(viewer, VECMPI, &xstar);CHKERRQ(ierr);
        ierr = VecAXPY(xstar, -1.0, x);CHKERRQ(ierr);
        ierr = VecNorm(xstar, NORM_2, &norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Error norm %A\n", norm);CHKERRQ(ierr);
        ierr = VecDestroy(xstar);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      }
      if (outputSoln) {
        PetscViewer viewer;

        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution.petsc",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = VecView(x, viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      }

      flg  = PETSC_FALSE;
      ierr = PetscOptionsGetTruth(PETSC_NULL, "-ksp_reason", &flg,PETSC_NULL);CHKERRQ(ierr);
      if (flg){
        KSPConvergedReason reason;
        ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason); 
      }
       
    } /* while ( num_numfac-- ) */

    /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = MatDestroy(A);CHKERRQ(ierr); ierr = VecDestroy(b);CHKERRQ(ierr);
    ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = KSPDestroy(ksp);CHKERRQ(ierr); 
    if (flgB) { ierr = MatDestroy(B);CHKERRQ(ierr); }
  PreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

