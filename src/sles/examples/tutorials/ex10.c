/*$Id: ex10.c,v 1.58 2001/09/11 16:33:29 bsmith Exp $*/

static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This version first preloads and solves a small system, then loads \n\
another (larger) system and solves it as well.  This example illustrates\n\
preloading of instructions with the smaller system so that more accurate\n\
performance monitoring can be done with the larger one (that actually\n\
is the system of interest).  See the 'Performance Hints' chapter of the\n\
users manual for a discussion of preloading.  Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n\
  -trans  : solve transpose system instead\n\n";
/*
  This code can be used to test PETSc interface to other packages.\n\
  Examples of command line options:       \n\
   ex10 -f0 <datafile> -ksp_type preonly  \n\
        -help -sles_view                  \n\
        -num_numfac <num_numfac> -num_rhs <num_rhs> \n\
        -ksp_type preonly -pc_type lu -matload_type seqaijspooles/superlu/superlu_dist/aijmumps \n\
        -ksp_type preonly -pc_type cholesky -matload_type seqsbaijspooles/dscpack/sbaijmumps    \n\n";
*/
/*T
   Concepts: SLES^solving a linear system
   Processors: n
T*/

/* 
  Include "petscsles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscsles.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  SLES           sles;             /* linear solver context */
  Mat            A;                /* matrix */
  Vec            x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;               /* viewer */
  char           file[2][128];     /* input file name */
  PetscTruth     table,flg,trans,partition = PETSC_FALSE;
  int            ierr,its,ierrp;
  PetscReal      norm;
  PetscLogDouble tsetup,tsetup1,tsetup2,tsolve,tsolve1,tsolve2;
  PetscScalar    zero = 0.0,none = -1.0;
  PetscTruth     preload = PETSC_TRUE,diagonalscale,hasNullSpace,isSymmetric;
  int            num_numfac;
  PetscScalar    sigma;
  KSP            ksp;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = PetscOptionsHasName(PETSC_NULL,"-table",&table);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-trans",&trans);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-partition",&partition);CHKERRQ(ierr);

  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file[0],127,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcpy(file[1],file[0]);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(1,"Must indicate binary file with the -f0 or -f option");
    ierr = PetscOptionsGetString(PETSC_NULL,"-f1",file[1],127,&flg);CHKERRQ(ierr);
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
       Open binary file.  Note that we use PETSC_FILE_RDONLY to indicate
       reading from this file.
    */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PreLoadIt],PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);

    /*
       Load the matrix and vector; then destroy the viewer.
    */
    ierr  = MatLoad(fd,MATAIJ,&A);CHKERRQ(ierr);
    ierr  = PetscPushErrorHandler(PetscIgnoreErrorHandler,PETSC_NULL);CHKERRQ(ierr);
    ierrp = VecLoad(fd,&b);
    ierr  = PetscPopErrorHandler();CHKERRQ(ierr);
    if (ierrp) { /* if file contains no RHS, then use a vector of all ones */
      int         m;
      PetscScalar one = 1.0;
      PetscLogInfo(0,"Using vector of ones for RHS\n");
      ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
      ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(b);CHKERRQ(ierr);
      ierr = VecSet(&one,b);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

    /* Add a shift to A */
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-mat_sigma",&sigma,&flg);CHKERRQ(ierr);
    if(flg) {ierr = MatShift(&sigma,A);CHKERRQ(ierr); }

    /* Check whether A is symmetric */
    ierr = PetscOptionsHasName(PETSC_NULL, "-check_symmetry", &flg);CHKERRQ(ierr);
    if (flg) {
      Mat Atrans;
      ierr = MatTranspose(A, &Atrans);
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
    { 
      int         m,n,j,mvec,start,end,indx;
      Vec         tmp;
      PetscScalar *bold;

      /* Create a new vector b by padding the old one */
      ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
      ierr = VecSetSizes(tmp,m,PETSC_DECIDE);CHKERRQ(ierr);
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
    ierr = VecSet(&zero,x);CHKERRQ(ierr);

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                      Setup solve for system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


    if (partition) {
      MatPartitioning mpart;
      IS              mis,nis,isn,is;
      int             *count,size,rank;
      Mat             B;
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
      ierr = PetscMalloc(size*sizeof(int),&count);CHKERRQ(ierr);
      ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &mpart);CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency(mpart, A);CHKERRQ(ierr);
      /* ierr = MatPartitioningSetVertexWeights(mpart, weight);CHKERRQ(ierr); */
      ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
      ierr = MatPartitioningApply(mpart, &mis);CHKERRQ(ierr);
      ierr = MatPartitioningDestroy(mpart);CHKERRQ(ierr);
      ierr = ISPartitioningToNumbering(mis,&nis);CHKERRQ(ierr);
      ierr = ISPartitioningCount(mis,count);CHKERRQ(ierr);
      ierr = ISDestroy(mis);CHKERRQ(ierr);
      ierr = ISInvertPermutation(nis, count[rank], &is);CHKERRQ(ierr);
      ierr = PetscFree(count);CHKERRQ(ierr);
      ierr = ISDestroy(nis);CHKERRQ(ierr);
      ierr = ISSort(is);CHKERRQ(ierr);
      ierr = ISAllGather(is,&isn);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(A,is,isn,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

      /* need to move the vector also */
      ierr = ISDestroy(is);CHKERRQ(ierr);
      ierr = ISDestroy(isn);CHKERRQ(ierr);
      ierr = MatDestroy(A);CHKERRQ(ierr);
      A    = B;
    }
 
    /*
       Conclude profiling last stage; begin profiling next stage.
    */
    PreLoadStage("SLESSetUp");

    /*
       We also explicitly time this stage via PetscGetTime()
    */
    ierr = PetscGetTime(&tsetup1);CHKERRQ(ierr);

    /*
       Create linear solver; set operators; set runtime options.
    */
    ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRQ(ierr);

    num_numfac = 1;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-num_numfac",&num_numfac,PETSC_NULL);CHKERRQ(ierr);
    while ( num_numfac-- ){
      /* ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); */
    ierr = SLESSetOperators(sles,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);

    /* 
       Here we explicitly call SLESSetUp() and KSPSetUpOnBlocks() to
       enable more precise profiling of setting up the preconditioner.
       These calls are optional, since both will be called within
       SLESSolve() if they haven't been called already.
    */
    ierr = SLESSetUp(sles,b,x);CHKERRQ(ierr);
    ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
    ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
    ierr = PetscGetTime(&tsetup2);CHKERRQ(ierr);
    tsetup = tsetup2 - tsetup1;

    /*
      Test MatGetInertia()
      Usage:
      ex10 -f0 <mat_binaryfile> -ksp_type preonly -pc_type cholesky -mat_type seqsbaij -test_inertia -mat_sigma <sigma>
     */
    ierr = PetscOptionsHasName(PETSC_NULL,"-test_inertia",&flg);CHKERRQ(ierr);
    if (flg){
      PC      pc;
      int     nneg, nzero, npos;
      Mat     F;
      
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCGetFactoredMatrix(pc,&F);CHKERRQ(ierr);
      ierr = MatGetInertia(F,&nneg,&nzero,&npos);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF," MatInertia: nneg: %d, nzero: %d, npos: %d\n",nneg,nzero,npos);
    }

    /*
       Tests "diagonal-scaling of preconditioned residual norm" as used 
       by many ODE integrator codes including PVODE. Note this is different
       than diagonally scaling the matrix before computing the preconditioner
    */
    ierr = PetscOptionsHasName(PETSC_NULL,"-diagonal_scale",&diagonalscale);CHKERRQ(ierr);
    if (diagonalscale) {
      PC     pc;
      int    j,start,end,n;
      Vec    scale;
      
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

    ierr = PetscOptionsHasName(PETSC_NULL, "-null_space", &hasNullSpace);CHKERRQ(ierr);
    if (hasNullSpace == PETSC_TRUE) {
      MatNullSpace nullSpace;
      PC           pc;

      ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, 1, 0, PETSC_NULL, &nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullSpace, A);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCNullSpaceAttach(pc, nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(nullSpace);CHKERRQ(ierr); 
    }

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Solve system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling next stage
    */
    PreLoadStage("SLESSolve");

    /*
       Solve linear system; we also explicitly time this stage.
    */
    ierr = PetscGetTime(&tsolve1);CHKERRQ(ierr);
    if (trans) {
      ierr = SLESSolveTranspose(sles,b,x);CHKERRQ(ierr);
    } else {
      int  num_rhs=1;
      ierr = PetscOptionsGetInt(PETSC_NULL,"-num_rhs",&num_rhs,PETSC_NULL);CHKERRQ(ierr);
      while ( num_rhs-- ) {
        ierr = SLESSolve(sles,b,x);CHKERRQ(ierr);
      }
    }
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    ierr = PetscGetTime(&tsolve2);CHKERRQ(ierr);
    tsolve = tsolve2 - tsolve1;

   /* 
       Conclude profiling this stage
    */
    PreLoadStage("Cleanup");

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
    ierr = VecAXPY(&none,b,u);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);

    /*
       Write output (optinally using table for solver details).
        - PetscPrintf() handles output for multiprocessor jobs 
          by printing from only one processor in the communicator.
        - SLESView() prints information about the linear solver.
    */
    if (table) {
      char        *matrixname,slesinfo[120];
      PetscViewer viewer;

      /*
         Open a string viewer; then write info to it.
      */
      ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,slesinfo,120,&viewer);CHKERRQ(ierr);
      ierr = SLESView(sles,viewer);CHKERRQ(ierr);
      ierr = PetscStrrchr(file[PreLoadIt],'/',&matrixname);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3d %2.0e %2.1e %2.1e %2.1e %s \n",
                matrixname,its,norm,tsetup+tsolve,tsetup,tsolve,slesinfo);CHKERRQ(ierr);

      /*
         Destroy the viewer
      */
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %A\n",norm);CHKERRQ(ierr);
    }

    ierr = PetscOptionsHasName(PETSC_NULL, "-ksp_reason", &flg);CHKERRQ(ierr);
    if (flg){
      KSP ksp;
      KSPConvergedReason reason;
      ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %d\n", reason); 
    }
       
    } /* while ( num_numfac-- ) */

    /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = MatDestroy(A);CHKERRQ(ierr); ierr = VecDestroy(b);CHKERRQ(ierr);
    ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = SLESDestroy(sles);CHKERRQ(ierr); 
  PreLoadEnd();
  /* -----------------------------------------------------------
                      End of linear solver loop
     ----------------------------------------------------------- */

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

