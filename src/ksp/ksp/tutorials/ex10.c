
static char help[] = "Solve a small system and a large system through preloading\n\
  Input arguments are:\n\
   -f0 <small_sys_binary> -f1 <large_sys_binary> \n\n";

/*T
   Concepts: KSP^basic parallel example
   Concepts: Mat^loading a binary matrix and vector;
   Concepts: PetscLog^preloading executable
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

typedef enum {
  RHS_FILE,
  RHS_ONE,
  RHS_RANDOM
} RHSType;
const char *const RHSTypes[] = {"FILE", "ONE", "RANDOM", "RHSType", "RHS_", NULL};

/* ATTENTION: this is the example used in the Profiling chaper of the PETSc manual,
   where we referenced its profiling stages, preloading and output etc.
   When you modify it, please make sure it is still consistent with the manual.
 */
int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  Vec               x,b,b2;
  Mat               A;           /* linear system matrix */
  KSP               ksp;         /* Krylov subspace method context */
  PetscReal         norm;        /* norm of solution error */
  char              file[2][PETSC_MAX_PATH_LEN];
  PetscViewer       viewer;      /* viewer */
  PetscBool         flg,preload=PETSC_FALSE,same,trans=PETSC_FALSE;
  RHSType           rhstype = RHS_FILE;
  PetscInt          its,j,len,start,idx,n1,n2;
  const PetscScalar *val;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-trans",&trans,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr    = PetscStrcpy(file[1],file[0]);CHKERRQ(ierr);
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
    ierr = PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg);CHKERRQ(ierr);
    if (!flg) preload = PETSC_FALSE;   /* don't bother with second system */
  }
  ierr = PetscOptionsGetEnum(NULL,NULL,"-rhs",RHSTypes,(PetscEnum*)&rhstype,NULL);CHKERRQ(ierr);

  /*
    To use preloading, one usually has code like the following:

    PetscPreLoadBegin(preload,"first stage);
      lines of code
    PetscPreLoadStage("second stage");
      lines of code
    PetscPreLoadEnd();

    The two macro PetscPreLoadBegin() and PetscPreLoadEnd() implicitly form a
    loop with maximal two iterations, depending whether preloading is turned on or
    not. If it is, either through the preload arg of PetscPreLoadBegin or through
    -preload command line, the trip count is 2, otherwise it is 1. One can use the
    predefined variable PetscPreLoadIt within the loop body to get the current
    iteration number, which is 0 or 1. If preload is turned on, the runtime doesn't
    do profiling for the first iteration, but it will do profiling for the second
    iteration instead.

    One can solve a small system in the first iteration and a large system in
    the second iteration. This process preloads the instructions with the small
    system so that more accurate performance monitoring (via -log_view) can be done
    with the large one (that actually is the system of interest).

    But in this example, we turned off preloading and duplicated the code for
    the large system. In general, it is a bad practice and one should not duplicate
    code. We do that because we want to show profiling stages for both the small
    system and the large system.
  */
  PetscPreLoadBegin(preload,"Load System 0");

  /*=========================
      solve a small system
    =========================*/

  /* open binary file. Note that we use FILE_MODE_READ to indicate reading from this file */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  /* load the matrix and vector; then destroy the viewer */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  switch (rhstype) {
  case RHS_FILE:
    /* Vectors in the file might a different size than the matrix so we need a
     * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
     * can create the correct size Vec. */
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecLoad(b,viewer);CHKERRQ(ierr);
    break;
  case RHS_ONE:
    ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
    break;
  case RHS_RANDOM:
    ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
    break;
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* if the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector
   */
  ierr = MatGetLocalSize(A,NULL,&n1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(b,&n2);CHKERRQ(ierr);
  same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PETSC_COMM_WORLD);CHKERRMPI(ierr);

  if (!same) { /* create a new vector b by padding the old one */
    ierr = VecCreate(PETSC_COMM_WORLD,&b2);CHKERRQ(ierr);
    ierr = VecSetSizes(b2,n1,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b2);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,NULL);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&len);CHKERRQ(ierr);
    ierr = VecGetArrayRead(b,&val);CHKERRQ(ierr);
    for (j=0; j<len; j++) {
      idx = start+j;
      ierr = VecSetValues(b2,1,&idx,val+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(b,&val);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b2);CHKERRQ(ierr);
    b    = b2;
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  PetscPreLoadStage("KSPSetUp 0");
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /*
    Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
    enable more precise profiling of setting up the preconditioner.
    These calls are optional, since both will be called within
    KSPSolve() if they haven't been called already.
  */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  PetscPreLoadStage("KSPSolve 0");
  if (trans) {ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);}
  else       {ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);}

  ierr = KSPGetTotalIterations(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its);CHKERRQ(ierr);

  ierr = KSPGetResidualNorm(ksp,&norm);CHKERRQ(ierr);
  if (norm < 1.e-12) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-12\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %e\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  /*=========================
    solve a large system
    =========================*/
  /* the code is duplicated. Bad practice. See comments above */
  PetscPreLoadStage("Load System 1");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  /* load the matrix and vector; then destroy the viewer */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  switch (rhstype) {
  case RHS_FILE:
    /* Vectors in the file might a different size than the matrix so we need a
     * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
     * can create the correct size Vec. */
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecLoad(b,viewer);CHKERRQ(ierr);
    break;
  case RHS_ONE:
    ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
    break;
  case RHS_RANDOM:
    ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
    break;
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatGetLocalSize(A,NULL,&n1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(b,&n2);CHKERRQ(ierr);
  same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PETSC_COMM_WORLD);CHKERRMPI(ierr);

  if (!same) { /* create a new vector b by padding the old one */
    ierr = VecCreate(PETSC_COMM_WORLD,&b2);CHKERRQ(ierr);
    ierr = VecSetSizes(b2,n1,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b2);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,NULL);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&len);CHKERRQ(ierr);
    ierr = VecGetArrayRead(b,&val);CHKERRQ(ierr);
    for (j=0; j<len; j++) {
      idx = start+j;
      ierr = VecSetValues(b2,1,&idx,val+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(b,&val);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b2);CHKERRQ(ierr);
    b    = b2;
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  PetscPreLoadStage("KSPSetUp 1");
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  /*
    Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
    enable more precise profiling of setting up the preconditioner.
    These calls are optional, since both will be called within
    KSPSolve() if they haven't been called already.
  */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  PetscPreLoadStage("KSPSolve 1");
  if (trans) {ierr = KSPSolveTranspose(ksp,b,x);CHKERRQ(ierr);}
  else       {ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);}

  ierr = KSPGetTotalIterations(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its);CHKERRQ(ierr);

  ierr = KSPGetResidualNorm(ksp,&norm);CHKERRQ(ierr);
  if (norm < 1.e-12) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-12\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %e\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  PetscPreLoadEnd();
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 1
      nsize: 4
      output_file: output/ex10_1.out
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 2
      nsize: 4
      output_file: output/ex10_2.out
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi -trans

TEST*/
