
static char help[] = "Solve a small system and a large system through preloading\n\
  Input arguments are:\n\
  -permute <natural,rcm,nd,...> : solve system in permuted indexing\n\
  -f0 <small_sys_binary> -f1 <large_sys_binary> \n\n";

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

PetscErrorCode CheckResult(KSP *ksp, Mat *A, Vec *b, Vec *x, IS *rowperm)
{
  PetscReal         norm;        /* norm of solution error */
  PetscInt          its;
  PetscFunctionBegin;
  PetscCall(KSPGetTotalIterations(*ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its));

  PetscCall(KSPGetResidualNorm(*ksp,&norm));
  if (norm < 1.e-12) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-12\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %e\n",(double)norm));
  }

  PetscCall(KSPDestroy(ksp));
  PetscCall(MatDestroy(A));
  PetscCall(VecDestroy(x));
  PetscCall(VecDestroy(b));
  PetscCall(ISDestroy(rowperm));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSystem(const char filename[PETSC_MAX_PATH_LEN], RHSType rhstype, MatOrderingType ordering, PetscBool permute, IS *rowperm_out, Mat *A_out, Vec *b_out, Vec *x_out)
{

  Vec               x,b,b2;
  Mat               A;           /* linear system matrix */
  PetscViewer       viewer;      /* viewer */
  PetscBool         same;
  PetscInt          j,len,start,idx,n1,n2;
  const PetscScalar *val;
  IS                rowperm=NULL,colperm=NULL;

  PetscFunctionBegin;
  /* open binary file. Note that we use FILE_MODE_READ to indicate reading from this file */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));

  /* load the matrix and vector; then destroy the viewer */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,viewer));
  switch (rhstype) {
  case RHS_FILE:
    /* Vectors in the file might a different size than the matrix so we need a
     * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
     * can create the correct size Vec. */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
    PetscCall(VecLoad(b,viewer));
    break;
  case RHS_ONE:
    PetscCall(MatCreateVecs(A,&b,NULL));
    PetscCall(VecSet(b,1.0));
    break;
  case RHS_RANDOM:
    PetscCall(MatCreateVecs(A,&b,NULL));
    PetscCall(VecSetRandom(b,NULL));
    break;
  }
  PetscCall(PetscViewerDestroy(&viewer));

  /* if the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector
   */
  PetscCall(MatGetLocalSize(A,NULL,&n1));
  PetscCall(VecGetLocalSize(b,&n2));
  same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PETSC_COMM_WORLD));

  if (!same) { /* create a new vector b by padding the old one */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&b2));
    PetscCall(VecSetSizes(b2,n1,PETSC_DECIDE));
    PetscCall(VecSetFromOptions(b2));
    PetscCall(VecGetOwnershipRange(b,&start,NULL));
    PetscCall(VecGetLocalSize(b,&len));
    PetscCall(VecGetArrayRead(b,&val));
    for (j=0; j<len; j++) {
      idx = start+j;
      PetscCall(VecSetValues(b2,1,&idx,val+j,INSERT_VALUES));
    }
    PetscCall(VecRestoreArrayRead(b,&val));
    PetscCall(VecDestroy(&b));
    PetscCall(VecAssemblyBegin(b2));
    PetscCall(VecAssemblyEnd(b2));
    b    = b2;
  }
  PetscCall(VecDuplicate(b,&x));

  if (permute) {
    Mat Aperm;
    PetscCall(MatGetOrdering(A,ordering,&rowperm,&colperm));
    PetscCall(MatPermute(A,rowperm,colperm,&Aperm));
    PetscCall(VecPermute(b,colperm,PETSC_FALSE));
    PetscCall(MatDestroy(&A));
    A    = Aperm;               /* Replace original operator with permuted version */
    PetscCall(ISDestroy(&colperm));
  }

  *b_out = b;
  *x_out = x;
  *A_out = A;
  *rowperm_out = rowperm;

  PetscFunctionReturn(0);
}

/* ATTENTION: this is the example used in the Profiling chaper of the PETSc manual,
   where we referenced its profiling stages, preloading and output etc.
   When you modify it, please make sure it is still consistent with the manual.
 */
int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  Vec               x,b;
  Mat               A;           /* linear system matrix */
  KSP               ksp;         /* Krylov subspace method context */
  char              file[2][PETSC_MAX_PATH_LEN],ordering[256]=MATORDERINGRCM;
  RHSType           rhstype = RHS_FILE;
  PetscBool         flg,preload=PETSC_FALSE,trans=PETSC_FALSE,permute=PETSC_FALSE;
  IS                rowperm=NULL;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Preloading example options","");PetscCall(ierr);
  {
    /*
       Determine files from which we read the two linear systems
       (matrix and right-hand-side vector).
    */
    PetscCall(PetscOptionsBool("-trans","Solve transpose system instead","",trans,&trans,&flg));
    PetscCall(PetscOptionsString("-f","First file to load (small system)","",file[0],file[0],sizeof(file[0]),&flg));
    PetscCall(PetscOptionsFList("-permute","Permute matrix and vector to solve in new ordering","",MatOrderingList,ordering,ordering,sizeof(ordering),&permute));

    if (flg) {
      PetscCall(PetscStrcpy(file[1],file[0]));
      preload = PETSC_FALSE;
    } else {
      PetscCall(PetscOptionsString("-f0","First file to load (small system)","",file[0],file[0],sizeof(file[0]),&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
      PetscCall(PetscOptionsString("-f1","Second file to load (larger system)","",file[1],file[1],sizeof(file[1]),&flg));
      if (!flg) preload = PETSC_FALSE;   /* don't bother with second system */
    }

    PetscCall(PetscOptionsEnum("-rhs","Right hand side","",RHSTypes,(PetscEnum)rhstype,(PetscEnum*)&rhstype,NULL));
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

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

  /*=========================
      solve a small system
    =========================*/

  PetscPreLoadBegin(preload,"Load System 0");
  PetscCall(CreateSystem(file[0],rhstype,ordering,permute,&rowperm,&A,&b,&x));

  PetscPreLoadStage("KSPSetUp 0");
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
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

  PetscPreLoadStage("KSPSolve 0");
  if (trans) PetscCall(KSPSolveTranspose(ksp,b,x));
  else       PetscCall(KSPSolve(ksp,b,x));

  if (permute) PetscCall(VecPermute(x,rowperm,PETSC_TRUE));

  PetscCall(CheckResult(&ksp,&A,&b,&x,&rowperm));

  /*=========================
    solve a large system
    =========================*/

  PetscPreLoadStage("Load System 1");

  PetscCall(CreateSystem(file[1],rhstype,ordering,permute,&rowperm,&A,&b,&x));

  PetscPreLoadStage("KSPSetUp 1");
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
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

  PetscPreLoadStage("KSPSolve 1");
  if (trans) PetscCall(KSPSolveTranspose(ksp,b,x));
  else       PetscCall(KSPSolve(ksp,b,x));

  if (permute) PetscCall(VecPermute(x,rowperm,PETSC_TRUE));

  PetscCall(CheckResult(&ksp,&A,&b,&x,&rowperm));

  PetscPreLoadEnd();
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 1
      nsize: 4
      output_file: output/ex10_1.out
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 2
      nsize: 4
      output_file: output/ex10_2.out
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi -trans

   test:
      suffix: 3
      requires: double complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/nh-complex-int32-float64 -ksp_type bicg

   test:
      suffix: 4
      args: -f ${DATAFILESPATH}/matrices/medium -ksp_type bicg -permute rcm
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
