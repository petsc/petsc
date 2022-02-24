
static char help[] = "Solves a linear system in parallel with KSP. \n\
Contributed by Jose E. Roman, SLEPc developer, for testing repeated call of KSPSetOperators(), 2014 \n\n";

#include <petscksp.h>
int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscScalar    v;
  PC             pc;
  PetscInt       in;
  Mat            F,B;
  PetscBool      solve=PETSC_FALSE,sameA=PETSC_FALSE,setfromoptions_first=PETSC_FALSE;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
#if !defined(PETSC_HAVE_MUMPS)
  PetscMPIInt    size;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  CHKERRQ(PetscLogStageRegister("Assembly", &stage));
  CHKERRQ(PetscLogStagePush(stage));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogStagePop());

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* Create parallel vectors. */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,m*n));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(b,&x));

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    CHKERRQ(PetscRandomSetFromOptions(rctx));
    CHKERRQ(VecSetRandom(u,rctx));
    CHKERRQ(PetscRandomDestroy(&rctx));
  } else {
    CHKERRQ(VecSet(u,1.0));
  }
  CHKERRQ(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create linear solver context */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /* Set operators. */
  CHKERRQ(KSPSetOperators(ksp,A,A));

  CHKERRQ(KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-setfromoptions_first",&setfromoptions_first,NULL));
  if (setfromoptions_first) {
    /* code path for changing from KSPLSQR to KSPREONLY */
    CHKERRQ(KSPSetFromOptions(ksp));
  }
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));
#if defined(PETSC_HAVE_MUMPS)
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Spectrum slicing with MUMPS is not available for complex scalars");
#endif
  CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));
  /*
     must use runtime option '-mat_mumps_icntl_13 1' (turn off ScaLAPACK for
     matrix inertia), currently there is no better way of setting this in program
  */
  CHKERRQ(PetscOptionsInsertString(NULL,"-mat_mumps_icntl_13 1"));
#else
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size>1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Configure with MUMPS if you want to run this example in parallel");
#endif

  if (!setfromoptions_first) {
    /* when -setfromoptions_first is true, do not call KSPSetFromOptions() again and stick to KSPPREONLY */
    CHKERRQ(KSPSetFromOptions(ksp));
  }

  /* get inertia */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-solve",&solve,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sameA",&sameA,NULL));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatGetInertia(F,&in,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"INERTIA=%D\n",in));
  if (solve) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solving the intermediate KSP\n"));
    CHKERRQ(KSPSolve(ksp,b,x));
  } else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"NOT Solving the intermediate KSP\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  if (sameA) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setting A\n"));
    CHKERRQ(MatAXPY(A,1.1,B,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(KSPSetOperators(ksp,A,A));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setting B\n"));
    CHKERRQ(MatAXPY(B,1.1,A,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(KSPSetOperators(ksp,B,B));
  }
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatGetInertia(F,&in,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"INERTIA=%D\n",in));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(MatDestroy(&B));

  /* Free work space.*/
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
      requires: !complex

    test:
      args:

    test:
      suffix: 2
      args: -sameA

    test:
      suffix: 3
      args: -ksp_lsqr_monitor -ksp_type lsqr -setfromoptions_first {{0 1}separate output}

TEST*/
