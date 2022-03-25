
static char help[] = "Test PC redistribute on matrix with load imbalance. \n\
                      Modified from src/ksp/ksp/tutorials/ex2.c.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -n <mesh_y>       : number of mesh points\n\n";
/*
Example:
  mpiexec -n 8 ./ex3 -n 10000 -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_rtol 1.e-8 -log_view
  mpiexec -n 8 ./ex3 -n 10000 -ksp_type preonly -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc -redistribute_ksp_rtol 1.e-8 -log_view
*/

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m,n = 7,its,nloc,matdistribute=0;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscScalar    v;
  PetscMPIInt    rank,size;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheckFalse(size < 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example requires at least 2 MPI processes!");

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-matdistribute",&matdistribute,NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  switch(matdistribute) {
  case 1: /* very imbalanced process load for matrix A */
    m    = (1+size)*size;
    nloc = (rank+1)*n;
    if (rank == size-1) { /* proc[size-1] stores all remaining rows */
      nloc = m*n;
      for (i=0; i<size-1; i++) {
        nloc -= (i+1)*n;
      }
    }
    break;
  default: /* proc[0] and proc[1] load much smaller row blocks, the rest processes have same loads */
    if (rank == 0 || rank == 1) {
      nloc = n;
    } else {
      nloc = 10*n; /* 10x larger load */
    }
    m = 2 + (size-2)*10;
    break;
  }
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,nloc,nloc,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  nloc = Iend-Istart;
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] A Istart,Iend: %D %D; nloc %D\n",rank,Istart,Iend,nloc));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  PetscCall(PetscLogStageRegister("Assembly", &stage));
  PetscCall(PetscLogStagePush(stage));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* Create parallel vectors. */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,nloc,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(b,&x));

  /* Set exact solution; then compute right-hand-side vector. */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL));
  if (flg) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(u,rctx));
    PetscCall(PetscRandomDestroy(&rctx));
  } else {
    PetscCall(VecSet(u,1.0));
  }
  PetscCall(MatMult(A,u,b));

  /* View the exact solution vector if desired */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL));
  if (flg) PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),PETSC_DEFAULT,PETSC_DEFAULT,
                          PETSC_DEFAULT);PetscCall(ierr);
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));

  /* Free work space. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -n 100 -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_rtol 1.e-8

   test:
      suffix: 2
      nsize: 8
      args: -n 100 -ksp_type preonly -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc -redistribute_ksp_rtol 1.e-8

TEST*/
