
static char help[] = "Solves a tridiagonal linear system with KSP. \n\
                      Modified from ex1.c to illustrate reuse of preconditioner \n\
                      Written as requested by [petsc-maint #63875] \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,x2,b,u;     /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm,tol=100.*PETSC_MACHINE_EPSILON; /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3],its;
  PetscMPIInt    rank,size;
  PetscScalar    one = 1.0,value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create vectors.*/
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "Solution"));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));
  CHKERRQ(VecDuplicate(x,&x2));

  /* Create matrix. Only proc[0] sets values - not efficient for parallel processing!
     See ex23.c for efficient parallel assembly matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  if (rank == 0) {
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    }
    i    = n - 1; col[0] = n - 2; col[1] = n - 1;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));

    i    = 0; col[0] = n-1; value[0] = 0.5; /* make A non-symmetric */
    CHKERRQ(MatSetValues(A,1,&i,1,col,value,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Set exact solution */
  CHKERRQ(VecSet(u,one));

  /* Create linear solver context */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
#if defined(PETSC_HAVE_MUMPS)
  if (size > 1) {
    CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));
  }
#endif
  CHKERRQ(KSPSetFromOptions(ksp));

  /* 1. Solve linear system A x = b */
  CHKERRQ(MatMult(A,u,b));
  CHKERRQ(KSPSolve(ksp,b,x));

  /* Check the error */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"1. Norm of error for Ax=b: %g, Iterations %D\n",(double)norm,its));
  }

  /* 2. Solve linear system A^T x = b*/
  CHKERRQ(MatMultTranspose(A,u,b));
  CHKERRQ(KSPSolveTranspose(ksp,b,x2));

  /* Check the error */
  CHKERRQ(VecAXPY(x2,-1.0,u));
  CHKERRQ(VecNorm(x2,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2. Norm of error for A^T x=b: %g, Iterations %D\n",(double)norm,its));
  }

  /* 3. Change A and solve A x = b with an iterative solver using A=LU as a preconditioner*/
  if (rank == 0) {
    i    = 0; col[0] = n-1; value[0] = 1.e-2;
    CHKERRQ(MatSetValues(A,1,&i,1,col,value,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatMult(A,u,b));
  CHKERRQ(KSPSolve(ksp,b,x));

  /* Check the error */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"3. Norm of error for (A+Delta) x=b: %g, Iterations %D\n",(double)norm,its));
  }

  /* Free work space. */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x2));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: mumps

   test:
      suffix: 2
      nsize: 2
      requires: mumps
      output_file: output/ex53.out

TEST*/
