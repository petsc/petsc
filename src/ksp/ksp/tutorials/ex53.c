
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* Create vectors.*/
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&x2);CHKERRQ(ierr);

  /* Create matrix. Only proc[0] sets values - not efficient for parallel processing!
     See ex23.c for efficient parallel assembly matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  if (rank == 0) {
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    i    = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

    i    = 0; col[0] = n-1; value[0] = 0.5; /* make A non-symmetric */
    ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Set exact solution */
  ierr = VecSet(u,one);CHKERRQ(ierr);

  /* Create linear solver context */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  if (size > 1) {
    ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  }
#endif
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* 1. Solve linear system A x = b */
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"1. Norm of error for Ax=b: %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }

  /* 2. Solve linear system A^T x = b*/
  ierr = MatMultTranspose(A,u,b);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ksp,b,x2);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x2,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x2,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"2. Norm of error for A^T x=b: %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }

  /* 3. Change A and solve A x = b with an iterative solver using A=LU as a preconditioner*/
  if (rank == 0) {
    i    = 0; col[0] = n-1; value[0] = 1.e-2;
    ierr = MatSetValues(A,1,&i,1,col,value,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"3. Norm of error for (A+Delta) x=b: %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }

  /* Free work space. */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

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
