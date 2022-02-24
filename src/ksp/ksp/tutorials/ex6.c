
static char help[] = "Solves a tridiagonal linear system with KSP. \n\
It illustrates how to do one symbolic factorization and multiple numeric factorizations using same matrix structure. \n\n";

#include <petscksp.h>
int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm;         /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,col[3],its,rstart,rend,N=10,num_numfac;
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  /* Create and assemble matrix. */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    if (i == 0) {
      CHKERRQ(MatSetValues(A,1,&i,2,col+1,value+1,INSERT_VALUES));
    } else if (i == N-1) {
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));

  /* Create vectors */
  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* Set exact solution; then compute right-hand-side vector. */
  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(A,u,b));

  /* Create the linear solver and set various options. */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCJACOBI));
  CHKERRQ(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));

  num_numfac = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL));
  while (num_numfac--) {
    /* An example on how to update matrix A for repeated numerical factorization and solve. */
    PetscScalar one=1.0;
    PetscInt    i = 0;
    CHKERRQ(MatSetValues(A,1,&i,1,&i,&one,ADD_VALUES));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    /* Update b */
    CHKERRQ(MatMult(A,u,b));

    /* Solve the linear system */
    CHKERRQ(KSPSolve(ksp,b,x));

    /* Check the solution and clean up */
    CHKERRQ(VecAXPY(x,-1.0,u));
    CHKERRQ(VecNorm(x,NORM_2,&norm));
    CHKERRQ(KSPGetIterationNumber(ksp,&its));
    if (norm > 100*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));
    }
  }

  /* Free work space. */
  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b)); CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -num_numfac 2 -pc_type lu

    test:
      suffix: 2
      args: -num_numfac 2 -pc_type lu -pc_factor_mat_solver_type mumps
      requires: mumps

    test:
      suffix: 3
      nsize: 3
      args: -num_numfac 2 -pc_type lu -pc_factor_mat_solver_type mumps
      requires: mumps

TEST*/
