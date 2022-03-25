
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
  PetscInt       i,col[3],its,rstart,rend,N=10,num_numfac;
  PetscScalar    value[3];

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  /* Create and assemble matrix. */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    if (i == 0) {
      PetscCall(MatSetValues(A,1,&i,2,col+1,value+1,INSERT_VALUES));
    } else if (i == N-1) {
      PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    } else {
      PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));

  /* Create vectors */
  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(VecDuplicate(x,&u));

  /* Set exact solution; then compute right-hand-side vector. */
  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(A,u,b));

  /* Create the linear solver and set various options. */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCJACOBI));
  PetscCall(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));

  num_numfac = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-num_numfac",&num_numfac,NULL));
  while (num_numfac--) {
    /* An example on how to update matrix A for repeated numerical factorization and solve. */
    PetscScalar one=1.0;
    PetscInt    i = 0;
    PetscCall(MatSetValues(A,1,&i,1,&i,&one,ADD_VALUES));
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    /* Update b */
    PetscCall(MatMult(A,u,b));

    /* Solve the linear system */
    PetscCall(KSPSolve(ksp,b,x));

    /* Check the solution and clean up */
    PetscCall(VecAXPY(x,-1.0,u));
    PetscCall(VecNorm(x,NORM_2,&norm));
    PetscCall(KSPGetIterationNumber(ksp,&its));
    if (norm > 100*PETSC_MACHINE_EPSILON) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));
    }
  }

  /* Free work space. */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
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
