
static char help[] = "KSP linear solver on an operator with a null space.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;         /* KSP context */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3],its,i1,i2;
  PetscScalar    none = -1.0,value[3],avalue;
  PetscReal      norm;
  PC             pc;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* create a solution that is orthogonal to the constants */
  CHKERRQ(VecGetOwnershipRange(u,&i1,&i2));
  for (i=i1; i<i2; i++) {
    avalue = i;
    VecSetValues(u,1,&i,&avalue,INSERT_VALUES);
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecSum(u,&avalue));
  avalue = -avalue/(PetscReal)n;
  CHKERRQ(VecShift(u,avalue));

  /* Create and assemble matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1; value[1] = 1.0;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 1.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatMult(A,u,b));

  /* Create KSP context; set operators and options; solve linear system */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /* Insure that preconditioner has same null space as matrix */
  /* currently does not do anything */
  CHKERRQ(KSPGetPC(ksp,&pc));

  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));
  /* CHKERRQ(KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Check error */
  CHKERRQ(VecAXPY(x,none,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* Free work space */
  CHKERRQ(VecDestroy(&x));CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}
