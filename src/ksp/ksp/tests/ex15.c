
static char help[] = "KSP linear solver on an operator with a null space.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;         /* KSP context */
  PetscInt       i,n = 10,col[3],its,i1,i2;
  PetscScalar    none = -1.0,value[3],avalue;
  PetscReal      norm;
  PC             pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));

  /* create a solution that is orthogonal to the constants */
  PetscCall(VecGetOwnershipRange(u,&i1,&i2));
  for (i=i1; i<i2; i++) {
    avalue = i;
    VecSetValues(u,1,&i,&avalue,INSERT_VALUES);
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecSum(u,&avalue));
  avalue = -avalue/(PetscReal)n;
  PetscCall(VecShift(u,avalue));

  /* Create and assemble matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1; value[1] = 1.0;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 1.0; value[1] = -1.0;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatMult(A,u,b));

  /* Create KSP context; set operators and options; solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));

  /* Insure that preconditioner has same null space as matrix */
  /* currently does not do anything */
  PetscCall(KSPGetPC(ksp,&pc));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  /* PetscCall(KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Check error */
  PetscCall(VecAXPY(x,none,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %" PetscInt_FMT "\n",(double)norm,its));

  /* Free work space */
  PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}
