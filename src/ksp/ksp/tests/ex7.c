
static char help[] = "Illustrate how to solves a matrix-free linear system with KSP.\n\n";

/*
  Note: modified from ~src/ksp/ksp/tutorials/ex1.c
*/
#include <petscksp.h>

/*
   MatShellMult - Computes the matrix-vector product, y = As x.

   Input Parameters:
   As - the matrix-free matrix
   x  - vector

   Output Parameter:
   y - vector
 */
PetscErrorCode MyMatShellMult(Mat As,Vec x,Vec y)
{
  Mat               P;

  PetscFunctionBegin;
  /* printf("MatShellMult...user should implement this routine without using a matrix\n"); */
  PetscCall(MatShellGetContext(As,&P));
  PetscCall(MatMult(P,x,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            P,As;         /* preconditioner matrix, linear system (matrix-free) */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm;         /* norm of solution error */
  PetscInt       i,n = 100,col[3],its;
  PetscMPIInt    size;
  PetscScalar    one = 1.0,value[3];
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, As x = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(PetscObjectSetName((PetscObject) x, "Solution"));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));

  /* Create matrix P, to be used as preconditioner */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&P));
  PetscCall(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(P));
  PetscCall(MatSetUp(P));

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(P,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  PetscCall(MatSetValues(P,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  PetscCall(MatSetValues(P,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

  /* Set exact solution */
  PetscCall(VecSet(u,one));

  /* Create a matrix-free matrix As, P is used as a data context in MyMatShellMult() */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,P,&As));
  PetscCall(MatSetFromOptions(As));
  PetscCall(MatShellSetOperation(As,MATOP_MULT,(void(*)(void))MyMatShellMult));

  /* Check As is a linear operator: As*(ax + y) = a As*x + As*y */
  PetscCall(MatIsLinear(As,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Shell matrix As is non-linear! Use '-info |grep MatIsLinear' to get detailed report");

  /* Compute right-hand-side vector. */
  PetscCall(MatMult(As,u,b));

  PetscCall(MatSetOption(As,MAT_SYMMETRIC,PETSC_TRUE));
  PetscCall(MatMultTranspose(As,u,x));
  PetscCall(VecAXPY(x,-1.0,b));
  PetscCall(VecNorm(x,NORM_INFINITY,&norm));
  PetscCheck(norm <= PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^T x||_\\infty: %1.6e",(double)norm);
  PetscCall(MatSetOption(As,MAT_HERMITIAN,PETSC_TRUE));
  PetscCall(MatMultHermitianTranspose(As,u,x));
  PetscCall(VecAXPY(x,-1.0,b));
  PetscCall(VecNorm(x,NORM_INFINITY,&norm));
  PetscCheck(norm <= PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^H x||_\\infty: %1.6e",(double)norm);

  /* Create the linear solver and set various options */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,As,P));

  /* Set linear solver defaults for this problem (optional). */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /* Set runtime options */
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve linear system */
  PetscCall(KSPSolve(ksp,b,x));

  /* Check the error */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %" PetscInt_FMT "\n",(double)norm,its));

  /* Free work space. */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&As));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_monitor_short -ksp_max_it 10
   test:
      suffix: 2
      args: -ksp_monitor_short -ksp_max_it 10

TEST*/
