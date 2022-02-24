
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
  CHKERRQ(MatShellGetContext(As,&P));
  CHKERRQ(MatMult(P,x,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            P,As;         /* preconditioner matrix, linear system (matrix-free) */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm;         /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 100,col[3],its;
  PetscMPIInt    size;
  PetscScalar    one = 1.0,value[3];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, As x = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "Solution"));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* Create matrix P, to be used as preconditioner */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P));
  CHKERRQ(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(P));
  CHKERRQ(MatSetUp(P));

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(P,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  CHKERRQ(MatSetValues(P,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(P,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

  /* Set exact solution */
  CHKERRQ(VecSet(u,one));

  /* Create a matrix-free matrix As, P is used as a data context in MyMatShellMult() */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,P,&As));
  CHKERRQ(MatSetFromOptions(As));
  CHKERRQ(MatShellSetOperation(As,MATOP_MULT,(void(*)(void))MyMatShellMult));

  /* Check As is a linear operator: As*(ax + y) = a As*x + As*y */
  CHKERRQ(MatIsLinear(As,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Shell matrix As is non-linear! Use '-info |grep MatIsLinear' to get detailed report");

  /* Compute right-hand-side vector. */
  CHKERRQ(MatMult(As,u,b));

  CHKERRQ(MatSetOption(As,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatMultTranspose(As,u,x));
  CHKERRQ(VecAXPY(x,-1.0,b));
  CHKERRQ(VecNorm(x,NORM_INFINITY,&norm));
  PetscCheckFalse(norm > PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^T x||_\\infty: %1.6e",norm);
  CHKERRQ(MatSetOption(As,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(MatMultHermitianTranspose(As,u,x));
  CHKERRQ(VecAXPY(x,-1.0,b));
  CHKERRQ(VecNorm(x,NORM_INFINITY,&norm));
  PetscCheckFalse(norm > PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^H x||_\\infty: %1.6e",norm);

  /* Create the linear solver and set various options */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,As,P));

  /* Set linear solver defaults for this problem (optional). */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /* Set runtime options */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* Solve linear system */
  CHKERRQ(KSPSolve(ksp,b,x));

  /* Check the error */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* Free work space. */
  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b)); CHKERRQ(MatDestroy(&P));
  CHKERRQ(MatDestroy(&As));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -ksp_monitor_short -ksp_max_it 10
   test:
      suffix: 2
      args: -ksp_monitor_short -ksp_max_it 10

TEST*/
