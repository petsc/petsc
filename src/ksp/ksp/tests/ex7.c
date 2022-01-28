
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
  PetscErrorCode    ierr;
  Mat               P;

  PetscFunctionBegin;
  /* printf("MatShellMult...user should implement this routine without using a matrix\n"); */
  ierr = MatShellGetContext(As,&P);CHKERRQ(ierr);
  ierr = MatMult(P,x,y);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, As x = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /* Create matrix P, to be used as preconditioner */
  ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
  ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = MatSetUp(P);CHKERRQ(ierr);

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(P,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(P,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(P,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Set exact solution */
  ierr = VecSet(u,one);CHKERRQ(ierr);

  /* Create a matrix-free matrix As, P is used as a data context in MyMatShellMult() */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,P,&As);CHKERRQ(ierr);
  ierr = MatSetFromOptions(As);CHKERRQ(ierr);
  ierr = MatShellSetOperation(As,MATOP_MULT,(void(*)(void))MyMatShellMult);CHKERRQ(ierr);

  /* Check As is a linear operator: As*(ax + y) = a As*x + As*y */
  ierr = MatIsLinear(As,10,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Shell matrix As is non-linear! Use '-info |grep MatIsLinear' to get detailed report");

  /* Compute right-hand-side vector. */
  ierr = MatMult(As,u,b);CHKERRQ(ierr);

  ierr = MatSetOption(As,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMultTranspose(As,u,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  PetscAssertFalse(norm > PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^T x||_\\infty: %1.6e",norm);
  ierr = MatSetOption(As,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(As,u,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  PetscAssertFalse(norm > PETSC_SMALL,PetscObjectComm((PetscObject)As),PETSC_ERR_PLIB,"Error ||A x-A^H x||_\\infty: %1.6e",norm);

  /* Create the linear solver and set various options */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,As,P);CHKERRQ(ierr);

  /* Set linear solver defaults for this problem (optional). */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* Set runtime options */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Solve linear system */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  /* Free work space. */
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&As);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

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
