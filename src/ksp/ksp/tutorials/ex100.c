#include <petscksp.h>

/* ------------------------------------------------------- */

PetscErrorCode RunTest(void)
{
  PetscInt       N    = 100, its = 0;
  PetscBool      draw = PETSC_FALSE, test = PETSC_FALSE;
  PetscReal      rnorm;
  Mat            A;
  Vec            b,x,r;
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw",&draw,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATPYTHON);CHKERRQ(ierr);
  ierr = MatPythonSetType(A,"example100.py:Laplace1D");CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecSet(b,1);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPYTHON);CHKERRQ(ierr);
  ierr = KSPPythonSetType(ksp,"example100.py:ConjGrad");CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCPYTHON);CHKERRQ(ierr);
  ierr = PCPythonSetType(pc,"example100.py:Jacobi");CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  if (test) {
    ierr = KSPGetTotalIterations(ksp,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of KSP iterations = %D\n", its);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
    ierr = MatMult(A,x,r);CHKERRQ(ierr);
    ierr = VecAYPX(r,-1,b);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"error norm = %g\n",rnorm);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }

  if (draw) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = PetscSleep(2);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------- */

static char help[] = "Python-implemented Mat/KSP/PC.\n\n";

/*
#define PYTHON_EXE "python2.5"
#define PYTHON_LIB "/usr/lib/libpython2.5"
*/

#if !defined(PYTHON_EXE)
#define PYTHON_EXE 0
#endif
#if !defined(PYTHON_LIB)
#define PYTHON_LIB 0
#endif

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = PetscPythonInitialize(PYTHON_EXE,PYTHON_LIB);CHKERRQ(ierr);
  ierr = RunTest();PetscPythonPrintError();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_monitor_short
      requires: petsc4py
      localrunfiles: example100.py

TEST*/
