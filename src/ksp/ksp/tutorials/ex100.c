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

  PetscFunctionBegin;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-draw",&draw,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetType(A,MATPYTHON));
  CHKERRQ(MatPythonSetType(A,"example100.py:Laplace1D"));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecSet(b,1));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPYTHON));
  CHKERRQ(KSPPythonSetType(ksp,"example100.py:ConjGrad"));

  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCPYTHON));
  CHKERRQ(PCPythonSetType(pc,"example100.py:Jacobi"));

  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  if (test) {
    CHKERRQ(KSPGetTotalIterations(ksp,&its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of KSP iterations = %D\n", its));
  } else {
    CHKERRQ(VecDuplicate(b,&r));
    CHKERRQ(MatMult(A,x,r));
    CHKERRQ(VecAYPX(r,-1,b));
    CHKERRQ(VecNorm(r,NORM_2,&rnorm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"error norm = %g\n",rnorm));
    CHKERRQ(VecDestroy(&r));
  }

  if (draw) {
    CHKERRQ(VecView(x,PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(PetscSleep(2));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

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

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  CHKERRQ(PetscPythonInitialize(PYTHON_EXE,PYTHON_LIB));
  CHKERRQ(RunTest();PetscPythonPrintError());
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ksp_monitor_short
      requires: petsc4py
      localrunfiles: example100.py

TEST*/
