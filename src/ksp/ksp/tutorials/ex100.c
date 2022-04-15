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

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-draw",&draw,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetType(A,MATPYTHON));
  PetscCall(MatPythonSetType(A,"example100.py:Laplace1D"));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(VecSet(b,1));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPPYTHON));
  PetscCall(KSPPythonSetType(ksp,"example100.py:ConjGrad"));

  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCPYTHON));
  PetscCall(PCPythonSetType(pc,"example100.py:Jacobi"));

  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));

  if (test) {
    PetscCall(KSPGetTotalIterations(ksp,&its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of KSP iterations = %" PetscInt_FMT "\n", its));
  } else {
    PetscCall(VecDuplicate(b,&r));
    PetscCall(MatMult(A,x,r));
    PetscCall(VecAYPX(r,-1,b));
    PetscCall(VecNorm(r,NORM_2,&rnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"error norm = %g\n",(double)rnorm));
    PetscCall(VecDestroy(&r));
  }

  if (draw) {
    PetscCall(VecView(x,PETSC_VIEWER_DRAW_WORLD));
    PetscCall(PetscSleep(2));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

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

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscPythonInitialize(PYTHON_EXE,PYTHON_LIB));
  PetscCall(RunTest();PetscPythonPrintError());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ksp_monitor_short
      requires: petsc4py
      localrunfiles: example100.py

TEST*/
