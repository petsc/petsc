
static char help[] = "Reads a PETSc matrix and vector from a socket connection,  solves a linear system and sends the result back.\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  KSP         ksp;  /* linear solver context */
  Mat         A;    /* matrix */
  Vec         x, b; /* approx solution, RHS, exact solution */
  PetscViewer fd;   /* viewer */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  fd = PETSC_VIEWER_SOCKET_WORLD;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
  PetscCall(VecLoad(b, fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(VecDuplicate(b, &x));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecView(x, fd));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: defined(PETSC_USE_SOCKET_VIEWER)

     test:
       TODO: Need to figure out how to test examples that use sockets

TEST*/
