
static char help[] = "Reads a PETSc matrix and vector from a socket connection,  solves a linear system and sends the result back.\n";

/*T
   Concepts: KSP^solving a linear system
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;            /* matrix */
  Vec            x,b;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;               /* viewer */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  fd = PETSC_VIEWER_SOCKET_WORLD;

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecLoad(b,fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(VecDuplicate(b,&x));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(VecView(x,fd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: defined(PETSC_USE_SOCKET_VIEWER)

     test:
       TODO: Need to figure out how to test examples that use sockets

TEST*/
