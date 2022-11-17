
static char help[] = "Sends a PETSc vector to a socket connection, receives it back, within a loop. Must be run with ex42.c.\n";

#include <petscvec.h>

int main(int argc, char **args)
{
  Vec         b;
  PetscViewer fd;
  PetscInt    i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /* server indicates we WAIT for someone to connect to our socket */
  PetscCall(PetscViewerSocketOpen(PETSC_COMM_WORLD, "server", PETSC_DEFAULT, &fd));

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 10000, PETSC_DECIDE, &b));
  for (i = 0; i < 1000; i++) {
    PetscCall(VecView(b, fd));
    PetscCall(VecDestroy(&b));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecLoad(b, fd));
  }
  PetscCall(VecDestroy(&b));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(PetscFinalize());
  return 0;
}
