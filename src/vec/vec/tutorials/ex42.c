
static char help[] = "Reads a PETSc vector from a socket connection, then sends it back within a loop 1000 times. Works with ex42.m or ex42a.c\n";

#include <petscvec.h>

int main(int argc, char **args)
{
  Vec         b;
  PetscViewer fd; /* viewer */
  PetscInt    i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  fd = PETSC_VIEWER_SOCKET_WORLD;

  for (i = 0; i < 1000; i++) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecLoad(b, fd));
    PetscCall(VecView(b, fd));
    PetscCall(VecDestroy(&b));
  }
  PetscCall(PetscFinalize());
  return 0;
}
