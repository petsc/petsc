
static char help[] = "Sends a PETSc vector to a socket connection, receives it back, within a loop. Works with ex42.c.\n";

#include <petscvec.h>

int main(int argc,char **args)
{
  Vec            b;
  PetscViewer    fd;
  PetscInt       i;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* server indicates we WAIT for someone to connect to our socket */
  CHKERRQ(PetscViewerSocketOpen(PETSC_COMM_WORLD,"server",PETSC_DEFAULT,&fd));

  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,10000,PETSC_DECIDE,&b));
  for (i=0; i<1000; i++) {
    CHKERRQ(VecView(b,fd));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecLoad(b,fd));
  }
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscFinalize());
  return 0;
}
