
static char help[] = "Reads a PETSc vector from a socket connection, then sends it back within a loop. Works with ex42.m or ex42a.c\n";

#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            b;       
  PetscViewer    fd;               /* viewer */
  PetscErrorCode ierr;
  PetscInt       i;

  PetscInitialize(&argc,&args,(char *)0,help);
  fd = PETSC_VIEWER_SOCKET_WORLD;

  for (i=0;i<1000;i++){
    ierr = VecLoad(fd,VECMPI,&b);CHKERRQ(ierr);
    ierr = VecView(b,fd);CHKERRQ(ierr);
    ierr = VecDestroy(b);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

