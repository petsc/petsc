
static char help[] = "Sends a PETSc vector to a socket connection, receives it back, within a loop. Works with ex42.c.\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            b; 
  PetscViewer    fd;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscInitialize(&argc,&args,(char *)0,help);
  /* server indicates we WAIT for someone to connect to our socket */
  ierr = PetscViewerSocketOpen(PETSC_COMM_WORLD,"server",PETSC_DEFAULT,&fd);CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,10000,PETSC_DECIDE,&b);CHKERRQ(ierr);
  for (i=0;i<1000;i++){
    ierr = VecView(b,fd);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

