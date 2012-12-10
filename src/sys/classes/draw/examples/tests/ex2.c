
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  int            x = 0,y = 0,width = 256,height = 256,i;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  for (i=0; i<256; i++) {
    ierr = PetscDrawLine(draw,0.0,((PetscReal)i)/256.,1.0,((PetscReal)i)/256.,i);CHKERRQ(ierr);
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

