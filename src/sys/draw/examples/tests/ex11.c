
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",0,0,256,256,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);

  ierr = PetscDrawBoxedString(draw,.5,.5,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"Greetings",PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscDrawBoxedString(draw,.25,.75,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"How\nare\nyou?",PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscDrawBoxedString(draw,.25,.25,PETSC_DRAW_GREEN,PETSC_DRAW_RED,"Long line followed by a very\nshort line",PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

