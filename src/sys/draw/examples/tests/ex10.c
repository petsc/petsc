
static char help[] = "Tests repeatedly setting a window type.\n";

#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  int            x = 0,y = 0,width = 300,height = 300;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
#if defined (PETSC_HAVE_X11)
  ierr = PetscDrawSetType(draw,"x");CHKERRQ(ierr);
  ierr = PetscDrawSetType(draw,"null");CHKERRQ(ierr);
  ierr = PetscDrawSetType(draw,"x");CHKERRQ(ierr);
#else
  ierr = PetscDrawSetType(draw,"null");CHKERRQ(ierr);
#endif
  ierr = PetscDrawSetViewPort(draw,.25,.25,.75,.75);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawStringSetSize(draw,.5,.5);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_BLUE,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr); ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawResizeWindow(draw,600,600);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
