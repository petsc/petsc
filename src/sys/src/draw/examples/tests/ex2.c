/*$Id: ex2.c,v 1.30 2001/01/15 21:43:34 bsmith Exp bsmith $*/

static char help[] = "Demonstrates us of color map\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscDraw draw;
  int     ierr,x = 0,y = 0,width = 256,height = 256,i; 

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  for (i=0; i<256; i++) {
    ierr = PetscDrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i);
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
