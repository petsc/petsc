/*$Id: ex4.c,v 1.9 2001/01/22 23:01:52 bsmith Exp balay $*/

static char help[] = "Demonstrates use of PetscDrawZoom()\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "zoomfunction"
int zoomfunction(PetscDraw draw,void *dummy)
{
  int  ierr,i; 

  for (i=0; i<256; i++) {
    ierr = PetscDrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscDraw draw;
  int  ierr,x = 0,y = 0,width = 256,height = 256; 

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,zoomfunction,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

 
