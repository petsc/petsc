/*$Id: ex4.c,v 1.6 2000/01/11 20:59:19 bsmith Exp bsmith $*/

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

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRA(ierr);
  ierr = PetscDrawZoom(draw,zoomfunction,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

 
