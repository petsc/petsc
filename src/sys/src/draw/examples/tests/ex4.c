/*$Id: ex4.c,v 1.4 1999/10/24 14:01:18 bsmith Exp bsmith $*/

static char help[] = "Demonstrates use of DrawZoom()\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "zoomfunction"
int zoomfunction(Draw draw,void *dummy)
{
  int  ierr, i; 

  for ( i=0; i<256; i++) {
    ierr = DrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw draw;
  int  ierr, x = 0, y = 0, width = 256, height = 256; 

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* ierr = DrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);*/
  ierr = DrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = DrawSetFromOptions(draw);CHKERRA(ierr);
  ierr = DrawZoom(draw,zoomfunction,PETSC_NULL);CHKERRA(ierr);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

 
