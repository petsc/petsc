#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.1 1999/08/31 21:09:56 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates us of color map\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "zoomfunction"
int zoomfunction(Draw draw,void *dummy)
{
  int  ierr, i; 

  for ( i=0; i<256; i++) {
    ierr = DrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i);
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw draw;
  int  ierr, x = 0, y = 0, width = 256, height = 256,i; 

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = DrawZoom(draw,zoomfunction,PETSC_NULL);CHKERRA(ierr);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

 
