/*$Id: ex10.c,v 1.10 2000/01/11 20:59:19 bsmith Exp bsmith $*/
static char help[] = "Tests repeatedly setting a window type\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscDraw draw;
  int  ierr,x = 0,y = 0,width = 300,height = 300;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
#if defined (PETSC_HAVE_X11)
  ierr = PetscDrawSetType(draw,"x");CHKERRA(ierr);
  ierr = PetscDrawSetType(draw,"null");CHKERRA(ierr);
  ierr = PetscDrawSetType(draw,"x");CHKERRA(ierr);
#else
  ierr = PetscDrawSetType(draw,"null");CHKERRA(ierr);
#endif
  ierr = PetscDrawSetViewPort(draw,.25,.25,.75,.75);CHKERRA(ierr);
  ierr = PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK);CHKERRA(ierr);
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text");CHKERRA(ierr);
  ierr = PetscDrawStringSetSize(draw,.5,.5);CHKERRA(ierr);
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_BLUE,"Some Text");CHKERRA(ierr);
  ierr = PetscDrawFlush(draw);CHKERRA(ierr);
  ierr = PetscSleep(2);CHKERRA(ierr);
  ierr = PetscDrawClear(draw);CHKERRA(ierr); ierr = PetscDrawFlush(draw);CHKERRA(ierr);
  ierr = PetscDrawResizeWindow(draw,600,600);CHKERRA(ierr);
  ierr = PetscSleep(2);CHKERRA(ierr);
  ierr = PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE);
  ierr = PetscDrawFlush(draw);CHKERRA(ierr);
  ierr = PetscSleep(2);CHKERRA(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
