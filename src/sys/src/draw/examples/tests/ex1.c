#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.24 1999/01/12 23:16:53 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates opening and drawing a window\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw draw;
  int  ierr, x = 0, y = 0, width = 300, height = 300;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw); CHKERRA(ierr);
  ierr = DrawSetFromOptions(draw); CHKERRA(ierr);
  ierr = DrawSetViewPort(draw,.25,.25,.75,.75); CHKERRA(ierr);
  ierr = DrawLine(draw,0.0,0.0,1.0,1.0,DRAW_BLACK); CHKERRA(ierr);
  ierr = DrawString(draw,.2,.2,DRAW_RED,"Some Text"); CHKERRA(ierr);
  ierr = DrawStringSetSize(draw,.5,.5); CHKERRA(ierr);
  ierr = DrawString(draw,.2,.2,DRAW_BLUE,"Some Text"); CHKERRA(ierr);
  ierr = DrawFlush(draw); CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawClear(draw); CHKERRA(ierr); ierr = DrawFlush(draw); CHKERRA(ierr);
  ierr = DrawResizeWindow(draw,600,600); CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawLine(draw,0.0,1.0,1.0,0.0,DRAW_BLUE);
  ierr = DrawFlush(draw); CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawDestroy(draw); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
