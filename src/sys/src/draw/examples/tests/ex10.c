#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.5 1999/03/26 18:53:52 balay Exp balay $";
#endif

static char help[] = "Tests repeatedly setting a window type\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw draw;
  int  ierr, x = 0, y = 0, width = 300, height = 300;
 
  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
#if defined (HAVE_X11)
  ierr = DrawSetType(draw,"x");CHKERRA(ierr);
  ierr = DrawSetType(draw,"null");CHKERRA(ierr);
  ierr = DrawSetType(draw,"x");CHKERRA(ierr);
#else
  ierr = DrawSetType(draw,"null");CHKERRA(ierr);
#endif
  ierr = DrawSetViewPort(draw,.25,.25,.75,.75);CHKERRA(ierr);
  ierr = DrawLine(draw,0.0,0.0,1.0,1.0,DRAW_BLACK);CHKERRA(ierr);
  ierr = DrawString(draw,.2,.2,DRAW_RED,"Some Text");CHKERRA(ierr);
  ierr = DrawStringSetSize(draw,.5,.5);CHKERRA(ierr);
  ierr = DrawString(draw,.2,.2,DRAW_BLUE,"Some Text");CHKERRA(ierr);
  ierr = DrawFlush(draw);CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawClear(draw);CHKERRA(ierr); ierr = DrawFlush(draw);CHKERRA(ierr);
  ierr = DrawResizeWindow(draw,600,600);CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawLine(draw,0.0,1.0,1.0,0.0,DRAW_BLUE);
  ierr = DrawFlush(draw);CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
