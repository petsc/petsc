#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.21 1997/04/10 00:05:05 bsmith Exp balay $";
#endif

static char help[] = "Demonstrates us of color map\n";

#include "petsc.h"
#include <math.h>

int main(int argc,char **argv)
{
  Draw draw;
  int     ierr, x = 0, y = 0, width = 256, height = 256,i; 

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  for ( i=0; i<256; i++) {
    ierr = DrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i);
  }
  ierr = DrawFlush(draw); CHKERRA(ierr);
  PetscSleep(2);
  ierr = DrawDestroy(draw); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
