#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.16 1995/10/12 04:19:15 bsmith Exp bsmith $";
#endif

static char help[] = "Example demonstrating color map\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  Draw draw;
  int     ierr, x = 0, y = 0, width = 256, height = 256,i; 

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);

  ierr = DrawOpenX(MPI_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  for ( i=0; i<256; i++) {
    ierr = DrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i);
  }
  ierr = DrawFlush(draw); CHKERRA(ierr);
  PetscSleep(2);
  PetscFinalize();
  return 0;
}
 
