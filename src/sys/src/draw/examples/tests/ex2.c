#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.22 1995/09/30 19:26:45 bsmith Exp $";
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
  DrawCtx draw;
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
 
