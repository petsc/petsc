

static char help[] = "Example demonstrating opening and drawing a window\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "petsc.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  DrawCtx draw;
  int     ierr, x = 0, y = 0, width = 300, height = 300;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,help);

  ierr = DrawOpenX(MPI_COMM_WORLD,0,"Window Title",x,y,width,height,&draw);
  CHKERRA(ierr);
  ierr = DrawSetViewPort(draw,.25,.25,.75,.75); CHKERRA(ierr);
  ierr = DrawLine(draw,0.0,0.0,1.0,1.0,DRAW_BLACK);
  ierr = DrawText(draw,.2,.2,DRAW_RED,"Some Text");
  ierr = DrawTextSetSize(draw,.5,.5);
  ierr = DrawText(draw,.2,.2,DRAW_BLUE,"Some Text");
  ierr = DrawFlush(draw);
  sleep(2);
  ierr = DrawClear(draw);  ierr = DrawFlush(draw);
  sleep(2);
  ierr = DrawLine(draw,0.0,1.0,1.0,0.0,DRAW_BLUE);
  ierr = DrawFlush(draw);
  sleep(2);
  PetscFinalize();
  return 0;
}
 
