

static char help[] = "Example demonstrating opening and drawing a window\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  DrawCtx draw;
  int     ierr, x = 0, y = 0, width = 300, height = 300;
 

  OptionsCreate(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,help);

  ierr = DrawOpenX(0,"Window Title",x,y,width,height,&draw); CHKERR(ierr);
  ierr = DrawSetViewPort(draw,.25,.25,.75,.75); CHKERR(ierr);
  ierr = DrawLine(draw,0.0,0.0,1.0,1.0,DRAW_BLACK,DRAW_BLACK);
  ierr = DrawText(draw,.2,.2,DRAW_RED,"Some Text");
  ierr = DrawTextSetSize(draw,.5,.5);
  ierr = DrawText(draw,.2,.2,DRAW_BLUE,"Some Text");
  ierr = DrawFlush(draw);
  sleep(5);
  ierr = DrawClear(draw);  ierr = DrawFlush(draw);
  sleep(5);
  ierr = DrawLine(draw,0.0,1.0,1.0,0.0,DRAW_BLUE,DRAW_BLUE);
  ierr = DrawFlush(draw);
  sleep(5);
  PetscFinalize();
  return 0;
}
 
