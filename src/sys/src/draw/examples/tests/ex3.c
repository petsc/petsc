

static char help[] = "Example demonstrating opening and drawing a window\n";

#include "petsc.h"
#include "comm.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  DrawCtx     draw;
  DrawAxisCtx axis;
  int         ierr, x = 0, y = 0, width = 300, height = 300;
  double      xlow = 0.0,ylow = 0.0, xhigh = 10.0, yhigh = 1.0;
  char        *xlabel,*ylabel,*toplabel;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  OptionsCreate(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);
  OptionsGetInt(0,"-width",&width);
  OptionsGetInt(0,"-height",&height);
  OptionsGetDouble(0,"-xlow",&xlow);
  OptionsGetDouble(0,"-xhigh",&xhigh);
  OptionsGetDouble(0,"-ylow",&ylow);
  OptionsGetDouble(0,"-yhigh",&yhigh);
  if (OptionsHasName(0,"-nolabels")) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  ierr = DrawOpenX(0,"Window Title",x,y,width,height,&draw); CHKERR(ierr);
  ierr = DrawAxisCreate(draw,&axis); CHKERR(ierr);
  ierr = DrawAxisSetLimits(axis,xlow,xhigh,ylow,yhigh); CHKERR(ierr);
  ierr = DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE);
  ierr = DrawAxisSetLabels(axis,toplabel,xlabel,ylabel);
  ierr = DrawAxis(axis); CHKERR(ierr);
  ierr = DrawFlush(draw);
  sleep(500);

  PetscFinalize();
  return 0;
}
 
