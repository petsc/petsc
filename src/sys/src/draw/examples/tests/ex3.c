

static char help[] = "Plots a line graph\n";

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
  DrawLGCtx   lg;
  DrawAxisCtx axis;
  int         i, ierr, x = 0, y = 0, width = 300, height = 300;
  char        *xlabel,*ylabel,*toplabel;
  double      xd,yd;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  OptionsCreate(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);
  OptionsGetInt(0,"-width",&width);
  OptionsGetInt(0,"-height",&height);
  if (OptionsHasName(0,"-nolabels")) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  ierr = DrawOpenX(0,"Window Title",x,y,width,height,&draw); CHKERR(ierr);
  ierr = DrawLGCreate(draw,1,&lg); CHKERR(ierr);
  ierr = DrawLGGetAxisCtx(lg,&axis); CHKERR(ierr);
  ierr = DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE);
  ierr = DrawAxisSetLabels(axis,toplabel,xlabel,ylabel);

  for ( i=0; i<20 ; i++ ) {
    xd = (double)( i - 5 ); yd = xd*xd;
    DrawLGAddPoint(lg,&xd,&yd);
  }

  ierr = DrawLG(lg); CHKERR(ierr);
  ierr = DrawFlush(draw);
  sleep(500);

  DrawLGDestroy(lg);
  PetscFinalize();
  return 0;
}
 
