#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.18 1995/10/12 04:19:15 bsmith Exp bsmith $";
#endif

static char help[] = "Plots a simple line graph\n";

#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  Draw     draw;
  DrawLG   lg;
  DrawAxis axis;
  int         n = 20,i, ierr, x = 0, y = 0, width = 300, height = 300;
  char        *xlabel,*ylabel,*toplabel;
  double      xd,yd;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  OptionsGetInt(0,"-width",&width); OptionsGetInt(0,"-height",&height);
  OptionsGetInt(0,"-n",&n);
  if (OptionsHasName(0,"-nolabels")) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  ierr = DrawOpenX(MPI_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = DrawLGCreate(draw,1,&lg); CHKERRA(ierr);
  ierr = DrawLGGetAxis(lg,&axis); CHKERRA(ierr);
  ierr = DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE); CHKERRA(ierr);
  ierr = DrawAxisSetLabels(axis,toplabel,xlabel,ylabel); CHKERRA(ierr);

  for ( i=0; i<n ; i++ ) {
    xd = (double)( i - 5 ); yd = xd*xd;
    ierr = DrawLGAddPoint(lg,&xd,&yd); CHKERRA(ierr);
  }
  ierr = DrawLGIndicateDataPoints(lg); CHKERRA(ierr);
  ierr = DrawLGDraw(lg); CHKERRA(ierr);
  ierr = DrawFlush(draw); CHKERRA(ierr); PetscSleep(2);

  ierr = DrawLGDestroy(lg); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
