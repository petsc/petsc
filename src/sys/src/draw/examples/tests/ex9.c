/*$Id: ex9.c,v 1.4 1999/05/04 20:28:42 balay Exp bsmith $*/

static char help[] = "Makes a simple histogram\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw     draw;
  DrawHist hist;
  DrawAxis axis;
  int      n = 20, i, ierr, x = 0, y = 0, width = 300, height = 300,flg,bins = 8;
  char     *xlabel, *ylabel, *toplabel;
  double   xd;
  int      color = DRAW_GREEN;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-width",&width,&flg); 
  OptionsGetInt(PETSC_NULL,"-height",&height,&flg);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  OptionsGetInt(PETSC_NULL,"-bins",&bins,&flg); 
  OptionsGetInt(PETSC_NULL,"-color",&color,&flg); 
  OptionsHasName(PETSC_NULL,"-nolabels",&flg); 
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  ierr = DrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = DrawHistCreate(draw,bins,&hist);CHKERRA(ierr);
  ierr = DrawHistGetAxis(hist,&axis);CHKERRA(ierr);
  ierr = DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE);CHKERRA(ierr);
  ierr = DrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRA(ierr);

  for ( i=0; i<n ; i++ ) {
    xd = (double)( i - 5 );
    ierr = DrawHistAddValue(hist,xd*xd);CHKERRA(ierr);
  }
  ierr = DrawHistSetColor(hist,color);CHKERRA(ierr);
  ierr = DrawHistDraw(hist);CHKERRA(ierr);
  ierr = DrawFlush(draw);CHKERRA(ierr);

  ierr = DrawHistDestroy(hist);CHKERRA(ierr);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
