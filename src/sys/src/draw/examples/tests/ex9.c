/*$Id: ex9.c,v 1.7 1999/11/24 21:52:55 bsmith Exp bsmith $*/

static char help[] = "Makes a simple histogram\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Draw       draw;
  DrawHG     hist;
  DrawAxis   axis;
  int        n = 20,i,ierr,x = 0,y = 0,width = 300,height = 300,bins = 8;
  int        color = DRAW_GREEN;
  char       *xlabel,*ylabel,*toplabel;
  double     xd;
  PetscTruth flg;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-width",&width,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-height",&height,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-bins",&bins,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-color",&color,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-nolabels",&flg);CHKERRA(ierr);
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  /* ierr = DrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);*/
  ierr = DrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = DrawSetType(draw,DRAW_X);CHKERRA(ierr);
  ierr = DrawHGCreate(draw,bins,&hist);CHKERRA(ierr);
  ierr = DrawHGGetAxis(hist,&axis);CHKERRA(ierr);
  ierr = DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE);CHKERRA(ierr);
  ierr = DrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRA(ierr);

  for (i=0; i<n ; i++) {
    xd = (double)(i - 5);
    ierr = DrawHGAddValue(hist,xd*xd);CHKERRA(ierr);
  }
  ierr = DrawHGSetColor(hist,color);CHKERRA(ierr);
  ierr = DrawHGDraw(hist);CHKERRA(ierr);
  ierr = DrawFlush(draw);CHKERRA(ierr);

  ierr = DrawHGDestroy(hist);CHKERRA(ierr);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
