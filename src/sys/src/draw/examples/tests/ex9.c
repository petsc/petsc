/*$Id: ex9.c,v 1.8 2000/01/11 20:59:19 bsmith Exp bsmith $*/

static char help[] = "Makes a simple histogram\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscDraw       draw;
  PetscDrawHG     hist;
  PetscDrawAxis   axis;
  int        n = 20,i,ierr,x = 0,y = 0,width = 300,height = 300,bins = 8;
  int        color = PETSC_DRAW_GREEN;
  char       *xlabel,*ylabel,*toplabel;
  double     xd;
  PetscTruth flg;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-width",&width,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-height",&height,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bins",&bins,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-color",&color,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nolabels",&flg);CHKERRA(ierr);
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = PetscDrawSetType(draw,PETSC_DRAW_X);CHKERRA(ierr);
  ierr = PetscDrawHGCreate(draw,bins,&hist);CHKERRA(ierr);
  ierr = PetscDrawHGGetAxis(hist,&axis);CHKERRA(ierr);
  ierr = PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE);CHKERRA(ierr);
  ierr = PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRA(ierr);

  for (i=0; i<n ; i++) {
    xd = (double)(i - 5);
    ierr = PetscDrawHGAddValue(hist,xd*xd);CHKERRA(ierr);
  }
  ierr = PetscDrawHGSetColor(hist,color);CHKERRA(ierr);
  ierr = PetscDrawHGDraw(hist);CHKERRA(ierr);
  ierr = PetscDrawFlush(draw);CHKERRA(ierr);

  ierr = PetscDrawHGDestroy(hist);CHKERRA(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
